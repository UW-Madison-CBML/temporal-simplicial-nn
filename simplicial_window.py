"""Build the per-window inputs that ``SCCNTemporalClassifier`` / ``SCCNTemporalLayer``
expect, from a :class:`toponetx.SimplicialComplex`.

The model's ``forward`` consumes, per window, a dict with ``features`` / ``adjacencies``
/ ``incidences`` / ``keys``. This module generates that dict and -- crucially -- emits
``keys`` (stable cell identities = sorted vertex tuples) **in the exact row order of the
matrices**, and identically across windows, so the temporal recurrence aligns each cell's
past to its present correctly even as edges/triangles appear and disappear.

Why keys must come from here: temporal alignment
(:func:`sccn_temporal_layer.align_temporal_features`) maps row ``i`` of the carried hidden
state to the cell whose identity is ``keys[i]``. If ``keys`` were not in the same order as
the adjacency/incidence matrices, the ``A+I`` self-loop and the temporal ``A``/``B`` blocks
would connect *wrong* cells. toponetx orders cells consistently across
``incidence_matrix`` / ``adjacency_matrix`` (sorted vertex tuples), which is what we read.

Matrix conventions (configurable):
- ``incidences[f"rank_{r}"]`` = ``incidence_matrix(r, signed=signed)`` -> ``B_r`` of shape
  ``(n_{r-1}, n_r)``, for ``r = 1..max_rank``.
- ``adjacencies[f"rank_{r}"]`` is derived from the incidences so its ordering is guaranteed
  to match: lower (``B_r^T B_r``, cells sharing an ``(r-1)``-face) and/or upper
  (``B_{r+1} B_{r+1}^T``, cells sharing an ``(r+1)``-coface), selected by ``adjacency``.
  This reproduces toponetx ``coadjacency_matrix`` + ``adjacency_matrix`` (binarized).

This module lives in the project root (outside ``topomodelx``); ``toponetx`` is imported
lazily so the model/layer files do not depend on it.
"""
import torch


def _empty_sparse(rows, cols, device, dtype):
    return torch.sparse_coo_tensor(
        torch.empty(2, 0, dtype=torch.long, device=device),
        torch.empty(0, device=device, dtype=dtype),
        (rows, cols),
    ).coalesce()


def _binarize_offdiag(dense):
    m = (dense != 0).to(dense.dtype)
    if m.numel():
        m.fill_diagonal_(0)
    return m


def _order_features(feat, keys, channels, device, dtype):
    """Return an ``(n_r, channels)`` tensor aligned to ``keys``.

    ``feat`` is either an already-ordered tensor (rows must match ``keys``) or a
    ``{cell_key -> vector}`` mapping that is reordered to ``keys``.
    """
    if isinstance(feat, torch.Tensor):
        if feat.shape[0] != len(keys):
            raise ValueError(
                f"feature tensor has {feat.shape[0]} rows but rank has {len(keys)} cells"
            )
        return feat.to(device=device, dtype=dtype)
    if len(keys) == 0:
        return torch.zeros((0, channels), device=device, dtype=dtype)
    rows = []
    for k in keys:
        if k not in feat:
            raise KeyError(f"no feature provided for cell {k}")
        rows.append(torch.as_tensor(feat[k], device=device, dtype=dtype))
    return torch.stack(rows, dim=0)


def simplicial_window(
    sc,
    features,
    max_rank,
    *,
    signed: bool = False,
    adjacency: str = "both",
    weighted_adjacency: bool = False,
    channels: int | None = None,
    device=None,
    dtype=torch.float32,
):
    """Build one window dict for the spatio-temporal SCCN model.

    Parameters
    ----------
    sc : toponetx.SimplicialComplex
        The complex for this window.
    features : dict
        Per-rank input features, keyed by ``"rank_{r}"`` (or ``int`` ``r``). Each value is
        either an already-ordered ``(n_r, d_r)`` tensor (rows in the SAME order this
        function emits ``keys``) or a ``{cell_key -> vector}`` mapping that this function
        reorders. ``cell_key`` is a sorted vertex tuple, e.g. ``(1, 2)`` for an edge.
    max_rank : int
        Maximum rank to include (``>= 1``). Ranks above ``sc.dim`` come out empty.
    signed : bool, default=False
        Passed to ``incidence_matrix`` (unsigned 0/1 boundary by default).
    adjacency : {"both", "lower", "upper"}, default="both"
        Which same-rank adjacency to build: lower (share an ``(r-1)``-face), upper (share
        an ``(r+1)``-coface), or both summed.
    weighted_adjacency : bool, default=False
        If ``False`` the adjacency is binarized (0/1) off-diagonal; if ``True`` raw counts
        are kept (diagonal still zeroed).
    channels : int, optional
        Feature width; only used to shape empty ranks. Inferred from a non-empty rank's
        features if omitted.
    device, dtype : optional
        Target device/dtype for all tensors.

    Returns
    -------
    dict
        ``{"features", "adjacencies", "incidences", "keys"}`` ready for
        ``SCCNTemporalClassifier.forward`` (as one element of the ``windows`` list).
    """
    from topomodelx.utils.sparse import from_sparse  # lazy: only needed here

    if adjacency not in ("both", "lower", "upper"):
        raise ValueError(f"adjacency must be 'both'/'lower'/'upper', got {adjacency!r}")

    # normalize feature keys to "rank_{r}"
    feats_in = {}
    for k, v in features.items():
        feats_in[f"rank_{k}" if isinstance(k, int) else k] = v
    if channels is None:
        for v in feats_in.values():
            if isinstance(v, torch.Tensor) and v.shape[0] > 0:
                channels = v.shape[1]
                break

    dim = sc.dim  # highest rank actually present in this window

    # ---- ordered keys + incidences, per rank ----
    keys = {}
    incidences = {}
    b_dense = {}  # for adjacency derivation, indexed by int rank

    node_idx, _ = sc.adjacency_matrix(0, index=True)
    keys["rank_0"] = [tuple(k) for k in node_idx.keys()]

    for r in range(1, max_rank + 1):
        m = len(keys[f"rank_{r-1}"])
        if r > dim:  # rank absent this window -> empty
            keys[f"rank_{r}"] = []
            incidences[f"rank_{r}"] = _empty_sparse(m, 0, device, dtype)
            b_dense[r] = torch.zeros((m, 0), device=device, dtype=dtype)
            continue
        _, col_idx, b_csr = sc.incidence_matrix(r, signed=signed, index=True)
        keys[f"rank_{r}"] = [tuple(k) for k in col_idx.keys()]
        b = from_sparse(b_csr).to(device=device, dtype=dtype).coalesce()
        incidences[f"rank_{r}"] = b
        b_dense[r] = b.to_dense()

    n = {r: len(keys[f"rank_{r}"]) for r in range(max_rank + 1)}

    # ---- same-rank adjacencies, derived from incidences (ordering guaranteed) ----
    adjacencies = {}
    for r in range(max_rank + 1):
        a = torch.zeros((n[r], n[r]), device=device, dtype=dtype)
        if adjacency in ("both", "lower") and r >= 1 and n[r] and n[r - 1]:
            a = a + b_dense[r].T @ b_dense[r]
        if adjacency in ("both", "upper") and r < max_rank and n[r] and n[r + 1]:
            a = a + b_dense[r + 1] @ b_dense[r + 1].T
        a = a if weighted_adjacency else _binarize_offdiag(a)
        if weighted_adjacency and a.numel():
            a.fill_diagonal_(0)
        adjacencies[f"rank_{r}"] = a.to_sparse_coo().coalesce()

    # ---- features, ordered to match keys ----
    out_features = {}
    for r in range(max_rank + 1):
        rk = f"rank_{r}"
        if rk not in feats_in:
            raise KeyError(f"features missing for {rk}")
        out_features[rk] = _order_features(feats_in[rk], keys[rk], channels, device, dtype)

    return {
        "features": out_features,
        "adjacencies": adjacencies,
        "incidences": incidences,
        "keys": keys,
    }


def simplicial_sample(complexes, features_seq, max_rank, **kwargs):
    """Build a full sample (list of window dicts) from a sequence of complexes.

    Parameters
    ----------
    complexes : list[toponetx.SimplicialComplex]
        One complex per window, in temporal order.
    features_seq : list[dict]
        One ``features`` dict per window (see :func:`simplicial_window`).
    max_rank : int
        Maximum rank to include.
    **kwargs
        Forwarded to :func:`simplicial_window`.

    Returns
    -------
    list[dict]
        The ``windows`` argument for ``SCCNTemporalClassifier.forward``.
    """
    if len(complexes) != len(features_seq):
        raise ValueError("complexes and features_seq must have the same length")
    return [
        simplicial_window(sc, feats, max_rank, **kwargs)
        for sc, feats in zip(complexes, features_seq)
    ]
