"""Convert a SEED ``WindowComplexRecord`` into one SCCN-temporal window dict.

Builds ``features``, ``incidences``, ``adjacencies``, and ``keys`` **directly** from
the extracted edge/triangle lists and stored features/weights. Does **not** rebuild a
TopoNetX complex or call ``simplicial_window``.
"""

from __future__ import annotations

from typing import Any

import torch

N_CHANNELS = 62
DEFAULT_FEAT_DIM = 5


def _as_tuple_feat(vec: Any, dim: int, device, dtype) -> torch.Tensor:
    t = torch.as_tensor(vec, device=device, dtype=dtype).reshape(-1)
    if t.numel() == dim:
        return t
    if t.numel() == 1:
        return t.repeat(dim)
    if t.numel() == 0:
        return torch.ones(dim, device=device, dtype=dtype)
    out = torch.zeros(dim, device=device, dtype=dtype)
    n = min(dim, t.numel())
    out[:n] = t[:n]
    return out


def _empty_sparse(rows: int, cols: int, device, dtype) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.empty(2, 0, dtype=torch.long, device=device),
        torch.empty(0, device=device, dtype=dtype),
        (rows, cols),
    ).coalesce()


def _binarize_offdiag(dense: torch.Tensor) -> torch.Tensor:
    m = (dense != 0).to(dense.dtype)
    if m.numel():
        m.fill_diagonal_(0)
    return m


def _cell_nodes(keys: list[tuple[int, ...]], arity: int, device) -> torch.Tensor:
    """Stack a rank's cell keys into a ``(n_r, arity)`` tensor of node indices.

    Every cell identity is already a tuple of channel indices (``(i,)`` for nodes,
    ``(i, j)`` for edges, ``(i, j, k)`` for triangles), so this is just those keys in
    tensor form. Downstream this is what lets a model look up per-channel information
    (learned embeddings, electrode coordinates) for cells of *any* rank.
    """
    if not keys:
        return torch.zeros((0, arity), dtype=torch.long, device=device)
    return torch.tensor(keys, dtype=torch.long, device=device).reshape(len(keys), arity)


def _canon_edge(e) -> tuple[int, int]:
    a, b = (int(e[0]), int(e[1]))
    return (a, b) if a <= b else (b, a)


def _canon_tri(t) -> tuple[int, int, int]:
    return tuple(sorted(int(v) for v in t))  # type: ignore[return-value]


def _edge_feature(record, i: int, key: tuple[int, int], feat_dim: int, device, dtype) -> torch.Tensor:
    if record.edge_features and i < len(record.edge_features) and record.edge_features[i]:
        return _as_tuple_feat(record.edge_features[i], feat_dim, device, dtype)
    if record.edge_weights and i < len(record.edge_weights):
        return _as_tuple_feat(record.edge_weights[i], feat_dim, device, dtype)
    return torch.ones(feat_dim, device=device, dtype=dtype)


def _tri_feature(record, i: int, key: tuple[int, int, int], feat_dim: int, device, dtype) -> torch.Tensor:
    if (
        record.triangle_features
        and i < len(record.triangle_features)
        and record.triangle_features[i]
    ):
        return _as_tuple_feat(record.triangle_features[i], feat_dim, device, dtype)
    if record.triangle_weights and i < len(record.triangle_weights):
        return _as_tuple_feat(record.triangle_weights[i], feat_dim, device, dtype)
    return torch.ones(feat_dim, device=device, dtype=dtype)


def _node_features(record, feat_dim: int, device, dtype) -> torch.Tensor:
    if record.node_features and len(record.node_features) == N_CHANNELS:
        return torch.stack(
            [_as_tuple_feat(row, feat_dim, device, dtype) for row in record.node_features],
            dim=0,
        )
    return torch.zeros((N_CHANNELS, feat_dim), device=device, dtype=dtype)


def _build_incidence_nodes_edges(
    edge_keys: list[tuple[int, int]],
    *,
    device,
    dtype,
) -> torch.Tensor:
    """Unsigned B1 of shape (n0, n1)."""
    n0, n1 = N_CHANNELS, len(edge_keys)
    if n1 == 0:
        return _empty_sparse(n0, 0, device, dtype)
    rows: list[int] = []
    cols: list[int] = []
    for c, (i, j) in enumerate(edge_keys):
        rows.extend([i, j])
        cols.extend([c, c])
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.ones(len(rows), device=device, dtype=dtype)
    return torch.sparse_coo_tensor(idx, vals, (n0, n1)).coalesce()


def _build_incidence_edges_triangles(
    edge_keys: list[tuple[int, int]],
    tri_keys: list[tuple[int, int, int]],
    *,
    device,
    dtype,
) -> torch.Tensor:
    """Unsigned B2 of shape (n1, n2); only faces present in ``edge_keys`` are linked."""
    n1, n2 = len(edge_keys), len(tri_keys)
    if n2 == 0:
        return _empty_sparse(n1, 0, device, dtype)
    edge_index = {e: i for i, e in enumerate(edge_keys)}
    rows: list[int] = []
    cols: list[int] = []
    for c, (u, v, w) in enumerate(tri_keys):
        for face in ((u, v), (u, w), (v, w)):
            a, b = (face if face[0] <= face[1] else (face[1], face[0]))
            r = edge_index.get((a, b))
            if r is not None:
                rows.append(r)
                cols.append(c)
    if not rows:
        return _empty_sparse(n1, n2, device, dtype)
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.ones(len(rows), device=device, dtype=dtype)
    return torch.sparse_coo_tensor(idx, vals, (n1, n2)).coalesce()


def _adjacency_from_incidences(
    b_dense: dict[int, torch.Tensor],
    n: dict[int, int],
    max_rank: int,
    *,
    adjacency: str,
    weighted_adjacency: bool,
    device,
    dtype,
) -> dict[str, torch.Tensor]:
    """Same-rank A_r from B_r / B_{r+1}, matching simplicial_window defaults."""
    adjacencies: dict[str, torch.Tensor] = {}
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
    return adjacencies


def record_to_sccn_window(
    record,
    *,
    max_rank: int = 2,
    feat_dim: int = DEFAULT_FEAT_DIM,
    signed: bool = False,  # kept for API compat; matrices are unsigned 0/1
    adjacency: str = "both",
    weighted_adjacency: bool = False,
    device=None,
    dtype=torch.float32,
) -> dict:
    """Build one SCCN window dict purely from extracted SEED fields.

    Uses ``edges``, ``triangles``, ``node_features``, ``edge_features`` /
    ``triangle_features`` (with weight fallbacks). No TopoNetX rebuild.
    """
    del signed  # unsigned incidences only
    if adjacency not in ("both", "lower", "upper"):
        raise ValueError(f"adjacency must be 'both'/'lower'/'upper', got {adjacency!r}")

    # ---- ordered keys from extracted lists (stable cell identities) ----
    keys0 = [(i,) for i in range(N_CHANNELS)]

    edge_keys: list[tuple[int, int]] = []
    edge_feats: list[torch.Tensor] = []
    seen_e: set[tuple[int, int]] = set()
    for i, e in enumerate(record.edges):
        key = _canon_edge(e)
        if key in seen_e:
            continue
        seen_e.add(key)
        edge_keys.append(key)
        edge_feats.append(_edge_feature(record, i, key, feat_dim, device, dtype))

    tri_keys: list[tuple[int, int, int]] = []
    tri_feats: list[torch.Tensor] = []
    seen_t: set[tuple[int, int, int]] = set()
    for i, t in enumerate(record.triangles):
        key = _canon_tri(t)
        if key in seen_t:
            continue
        seen_t.add(key)
        tri_keys.append(key)
        tri_feats.append(_tri_feature(record, i, key, feat_dim, device, dtype))

    if max_rank < 1:
        edge_keys, edge_feats = [], []
    if max_rank < 2:
        tri_keys, tri_feats = [], []

    keys = {
        "rank_0": keys0,
        "rank_1": edge_keys,
        "rank_2": tri_keys if max_rank >= 2 else [],
    }
    # pad empty higher ranks if max_rank > 2 (not used for SEED)
    for r in range(3, max_rank + 1):
        keys[f"rank_{r}"] = []

    # ---- features aligned to keys ----
    x0 = _node_features(record, feat_dim, device, dtype)
    features = {
        "rank_0": x0,
        "rank_1": (
            torch.stack(edge_feats, dim=0)
            if edge_feats
            else torch.zeros((0, feat_dim), device=device, dtype=dtype)
        ),
        "rank_2": (
            torch.stack(tri_feats, dim=0)
            if tri_feats
            else torch.zeros((0, feat_dim), device=device, dtype=dtype)
        ),
    }
    for r in range(3, max_rank + 1):
        features[f"rank_{r}"] = torch.zeros((0, feat_dim), device=device, dtype=dtype)

    # ---- incidences B_r ----
    incidences: dict[str, torch.Tensor] = {}
    b_dense: dict[int, torch.Tensor] = {}
    n = {0: N_CHANNELS, 1: len(edge_keys), 2: len(tri_keys)}
    for r in range(3, max_rank + 1):
        n[r] = 0

    if max_rank >= 1:
        b1 = _build_incidence_nodes_edges(edge_keys, device=device, dtype=dtype)
        incidences["rank_1"] = b1
        b_dense[1] = b1.to_dense()
    if max_rank >= 2:
        b2 = _build_incidence_edges_triangles(
            edge_keys, tri_keys, device=device, dtype=dtype
        )
        incidences["rank_2"] = b2
        b_dense[2] = b2.to_dense()
    for r in range(3, max_rank + 1):
        incidences[f"rank_{r}"] = _empty_sparse(n[r - 1], 0, device, dtype)
        b_dense[r] = torch.zeros((n[r - 1], 0), device=device, dtype=dtype)

    # ---- adjacencies A_r ----
    adjacencies = _adjacency_from_incidences(
        b_dense,
        n,
        max_rank,
        adjacency=adjacency,
        weighted_adjacency=weighted_adjacency,
        device=device,
        dtype=dtype,
    )

    # Constituent channel indices per cell, parallel to ``keys`` (see ``_cell_nodes``).
    cell_nodes = {
        "rank_0": _cell_nodes(keys0, 1, device),
        "rank_1": _cell_nodes(edge_keys, 2, device),
        "rank_2": _cell_nodes(tri_keys if max_rank >= 2 else [], 3, device),
    }
    for r in range(3, max_rank + 1):
        cell_nodes[f"rank_{r}"] = _cell_nodes([], r + 1, device)

    return {
        "features": features,
        "adjacencies": adjacencies,
        "incidences": incidences,
        "keys": keys,
        "cell_nodes": cell_nodes,
        # Where this window sits inside its trial. ``t_frac`` is filled in by
        # ``records_to_sccn_windows``, which is the only place that knows the
        # trial's total length; ``window_idx`` is the raw extracted index.
        "window_idx": int(getattr(record, "window_idx", 0)),
        "t_frac": 0.0,
        # (eye_dim,) for the multimodal extract, (0,) otherwise. One vector per
        # window -- eye is a window-level modality, not a per-cell one.
        "eye": (
            torch.as_tensor(record.eye_features, device=device, dtype=dtype)
            if getattr(record, "eye_features", ())
            else torch.zeros(0, device=device, dtype=dtype)
        ),
    }


def assign_local_cell_ids(windows: list[dict]) -> None:
    """Give every cell in a trial a stable integer id, shared across its windows.

    The temporal (LSTM) state has to follow a cell from window to window even as
    edges/triangles appear and disappear. Keyed by cell *identity* that means a Python
    dict lookup per cell per rank per window -- hundreds of tiny tensor ops that
    dominate runtime. Numbering each trial's cells 0..N-1 once, up front, lets the
    recurrent state live in a dense ``(N, channels)`` tensor instead, so alignment
    becomes a single gather and the write-back a single scatter.

    Adds ``local_ids`` (``{rank: LongTensor(n_r)}``, parallel to ``keys``) and
    ``n_cells`` (``{rank: int}``, the trial-wide total) to every window in place.
    """
    if not windows:
        return
    ranks = list(windows[0]["keys"].keys())
    for r in ranks:
        id_of: dict = {}
        for w in windows:
            for k in w["keys"][r]:
                if k not in id_of:
                    id_of[k] = len(id_of)
        total = len(id_of)
        for w in windows:
            keys_r = w["keys"][r]
            ids = torch.tensor(
                [id_of[k] for k in keys_r], dtype=torch.long
            ) if keys_r else torch.zeros(0, dtype=torch.long)
            w.setdefault("local_ids", {})[r] = ids
            w.setdefault("n_cells", {})[r] = total


def records_to_sccn_windows(records: list, **kwargs) -> list[dict]:
    """Convert an ordered list of SEED records (e.g. one trial) into a window sequence.

    Also stamps each window's normalized position in the trial (``t_frac``, 0.0 at the
    first window through 1.0 at the last), which is what lets a model tell "early" from
    "mid" from "late" for trials of differing length, and assigns the trial-stable cell
    ids used by the dense temporal state (see :func:`assign_local_cell_ids`).
    """
    windows = [record_to_sccn_window(r, **kwargs) for r in records]
    last = len(windows) - 1
    for i, w in enumerate(windows):
        w["t_frac"] = (i / last) if last > 0 else 0.0
    assign_local_cell_ids(windows)
    return windows
