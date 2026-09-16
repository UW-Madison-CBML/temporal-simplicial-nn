"""Temporal-SCCN layer blocks over a sequence of simplicial windows.

This module defines **three** ways to combine temporal encoding with SCCN spatial
message passing, all sharing a uniform interface so a model can stack any of them and
swap freely. Each block owns a per-rank ``LSTMCell`` (the temporal encoder) and a plain
:class:`SCCNLayer` (the spatial step); the recurrent ``(h, c)`` state is carried across
windows in **identity-keyed** stores, re-indexed to each window's cell ordering with
:func:`align_temporal_features` (zeros for newly-appeared cells), so edges/triangles that
appear and disappear are handled correctly.

The three variants (``forward(features, incidences, adjacencies, keys, h_state,
c_state)``):

1. :class:`SiSTSCCNLayer` -- **simultaneous** spatial+temporal augmentation. The LSTM hidden ``h_p`` is
   the *temporal copy* of each cell;
   features ``[h_c ; h_p]`` are stacked and the matrices are **augmented** so a single
   shared-weight SCCN pass routes present *and* temporal messages in every SCCN direction
   at once; the current half of the output is sliced out. This is the only variant that
   augments the matrices.
2. :class:`TemporalThenSpatialSCCNLayer` -- **temporal then spatial** (no augmentation):
   LSTM-encode each cell over time, then a plain :class:`SCCNLayer` pass.
3. :class:`SpatialThenTemporalSCCNLayer` -- **spatial then temporal** (no augmentation):
   a plain :class:`SCCNLayer` pass, then LSTM-encode the result over time.

All variants support any maximum rank ``R = max_rank >= 1`` (nodes+edges, up to arbitrary
order); the per-rank logic is built from whatever ranks are present in the input dicts.

Simultaneous augmentation (variant 1), using SCCN's convention ``N[x, y]`` routes ``y -> x``,
per rank ``r`` with adjacency ``A_r`` and incidence ``B_r``::

    A_aug_r = [[ A_r + I , A_r + I ],   B_aug_r = [[ B_r , B_r ],
               [   0     ,   0     ]]              [ B_r ,  0  ]]

Top-left ``A_r (+I)`` = present same-rank (``I`` = own-current -> own-present self-loop,
added when ``present_self_loop=True``); top-right ``A_r + I`` = temporal same-rank
(``A_r`` = neighbours' past, ``I`` = own-past self-loop, added when
``temporal_self_loop=True``). ``B_aug_r`` is used directly (higher->lower) and transposed
by :class:`SCCNLayer` (lower->higher), so its blocks yield present + temporal in both.
Bottom rows are past *targets* and are discarded after the slice. This produces, for a
rank-``r`` cell, present+temporal messages in every available direction:

- same-rank (always):           ``(r)_c -> (r)_c`` , ``(r)_p -> (r)_c``
- higher -> lower (if r < R):   ``(r+1)_c -> (r)_c`` , ``(r+1)_p -> (r)_c``
- lower -> higher (if r > 0):   ``(r-1)_c -> (r)_c`` , ``(r-1)_p -> (r)_c``

This module lives outside the ``topomodelx`` package (in the project root) and only
*imports* ``SCCNLayer`` from it without modifying it. Run scripts from the project
root (or with the project root on ``PYTHONPATH``) so ``topomodelx`` resolves.
"""
from typing import Literal

import torch

from topomodelx.nn.simplicial.sccn_layer import SCCNLayer


def _eye_sparse(n: int, device, dtype) -> torch.Tensor:
    """Return a sparse ``(n, n)`` identity matrix.

    Parameters
    ----------
    n : int
        Size of the identity matrix.
    device : torch.device
        Device of the returned tensor.
    dtype : torch.dtype
        Dtype of the returned tensor.

    Returns
    -------
    torch.sparse_coo_tensor, shape = (n, n)
        Sparse identity matrix.
    """
    idx = torch.arange(n, device=device)
    return torch.sparse_coo_tensor(
        torch.stack([idx, idx]),
        torch.ones(n, device=device, dtype=dtype),
        (n, n),
    )


def _shift(mat: torch.Tensor, r_off: int, c_off: int, shape) -> torch.Tensor:
    """Place a sparse matrix as a block at a row/column offset within ``shape``.

    Parameters
    ----------
    mat : torch.sparse_coo_tensor
        Block to place.
    r_off : int
        Row offset of the block's top-left corner.
    c_off : int
        Column offset of the block's top-left corner.
    shape : tuple[int, int]
        Shape of the enclosing (output) matrix.

    Returns
    -------
    torch.sparse_coo_tensor, shape = shape
        Sparse matrix containing ``mat`` at the requested offset.
    """
    mat = mat.coalesce()
    idx = mat.indices().clone()
    idx[0] += r_off
    idx[1] += c_off
    return torch.sparse_coo_tensor(idx, mat.values(), shape)


def augment_adjacency(
    adjacency: torch.Tensor,
    temporal_self_loop: bool = True,
    present_self_loop: bool = True,
) -> torch.Tensor:
    """Build the augmented same-rank adjacency ``[[A(+I), A(+I)], [0, 0]]``.

    Parameters
    ----------
    adjacency : torch.sparse_coo_tensor, shape = (n, n)
        Original same-rank adjacency matrix ``A_r``.
    temporal_self_loop : bool, default=True
        Whether to add the identity (own-past -> own-present) self-loop to the
        temporal (top-right) block.
    present_self_loop : bool, default=True
        Whether to add the identity (own-current -> own-present) self-loop to the
        present (top-left) block, so a cell's own current feature contributes to its own
        updated feature (own-current -> own-present self-loop).

    Returns
    -------
    torch.sparse_coo_tensor, shape = (2 * n, 2 * n)
        Augmented adjacency. The top-left block is the present flow ``A`` (plus ``I`` if
        ``present_self_loop``); the top-right block is the temporal flow ``A``
        (neighbours' past) plus ``I`` if ``temporal_self_loop`` (own past). Bottom rows
        are zero (discarded past targets).
    """
    adjacency = adjacency.coalesce()
    n = adjacency.size(0)
    shp = (2 * n, 2 * n)
    out = _shift(adjacency, 0, 0, shp) + _shift(adjacency, 0, n, shp)
    eye = _eye_sparse(n, adjacency.device, adjacency.dtype)
    if present_self_loop:
        out = out + _shift(eye, 0, 0, shp)
    if temporal_self_loop:
        out = out + _shift(eye, 0, n, shp)
    return out.coalesce()


def augment_incidence(incidence: torch.Tensor) -> torch.Tensor:
    """Build the augmented incidence ``[[B, B], [B, 0]]``.

    Parameters
    ----------
    incidence : torch.sparse_coo_tensor, shape = (m, n)
        Original incidence matrix ``B_r`` mapping rank-``r`` cells to
        rank-``(r-1)`` cells.

    Returns
    -------
    torch.sparse_coo_tensor, shape = (2 * m, 2 * n)
        Augmented incidence. Used directly by :class:`SCCNLayer` for the
        higher->lower direction (present + temporal via the top blocks) and
        transposed for the lower->higher direction (present + temporal). The
        bottom-right block is zero.
    """
    incidence = incidence.coalesce()
    m, n = incidence.shape
    shp = (2 * m, 2 * n)
    return (
        _shift(incidence, 0, 0, shp)
        + _shift(incidence, 0, n, shp)
        + _shift(incidence, m, 0, shp)
    ).coalesce()


def align_temporal_features(
    store,
    current_keys,
    channels: int,
    *,
    init=None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Re-index a temporal-hidden store into the current cell ordering.

    Edges and triangles appear and disappear over time, so the carried temporal
    hidden state must be aligned to the *current* matrix indexing before it can be
    stacked with the current features. The store is keyed by a stable cell identity
    (e.g. ``tuple(sorted(simplex))``), not by row index.

    Parameters
    ----------
    store : dict
        Mapping ``cell_key -> torch.Tensor`` of shape ``(channels,)`` holding the
        previous temporal-hidden state of each cell.
    current_keys : list
        Ordered list of cell keys (length ``n_r``) matching the rows of the current
        ``A_r`` / ``B_r`` / ``h_c[rank]``.
    channels : int
        Feature dimension.
    init : torch.Tensor, optional
        Initialization for cells absent from ``store`` (newly appeared). Defaults to
        zeros. Pass a learnable ``torch.nn.Parameter`` to make the new-cell init
        trainable.
    device : torch.device, optional
        Device for default (zeros) initialization.
    dtype : torch.dtype, optional
        Dtype for default (zeros) initialization.

    Returns
    -------
    torch.Tensor, shape = (n_r, channels)
        Temporal-hidden features aligned to the current cell ordering. Returns a
        ``(0, channels)`` tensor when ``current_keys`` is empty (rank absent this step).
    """
    if len(current_keys) == 0:
        return torch.zeros((0, channels), device=device, dtype=dtype)
    rows = [
        store[k]
        if k in store
        else (
            torch.zeros(channels, device=device, dtype=dtype)
            if init is None
            else init
        )
        for k in current_keys
    ]
    return torch.stack(rows, dim=0)


def update_temporal_store(store, current_keys, new_hidden, *, detach: bool = True):
    """Write new temporal-hidden states back into the store, keyed by cell identity.

    Call this after the caller's recurrence step has produced the next hidden state.
    Cells absent from ``current_keys`` are kept by default (so they resume with memory
    if they reappear); prune them explicitly for eviction.

    Parameters
    ----------
    store : dict
        Mapping ``cell_key -> torch.Tensor`` to update in place.
    current_keys : list
        Ordered list of cell keys (length ``n_r``) matching the rows of ``new_hidden``.
    new_hidden : torch.Tensor, shape = (n_r, channels)
        New temporal-hidden states for the current cells.
    detach : bool, default=True
        Whether to detach before storing. ``True`` avoids cross-step
        backpropagation-through-time unless the caller wants it.

    Returns
    -------
    dict
        The updated ``store`` (same object, mutated in place).
    """
    for i, k in enumerate(current_keys):
        store[k] = new_hidden[i].detach() if detach else new_hidden[i]
    return store


class _TemporalSCCNBlockBase(torch.nn.Module):
    """Shared base for the temporal-SCCN variant blocks.

    Owns the recurrence (a per-rank :class:`torch.nn.LSTMCell`) and the spatial primitive
    (a plain :class:`SCCNLayer`), plus the identity-keyed temporal-state plumbing. The
    three subclasses differ only in how they order the temporal and spatial steps (and
    whether they augment the matrices). All share the forward signature::

        forward(features, incidences, adjacencies, keys, h_state, c_state) -> out_features

    where ``features`` are the current per-rank features ``{f"rank_{r}": (n_r, channels)}``,
    ``keys`` are stable per-rank cell identities (same row order as the matrices), and
    ``h_state`` / ``c_state`` are per-rank identity-keyed dicts carried across windows
    (mutated in place). Use :meth:`init_state` to create the empty state at the start of a
    sequence.

    Parameters
    ----------
    channels : int
        Shared feature width across ranks.
    max_rank : int
        Maximum rank (``>= 1``). Any order is supported.
    aggr_func : {"mean", "sum"}, default="sum"
        Aggregation inside the wrapped :class:`SCCNLayer`.
    update_func : {"relu", "sigmoid", "tanh", None}, default="sigmoid"
        Activation inside the wrapped :class:`SCCNLayer`.
    """

    def __init__(
        self,
        channels,
        max_rank,
        aggr_func: Literal["mean", "sum"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh"] | None = "sigmoid",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.max_rank = max_rank
        self.ranks = [f"rank_{r}" for r in range(max_rank + 1)]
        self.sccn = SCCNLayer(
            channels=channels,
            max_rank=max_rank,
            aggr_func=aggr_func,
            update_func=update_func,
        )
        self.lstms = torch.nn.ModuleDict(
            {r: torch.nn.LSTMCell(channels, channels) for r in self.ranks}
        )

    def init_state(self, n_cells=None, device=None, dtype=None):
        """Return fresh ``(h_state, c_state)`` stores for a new sequence.

        Two equivalent representations:

        - ``n_cells is None`` -> identity-keyed dicts (the original path).
        - ``n_cells`` given -> dense ``(n_cells[rank], channels)`` zero tensors indexed
          by the trial-stable cell ids from ``assign_local_cell_ids``. Same values, but
          alignment/write-back become one gather and one scatter instead of a Python
          loop over every cell.

        The dense form starts at zeros, which is exactly what the dict form returns for
        a cell it has never seen, so the two agree from the first window onward.
        """
        if n_cells is None:
            return ({r: {} for r in self.ranks}, {r: {} for r in self.ranks})
        h = {
            r: torch.zeros(n_cells[r], self.channels, device=device, dtype=dtype)
            for r in self.ranks
        }
        c = {
            r: torch.zeros(n_cells[r], self.channels, device=device, dtype=dtype)
            for r in self.ranks
        }
        return h, c

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.sccn.reset_parameters()
        for cell in self.lstms.values():
            cell.reset_parameters()

    def _temporal_step(self, feats, keys, h_state, c_state, local_ids=None):
        """Per-rank LSTM over time, aligning the recurrent state to this window.

        Aligns the previous ``(h, c)`` to the current cell ordering (zeros for new
        cells), runs the per-rank ``LSTMCell`` on ``feats``, writes the new ``(h, c)``
        back (gradients flow across windows), and returns the new per-rank hidden state
        ``{f"rank_{r}": (n_r, channels)}``.

        With ``local_ids`` the state is dense and indexed by trial-stable cell id, so
        alignment is one ``index_select`` and write-back one out-of-place
        ``index_copy``; without it, the original identity-keyed dict path runs. The two
        compute identical values -- ``index_copy`` leaves untouched rows alone, exactly
        as the dict keeps cells absent from ``current_keys``.
        """
        out = {}
        for r in self.ranks:
            if local_ids is not None:
                ids = local_ids[r]
                h_prev = h_state[r][ids]
                c_prev = c_state[r][ids]
                h_new, c_new = self.lstms[r](feats[r], (h_prev, c_prev))
                out[r] = h_new
                # out-of-place: an in-place write would break autograd through the
                # state that earlier windows still reference.
                h_state[r] = h_state[r].index_copy(0, ids, h_new)
                c_state[r] = c_state[r].index_copy(0, ids, c_new)
                continue

            keys_r = keys[r]
            dev, dt = feats[r].device, feats[r].dtype
            h_prev = align_temporal_features(
                h_state[r], keys_r, self.channels, device=dev, dtype=dt
            )
            c_prev = align_temporal_features(
                c_state[r], keys_r, self.channels, device=dev, dtype=dt
            )
            h_new, c_new = self.lstms[r](feats[r], (h_prev, c_prev))
            out[r] = h_new
            update_temporal_store(h_state[r], keys_r, h_new, detach=False)
            update_temporal_store(c_state[r], keys_r, c_new, detach=False)
        return out

    def forward(
        self, features, incidences, adjacencies, keys, h_state, c_state,
        local_ids=None, cache=None,
    ):
        raise NotImplementedError


class SiSTSCCNLayer(_TemporalSCCNBlockBase):
    """Simultaneous spatio-temporal SCCN block.

    Computes the temporal hidden ``h_p`` for each cell with the per-rank LSTM, stacks
    ``[features ; h_p]``, augments the matrices (:func:`augment_adjacency` /
    :func:`augment_incidence`), runs the wrapped :class:`SCCNLayer` once, and slices the
    current half. A single shared-weight pass thus routes present + temporal messages in
    every SCCN direction (see module docstring). This is the only variant that augments.

    Parameters
    ----------
    channels, max_rank, aggr_func, update_func
        See :class:`_TemporalSCCNBlockBase`.
    temporal_self_loop : bool, default=True
        Add the own-past -> own-present self-loop to the temporal same-rank block.
    present_self_loop : bool, default=True
        Add the own-current -> own-present self-loop to the present same-rank block.
    """

    def __init__(
        self,
        channels,
        max_rank,
        aggr_func: Literal["mean", "sum"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh"] | None = "sigmoid",
        temporal_self_loop: bool = True,
        present_self_loop: bool = True,
    ) -> None:
        super().__init__(channels, max_rank, aggr_func, update_func)
        self.temporal_self_loop = temporal_self_loop
        self.present_self_loop = present_self_loop

    def forward(
        self,
        features,
        incidences,
        adjacencies,
        keys,
        h_state,
        c_state,
        local_ids=None,
        cache=None,
    ):
        r"""Forward pass (simultaneous): temporal hidden -> augment -> SCCN -> slice.

        The augmented matrices depend only on the window's topology and the two
        self-loop flags -- never on learned parameters -- yet rebuilding them costs a
        ``coalesce()`` (a sort) per rank, and both stacked blocks rebuild the same ones.
        When ``cache`` is a dict (the window itself), the result is memoized under the
        flag pair, so the work happens once per window for the whole run instead of
        once per block per epoch.
        """
        h_p = self._temporal_step(features, keys, h_state, c_state, local_ids)
        feats = {k: torch.cat([features[k], h_p[k]], dim=0) for k in features}

        key = ("_aug", self.temporal_self_loop, self.present_self_loop)
        cached = cache.get(key) if cache is not None else None
        if cached is None:
            adj = {
                k: augment_adjacency(v, self.temporal_self_loop, self.present_self_loop)
                for k, v in adjacencies.items()
            }
            inc = {k: augment_incidence(v) for k, v in incidences.items()}
            if cache is not None:
                cache[key] = (adj, inc)
        else:
            adj, inc = cached

        out = self.sccn(feats, inc, adj)
        return {k: out[k][: features[k].size(0)] for k in out}


class TemporalThenSpatialSCCNLayer(_TemporalSCCNBlockBase):
    """Temporal-then-spatial SCCN block (no augmentation).

    LSTM-encodes each cell over time, then runs a plain :class:`SCCNLayer` pass on the
    temporally-encoded features using the window's *un-augmented* matrices.
    """

    def forward(
        self, features, incidences, adjacencies, keys, h_state, c_state,
        local_ids=None, cache=None,
    ):
        r"""Forward pass: temporal encode, then plain spatial SCCN.

        ``cache`` is accepted for interface parity and unused -- this variant does not
        augment the matrices, so there is nothing to memoize.
        """
        h = self._temporal_step(features, keys, h_state, c_state, local_ids)
        return self.sccn(h, incidences, adjacencies)


class SpatialThenTemporalSCCNLayer(_TemporalSCCNBlockBase):
    """Spatial-then-temporal SCCN block (no augmentation).

    Runs a plain :class:`SCCNLayer` pass on the current features using the window's
    *un-augmented* matrices, then LSTM-encodes the spatial output over time.
    """

    def forward(
        self, features, incidences, adjacencies, keys, h_state, c_state,
        local_ids=None, cache=None,
    ):
        r"""Forward pass: plain spatial SCCN, then temporal encode.

        ``cache`` is accepted for interface parity and unused (no augmentation).
        """
        h = self.sccn(features, incidences, adjacencies)
        return self._temporal_step(h, keys, h_state, c_state, local_ids)
