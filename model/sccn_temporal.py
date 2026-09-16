"""Spatio-Temporal SCCN classifier over a sequence of simplicial windows.

A *sample* is a variable-length sequence of simplicial-complex windows
``[S_1, ..., S_T]`` carrying one label. Across a sample the node count is constant, but
edges / triangles (and any higher-rank cells) appear and disappear, and ``T`` differs
between samples. This model:

1. encodes each window's per-rank features to a shared width ``channels``;
2. stacks ``n_layers`` temporal-SCCN blocks of a chosen ``variant`` (each owns a per-rank
   ``LSTMCell`` + a plain ``SCCNLayer`` and threads temporal state across windows):
   - ``"sist"`` -- simultaneous spatial+temporal augmentation,
   - ``"temporal_then_spatial"`` -- LSTM then plain SCCN (no augmentation),
   - ``"spatial_then_temporal"`` -- plain SCCN then LSTM (no augmentation);
3. pools cells per rank and aggregates over windows into a single sample embedding;
4. classifies into ``num_classes``.

The temporal state is carried in **identity-keyed** stores and re-indexed to each
window's cell ordering with :func:`align_temporal_features` (zeros for newly-appeared
cells), which is what makes appearing/disappearing edges and triangles work. Gradients
flow across windows (full backprop-through-time).

This module lives in the project root (outside the ``topomodelx`` package) and imports
the layer from :mod:`sccn_temporal_layer`. Run from the project root (or with it on
``PYTHONPATH``).
"""
import math
from typing import Literal

import torch

from sccn_temporal_layer import (
    SiSTSCCNLayer,
    SpatialThenTemporalSCCNLayer,
    TemporalThenSpatialSCCNLayer,
)


def block_diag_sparse(mats: list[torch.Tensor]) -> torch.Tensor:
    """Stack sparse matrices along the diagonal into one big sparse matrix.

    ``blockdiag(A_1..A_B) @ cat(x_1..x_B)`` equals ``cat(A_1 @ x_1, ..., A_B @ x_B)``,
    which is what lets a whole minibatch of complexes go through the layers as a single
    graph -- the standard graph-batching trick.

    Deliberately built with a fixed number of tensor ops rather than a per-matrix loop
    of shifts: this model is bound by kernel-launch overhead, so a Python loop issuing
    B adds would give back most of the benefit of batching in the first place. The
    per-matrix offsets are applied in one vectorized ``repeat_interleave`` + add.
    """
    if not mats:
        raise ValueError("block_diag_sparse needs at least one matrix")
    mats = [m.coalesce() for m in mats]
    device = mats[0].device
    rows = torch.tensor([m.size(0) for m in mats], device=device)
    cols = torch.tensor([m.size(1) for m in mats], device=device)
    nnz = torch.tensor([m._nnz() for m in mats], device=device)
    total_rows = int(rows.sum())
    total_cols = int(cols.sum())
    if int(nnz.sum()) == 0:
        return torch.sparse_coo_tensor(
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, device=device, dtype=mats[0].dtype),
            (total_rows, total_cols),
        ).coalesce()

    # exclusive cumulative sums = where each block starts
    row_off = torch.cumsum(rows, 0) - rows
    col_off = torch.cumsum(cols, 0) - cols
    idx = torch.cat([m._indices() for m in mats], dim=1)
    vals = torch.cat([m._values() for m in mats], dim=0)
    shift = torch.stack(
        [torch.repeat_interleave(row_off, nnz), torch.repeat_interleave(col_off, nnz)]
    )
    return torch.sparse_coo_tensor(
        idx + shift, vals, (total_rows, total_cols)
    ).coalesce()


def segment_lengths(counts: torch.Tensor) -> torch.Tensor:
    """``[2,3]`` -> ``[0,0,1,1,1]``: which segment each concatenated row belongs to."""
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts
    )


class CellLearner(torch.nn.Module):
    """Learnable per-channel identity embeddings, composed to cells of any rank.

    The window features are purely *signal* (Welch band power); nothing tells the model
    that a given node is Fp1 rather than Oz, or that a given edge joins those two
    specific electrodes. This module supplies that missing "innate" component: one
    learnable embedding per EEG channel, pooled over a cell's constituent channels and
    passed through a per-rank projection.

    Because every cell identity is a tuple of channel indices, one channel table covers
    all three learners -- rank 0 is the node learner (a channel's own embedding), rank 1
    the edge learner (its two endpoints), rank 2 the triangle learner (its three
    corners). Sharing the table is deliberate: an edge between Fp1 and Fp2 should be
    informed by both electrodes' identities, and it means edges/triangles never seen
    during training still get a sensible embedding.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (the identity vocabulary).
    channels : int
        Shared hidden width.
    max_rank : int
        Maximum cell rank; a projection is created for every rank ``0..max_rank``.
    """

    def __init__(self, n_channels: int, channels: int, max_rank: int) -> None:
        super().__init__()
        self.channels = channels
        self.ranks = [f"rank_{r}" for r in range(max_rank + 1)]
        self.node_emb = torch.nn.Embedding(n_channels, channels)
        self.rank_proj = torch.nn.ModuleDict(
            {r: torch.nn.Linear(channels, channels) for r in self.ranks}
        )

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.node_emb.reset_parameters()
        for proj in self.rank_proj.values():
            proj.reset_parameters()

    def forward(self, cell_nodes, device, dtype):
        """Return ``{rank: (n_r, channels)}`` learned identity embeddings."""
        out = {}
        for r in self.ranks:
            idx = cell_nodes[r]
            if idx.numel() == 0:
                out[r] = torch.zeros((0, self.channels), device=device, dtype=dtype)
                continue
            pooled = self.node_emb(idx).mean(dim=1)
            out[r] = self.rank_proj[r](pooled)
        return out


class ChannelPositionEncoder(torch.nn.Module):
    """Encode where a cell physically sits on the scalp.

    The complexes carry no geometry -- an edge is just a pair of indices, so a
    short-range frontal connection and a long-range fronto-occipital one are
    indistinguishable to the model. This projects the montage coordinates of a cell's
    channels into the hidden width, using two pieces:

    - **centroid**: the mean coordinate of the cell's channels (*where* it is);
    - **spread**: the per-axis standard deviation across those channels (*how far
      apart* they are -- identically zero for rank-0 nodes, and what distinguishes a
      local edge from a long-range one).

    Positions are a fixed (non-learned) buffer; only the projections train.

    Parameters
    ----------
    position_features : torch.Tensor, shape = (n_channels, D)
        Static per-channel coordinate features (see ``scripts/channel_positions.py``).
    channels : int
        Shared hidden width.
    max_rank : int
        Maximum cell rank; a projection is created for every rank ``0..max_rank``.
    """

    def __init__(
        self, position_features: torch.Tensor, channels: int, max_rank: int
    ) -> None:
        super().__init__()
        self.channels = channels
        self.ranks = [f"rank_{r}" for r in range(max_rank + 1)]
        self.register_buffer("positions", position_features.clone(), persistent=True)
        in_dim = 2 * position_features.size(-1)  # centroid ++ spread
        self.rank_proj = torch.nn.ModuleDict(
            {r: torch.nn.Linear(in_dim, channels) for r in self.ranks}
        )

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        for proj in self.rank_proj.values():
            proj.reset_parameters()

    def forward(self, cell_nodes, device, dtype):
        """Return ``{rank: (n_r, channels)}`` positional embeddings."""
        out = {}
        for r in self.ranks:
            idx = cell_nodes[r]
            if idx.numel() == 0:
                out[r] = torch.zeros((0, self.channels), device=device, dtype=dtype)
                continue
            coords = self.positions[idx].to(dtype)  # (n_r, arity, D)
            centroid = coords.mean(dim=1)
            spread = coords.std(dim=1, unbiased=False)
            out[r] = self.rank_proj[r](torch.cat([centroid, spread], dim=-1))
        return out


class TemporalPhaseEncoder(torch.nn.Module):
    """Encode *when* a window occurs within its trial (early / mid / late).

    The recurrent state already models how the signal *evolves*, but nothing tells the
    model where along the trial it currently is -- window 2 and window 30 are processed
    identically apart from accumulated history. Emotional response has structure over a
    trial (onset, build-up, sustained response), so the absolute phase is informative.

    Takes the window's normalized position ``t_frac in [0, 1]`` and lifts it into a
    sinusoidal basis (the same trick as transformer positional encodings, but over a
    continuous fraction so trials of different lengths stay comparable), then projects
    it to the hidden width. The result is one vector per window, broadcast to every
    cell of every rank.

    Parameters
    ----------
    channels : int
        Shared hidden width.
    n_bands : int, default=8
        Number of geometrically-spaced sinusoidal frequencies.
    """

    def __init__(self, channels: int, n_bands: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.n_bands = n_bands
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(n_bands, dtype=torch.float32), persistent=True
        )
        self.proj = torch.nn.Linear(2 * n_bands + 1, channels)

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.proj.reset_parameters()

    def forward(self, t_frac: float, device, dtype) -> torch.Tensor:
        """Return a ``(channels,)`` embedding of the window's phase in the trial."""
        t = torch.as_tensor([float(t_frac)], device=device, dtype=dtype)
        scaled = t * self.freqs.to(device=device, dtype=dtype) * math.pi
        feats = torch.cat([t, torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.proj(feats)

    def forward_batch(self, t_fracs, device, dtype) -> torch.Tensor:
        """Return ``(B, channels)`` for a batch of window phases.

        Trials in a batch have different lengths, so at the same time-step ``t`` each
        one sits at a *different* fraction through its own trial -- the phase has to be
        computed per trial, not shared across the batch.
        """
        t = torch.as_tensor(t_fracs, device=device, dtype=dtype).reshape(-1, 1)
        scaled = t * self.freqs.to(device=device, dtype=dtype) * math.pi
        feats = torch.cat([t, torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.proj(feats)


class EyeEncoder(torch.nn.Module):
    """Encode the per-window eye-tracking vector, and optionally model it over time.

    Eye features are a *window-level* modality: one vector per window, unlike the EEG
    features which live on cells. The encoded vector is concatenated onto each window's
    pooled EEG embedding, and the shared window-aggregation then handles both; eye never
    enters the recurrence.

    (A "tower" variant -- eye on its own ``LSTMCell``, fused only at trial level -- was
    tested and dropped: it scored 0.599 +/- 0.028 accuracy against 0.775 +/- 0.007 for
    this window fusion and 0.701 +/- 0.014 for EEG alone, i.e. it cost more in
    optimization difficulty than the eye branch contributed.)

    The raw features span ~7 orders of magnitude across dimensions (per-dim std from
    0.003 to 10,640), so a fixed standardization is applied first. Those statistics
    must come from training data only -- see ``eye_feature_stats`` in the trainer.

    Parameters
    ----------
    in_dim : int
        Width of the raw eye vector (33 for this extract).
    channels : int
        Output width, matching the model's shared hidden width.
    mode : str
        Fusion mode; ``"window"`` is the only one supported.
    stats : tuple[torch.Tensor, torch.Tensor], optional
        ``(mean, std)`` per input dimension. Identity if omitted.
    """

    def __init__(self, in_dim: int, channels: int, mode: str = "window", stats=None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.channels = channels
        self.mode = mode
        mean = stats[0] if stats is not None else torch.zeros(in_dim)
        std = stats[1] if stats is not None else torch.ones(in_dim)
        # clamp guards the near-constant dimensions from exploding after division
        self.register_buffer("mean", mean.clone().float(), persistent=True)
        self.register_buffer("std", std.clone().float().clamp(min=1e-6), persistent=True)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, channels),
        )

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        for m in self.proj:
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()

    def fill_missing(self, eye: torch.Tensor) -> torch.Tensor:
        """Substitute the training mean for a window with no eye measurement.

        Such a window standardizes to all-zeros, i.e. "no information", rather than a
        literal zero reading which would be a spurious measurement.
        """
        if eye.numel() == 0:
            return self.mean.detach().clone()
        return eye

    def encode(self, eye: torch.Tensor) -> torch.Tensor:
        """``(B, in_dim)`` raw eye features -> ``(B, channels)``."""
        x = (eye - self.mean.to(eye.dtype)) / self.std.to(eye.dtype)
        return self.proj(x)


class SCCNTemporalClassifier(torch.nn.Module):
    """Classify a sequence of simplicial windows into ``num_classes`` classes.

    Parameters
    ----------
    in_channels : int or dict[str, int]
        Input feature dimension per rank. An ``int`` applies to every rank; a dict keyed
        by ``"rank_{r}"`` gives a per-rank dimension.
    channels : int
        Shared hidden width used by every rank (SCCN requires one width).
    max_rank : int
        Maximum rank of the cells in the complex (``>= 1``; 1 = nodes+edges, 2 =
        +triangles, 3 = +tetrahedra, ...).
    num_classes : int
        Number of output classes.
    n_layers : int, default=2
        Number of stacked temporal-SCCN blocks (each owns its per-rank LSTM + SCCNLayer).
    variant : {"sist", "temporal_then_spatial", "spatial_then_temporal"}, default="sist"
        Which temporal-SCCN block to stack: ``"sist"`` = simultaneous augmentation
        (:class:`SiSTSCCNLayer`); ``"temporal_then_spatial"`` = LSTM then plain SCCN
        (:class:`TemporalThenSpatialSCCNLayer`); ``"spatial_then_temporal"`` = plain SCCN
        then LSTM (:class:`SpatialThenTemporalSCCNLayer`). The decoupled variants do not
        augment the matrices.
    aggr_func : {"mean", "sum"}, default="sum"
        Aggregation inside the wrapped SCCN layers.
    update_func : {"relu", "sigmoid", "tanh", None}, default="sigmoid"
        Activation inside the wrapped SCCN layers.
    temporal_self_loop : bool, default=True
        Own-past -> own-present self-loop. Only used by ``variant="sist"`` (the
        augmentation knob); ignored by the decoupled variants.
    present_self_loop : bool, default=True
        Own-current -> own-present self-loop. Only used by ``variant="sist"``; ignored by
        the decoupled variants.
    cell_pool : {"mean", "sum", "max"}, default="mean"
        How to pool the cells of a rank into a per-rank window vector.
    window_agg : {"mean", "sum", "max", "last", "attention"}, default="mean"
        How to aggregate the per-window embeddings into one sample embedding.
    dropout : float, default=0.0
        Dropout applied to the per-rank features between layers.
    cell_learner : bool, default=False
        Add learnable per-channel identity embeddings for every rank
        (:class:`CellLearner` -- the node / edge / triangle learners).
    position_features : torch.Tensor, optional
        ``(n_channels, D)`` static electrode coordinates. When given, cells are also
        encoded by scalp geometry (:class:`ChannelPositionEncoder`).
    temporal_phase : bool, default=False
        Add a sinusoidal encoding of each window's position within its trial
        (:class:`TemporalPhaseEncoder`).
    n_channels : int, default=62
        Size of the channel-identity vocabulary used by ``cell_learner``.
    cache_augmented : bool, default=True
        Memoize the SiST augmented matrices on each window dict. They depend only on
        the window's topology, so this trades GPU memory (roughly a few hundred MB
        across the full dataset) for dropping a per-window ``coalesce()`` from every
        epoch. Set False on a memory-tight GPU; results are identical either way.
    """

    def __init__(
        self,
        in_channels,
        channels,
        max_rank,
        num_classes,
        n_layers: int = 2,
        variant: Literal[
            "sist", "temporal_then_spatial", "spatial_then_temporal"
        ] = "sist",
        aggr_func: Literal["mean", "sum"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh"] | None = "sigmoid",
        temporal_self_loop: bool = True,
        present_self_loop: bool = True,
        cell_pool: Literal["mean", "sum", "max"] = "mean",
        window_agg: Literal["mean", "sum", "max", "last", "attention"] = "mean",
        dropout: float = 0.0,
        cell_learner: bool = False,
        position_features: torch.Tensor | None = None,
        temporal_phase: bool = False,
        n_channels: int = 62,
        cache_augmented: bool = True,
        eye_dim: int = 0,
        eye_fusion: str = "none",
        eye_stats=None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.cache_augmented = cache_augmented
        self.max_rank = max_rank
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.cell_pool = cell_pool
        self.window_agg = window_agg
        self.ranks = [f"rank_{r}" for r in range(max_rank + 1)]

        if isinstance(in_channels, int):
            in_channels = {r: in_channels for r in self.ranks}

        # per-rank input encoders -> shared `channels`
        self.encoders = torch.nn.ModuleDict(
            {r: torch.nn.Linear(in_channels[r], channels) for r in self.ranks}
        )

        # stacked temporal-SCCN blocks of the chosen variant (each owns its LSTM + SCCN)
        block_classes = {
            "sist": SiSTSCCNLayer,
            "temporal_then_spatial": TemporalThenSpatialSCCNLayer,
            "spatial_then_temporal": SpatialThenTemporalSCCNLayer,
        }
        if variant not in block_classes:
            raise ValueError(
                f"variant must be one of {list(block_classes)}, got {variant!r}"
            )
        self.variant = variant
        block_cls = block_classes[variant]
        # self-loop knobs only apply to the simultaneous (augmentation) variant
        sl_kwargs = (
            {
                "temporal_self_loop": temporal_self_loop,
                "present_self_loop": present_self_loop,
            }
            if variant == "sist"
            else {}
        )
        self.blocks = torch.nn.ModuleList(
            block_cls(
                channels=channels,
                max_rank=max_rank,
                aggr_func=aggr_func,
                update_func=update_func,
                **sl_kwargs,
            )
            for _ in range(n_layers)
        )

        self.dropout = torch.nn.Dropout(dropout)

        # optional priors injected alongside the encoded signal features. All three add
        # into the shared `channels` width (rather than concatenating) because SCCN
        # requires one uniform width across ranks.
        self.cell_learner = (
            CellLearner(n_channels, channels, max_rank) if cell_learner else None
        )
        self.position_encoder = (
            ChannelPositionEncoder(position_features, channels, max_rank)
            if position_features is not None
            else None
        )
        self.temporal_phase = (
            TemporalPhaseEncoder(channels) if temporal_phase else None
        )

        # Eye-tracking branch: widens each window embedding by `channels` before the
        # shared window aggregation.
        self.eye_fusion = eye_fusion if (eye_dim > 0 and eye_fusion != "none") else "none"
        self.eye_encoder = (
            EyeEncoder(eye_dim, channels, self.eye_fusion, eye_stats)
            if self.eye_fusion != "none"
            else None
        )

        # readout
        window_dim = channels * (max_rank + 1)
        if self.eye_fusion == "window":
            window_dim += channels
        self.attn_proj = torch.nn.Linear(window_dim, channels)
        self.attn_vec = torch.nn.Linear(channels, 1, bias=False)
        self.classifier = torch.nn.Linear(window_dim, num_classes)

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        for enc in self.encoders.values():
            enc.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        for extra in (self.cell_learner, self.position_encoder, self.temporal_phase, self.eye_encoder):
            if extra is not None:
                extra.reset_parameters()
        self.attn_proj.reset_parameters()
        self.attn_vec.reset_parameters()
        self.classifier.reset_parameters()

    def _pool_cells(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a rank's cell features ``(n_r, channels)`` into ``(channels,)``.

        Returns zeros when the rank is empty in this window (``n_r == 0``).
        """
        if x.size(0) == 0:
            return x.new_zeros(self.channels)
        if self.cell_pool == "sum":
            return x.sum(dim=0)
        if self.cell_pool == "max":
            return x.max(dim=0).values
        return x.mean(dim=0)

    def _aggregate_windows(self, window_embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate per-window embeddings ``(T, window_dim)`` into ``(window_dim,)``."""
        if self.window_agg == "last":
            return window_embeddings[-1]
        if self.window_agg == "sum":
            return window_embeddings.sum(dim=0)
        if self.window_agg == "max":
            return window_embeddings.max(dim=0).values
        if self.window_agg == "attention":
            scores = self.attn_vec(torch.tanh(self.attn_proj(window_embeddings)))  # (T, 1)
            weights = torch.softmax(scores, dim=0)
            return (weights * window_embeddings).sum(dim=0)
        return window_embeddings.mean(dim=0)

    def forward_batch(self, batch):
        """Forward a whole minibatch of samples as one block-diagonal graph.

        Parameters
        ----------
        batch : list[list[dict]]
            ``B`` window sequences, each in the format :meth:`forward` accepts. Lengths
            may differ.

        Returns
        -------
        torch.Tensor, shape = (B, num_classes)
            Class logits, one row per sample, in the order given.

        Notes
        -----
        Mathematically identical to calling :meth:`forward` on each sample: a
        block-diagonal neighbourhood matrix applied to concatenated features gives
        exactly the per-sample results concatenated, and every other step
        (pooling, aggregation, classification) is applied per sample. What changes is
        that one time-step costs a fixed handful of GPU kernels regardless of ``B``,
        instead of ``B`` separate passes -- the point, since this model is bound by
        launch overhead rather than arithmetic.

        Samples have different lengths, so at time-step ``t`` only the samples still
        running participate; the rest have already contributed all their windows.
        """
        ranks = self.ranks
        n_samples = len(batch)
        lengths = [len(w) for w in batch]
        max_t = max(lengths)
        ref = next(iter(batch[0][0]["features"].values()))
        device, dtype = ref.device, ref.dtype

        use_dense = "local_ids" in batch[0][0] and "n_cells" in batch[0][0]
        if use_dense:
            # One dense recurrent state spanning the batch; each sample occupies a
            # contiguous id range, so its cells never collide with another's.
            per_sample = [w[0]["n_cells"] for w in batch]
            base = {r: [] for r in ranks}
            total = {}
            for r in ranks:
                off = 0
                for nc in per_sample:
                    base[r].append(off)
                    off += nc[r]
                total[r] = off
            states = [block.init_state(total, device, dtype) for block in self.blocks]
        else:
            states = [block.init_state() for block in self.blocks]

        per_sample_windows: list[list[torch.Tensor]] = [[] for _ in range(n_samples)]

        for t in range(max_t):
            active = [b for b in range(n_samples) if lengths[b] > t]
            ws = [batch[b][t] for b in active]

            counts = {
                r: torch.tensor(
                    [w["features"][r].size(0) for w in ws], device=device
                )
                for r in ranks
            }
            feats = {r: torch.cat([w["features"][r] for w in ws], dim=0) for r in ranks}
            x = {r: self.encoders[r](feats[r]) for r in ranks}

            if self.cell_learner is not None or self.position_encoder is not None:
                cell_nodes = {
                    r: torch.cat([w["cell_nodes"][r] for w in ws], dim=0) for r in ranks
                }
                if self.cell_learner is not None:
                    learned = self.cell_learner(cell_nodes, device, dtype)
                    x = {r: x[r] + learned[r] for r in ranks}
                if self.position_encoder is not None:
                    pos = self.position_encoder(cell_nodes, device, dtype)
                    x = {r: x[r] + pos[r] for r in ranks}
            if self.temporal_phase is not None:
                phase = self.temporal_phase.forward_batch(
                    [w.get("t_frac", 0.0) for w in ws], device, dtype
                )
                # each sample's own phase, repeated across the cells it contributes
                x = {r: x[r] + phase[segment_lengths(counts[r])] for r in ranks}

            adjacencies = {
                r: block_diag_sparse([w["adjacencies"][r] for w in ws]) for r in ranks
            }
            incidences = {
                k: block_diag_sparse([w["incidences"][k] for w in ws])
                for k in ws[0]["incidences"]
            }
            keys = {r: [k for w in ws for k in w["keys"][r]] for r in ranks}

            local_ids = None
            if use_dense:
                local_ids = {}
                for r in ranks:
                    shift = torch.tensor(
                        [base[r][b] for b in active], device=device
                    ).repeat_interleave(counts[r])
                    local_ids[r] = (
                        torch.cat([w["local_ids"][r] for w in ws], dim=0) + shift
                    )

            for block, (h_state, c_state) in zip(self.blocks, states):
                x = block(x, incidences, adjacencies, keys, h_state, c_state, local_ids, None)
                x = {r: self.dropout(x[r]) for r in ranks}

            # pool each sample's cells separately (segment mean/sum/max over the
            # concatenated rows), matching _pool_cells per sample
            pooled = []
            for r in ranks:
                pooled.append(self._pool_cells_segments(x[r], counts[r], len(ws)))
            if self.eye_encoder is not None:
                eye_raw = torch.stack(
                    [self.eye_encoder.fill_missing(w["eye"]) for w in ws], dim=0
                )
                pooled.append(self.eye_encoder.encode(eye_raw))  # (n_active, channels)
            emb = torch.cat(pooled, dim=-1)  # (n_active, window_dim)
            for j, b in enumerate(active):
                per_sample_windows[b].append(emb[j])

        logits = []
        for b in range(n_samples):
            z = self._aggregate_windows(torch.stack(per_sample_windows[b], dim=0))
            logits.append(self.classifier(z))
        return torch.stack(logits, dim=0)

    def _pool_cells_segments(
        self, x: torch.Tensor, counts: torch.Tensor, n_seg: int
    ) -> torch.Tensor:
        """Per-sample :meth:`_pool_cells` over concatenated rows.

        ``counts[i]`` rows of ``x`` belong to sample ``i``. A sample with zero cells of
        this rank yields zeros, matching :meth:`_pool_cells`.
        """
        if x.size(0) == 0:
            return x.new_zeros(n_seg, self.channels)
        seg = segment_lengths(counts)
        if self.cell_pool == "max":
            out = x.new_full((n_seg, self.channels), float("-inf"))
            out = out.index_reduce(0, seg, x, "amax", include_self=True)
            # segments with no rows stay at -inf; they should be zero
            return torch.where(
                counts.unsqueeze(1) > 0, out, torch.zeros_like(out)
            )
        summed = x.new_zeros(n_seg, self.channels).index_add(0, seg, x)
        if self.cell_pool == "sum":
            return summed
        return summed / counts.clamp(min=1).unsqueeze(1).to(x.dtype)

    def forward(self, windows):
        """Forward pass over one sample.

        Parameters
        ----------
        windows : list[dict]
            Sequence of ``T`` windows. Each window is a dict with:

            - ``"features"`` : ``{f"rank_{r}": Tensor(n_r, in_channels_r)}`` for
              ``r = 0..max_rank``;
            - ``"adjacencies"`` : ``{f"rank_{r}": sparse(n_r, n_r)}`` for
              ``r = 0..max_rank`` (original ``A_r``);
            - ``"incidences"`` : ``{f"rank_{r}": sparse(n_{r-1}, n_r)}`` for
              ``r = 1..max_rank`` (original ``B_r``);
            - ``"keys"`` : ``{f"rank_{r}": list}`` of length ``n_r`` of stable cell
              identities (same row order as the matrices), used to align temporal state
              across windows. Nodes may use constant ids; edges/triangles e.g.
              ``tuple(sorted(simplex))``.

            Empty ranks in a window are allowed: pass ``(0, C)`` features, ``(0, 0)``
            adjacency, ``(n_{r-1}, 0)`` incidence and ``keys=[]``.

        Returns
        -------
        torch.Tensor, shape = (1, num_classes)
            Class logits for the sample. Batch by looping samples and concatenating.
        """
        ranks = self.ranks
        # Per-block temporal (h, c) state carried across windows. When the windows
        # carry trial-stable cell ids, use the dense representation (one gather /
        # scatter per rank per window instead of a Python loop over every cell); it
        # produces identical values to the identity-keyed dicts.
        first = windows[0] if windows else None
        use_dense = bool(first) and "local_ids" in first and "n_cells" in first
        if use_dense:
            ref = next(iter(first["features"].values()))
            states = [
                block.init_state(first["n_cells"], ref.device, ref.dtype)
                for block in self.blocks
            ]
        else:
            states = [block.init_state() for block in self.blocks]

        window_embeddings = []
        for w in windows:
            features, adjacencies, incidences, keys = (
                w["features"],
                w["adjacencies"],
                w["incidences"],
                w["keys"],
            )

            # encode raw per-rank features to the shared width
            x = {r: self.encoders[r](features[r]) for r in ranks}

            # inject the optional priors: who this cell is (learned channel identity),
            # where it sits on the scalp, and when in the trial this window occurs.
            if self.cell_learner is not None or self.position_encoder is not None:
                cell_nodes = w["cell_nodes"]
                ref = x[ranks[0]]
                if self.cell_learner is not None:
                    learned = self.cell_learner(cell_nodes, ref.device, ref.dtype)
                    x = {r: x[r] + learned[r] for r in ranks}
                if self.position_encoder is not None:
                    pos = self.position_encoder(cell_nodes, ref.device, ref.dtype)
                    x = {r: x[r] + pos[r] for r in ranks}
            if self.temporal_phase is not None:
                ref = x[ranks[0]]
                phase = self.temporal_phase(
                    w.get("t_frac", 0.0), ref.device, ref.dtype
                )
                x = {r: x[r] + phase for r in ranks}

            local_ids = w.get("local_ids") if use_dense else None
            # Memoize the parameter-independent augmented matrices on the window dict
            # itself, so they survive across blocks and across epochs.
            aug_cache = w if self.cache_augmented else None
            for block, (h_state, c_state) in zip(self.blocks, states):
                x = block(
                    x, incidences, adjacencies, keys, h_state, c_state,
                    local_ids, aug_cache,
                )
                # inner update_func / LSTM already nonlinear; just regularize here.
                x = {r: self.dropout(x[r]) for r in ranks}

            parts = [self._pool_cells(x[r]) for r in ranks]
            if self.eye_encoder is not None:
                parts.append(
                    self.eye_encoder.encode(
                        self.eye_encoder.fill_missing(w["eye"]).unsqueeze(0)
                    ).squeeze(0)
                )
            window_embeddings.append(torch.cat(parts, dim=-1))

        z = self._aggregate_windows(torch.stack(window_embeddings, dim=0))
        return self.classifier(z).unsqueeze(0)
