"""Spatio-Temporal SCCN classifier over a sequence of simplicial windows.

A *sample* is a variable-length sequence of simplicial-complex windows
``[S_1, ..., S_T]`` carrying one label. Across a sample the node count is constant, but
edges / triangles (and any higher-rank cells) appear and disappear, and ``T`` differs
between samples. This model:

1. encodes each window's per-rank features to a shared width ``channels``;
2. stacks ``n_layers`` temporal-SCCN blocks of a chosen ``variant`` (each owns a per-rank
   ``LSTMCell`` + a plain ``SCCNLayer`` and threads temporal state across windows):
   - ``"sist"`` -- simultaneous spatial+temporal (SiST-GNN augmentation),
   - ``"temporal_then_spatial"`` -- LSTM then plain SCCN (no augmentation),
   - ``"spatial_then_temporal"`` -- plain SCCN then LSTM (no augmentation);
3. pools cells per rank and aggregates over windows into a single sample embedding;
4. classifies into ``num_classes``.

The temporal state is carried in **identity-keyed** stores and re-indexed to each
window's cell ordering with :func:`align_temporal_features` (zeros for newly-appeared
cells), which is what makes appearing/disappearing edges and triangles work. Gradients
flow across windows (full backprop-through-time, like SiST-GNN).

This module lives in the project root (outside the ``topomodelx`` package) and imports
the layer from :mod:`sccn_temporal_layer`. Run from the project root (or with it on
``PYTHONPATH``).
"""
from typing import Literal

import torch

from sccn_temporal_layer import (
    SiSTSCCNLayer,
    SpatialThenTemporalSCCNLayer,
    TemporalThenSpatialSCCNLayer,
)


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
        Which temporal-SCCN block to stack: ``"sist"`` = simultaneous SiST-GNN augmentation
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
    ) -> None:
        super().__init__()
        self.channels = channels
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

        # readout
        window_dim = channels * (max_rank + 1)
        self.attn_proj = torch.nn.Linear(window_dim, channels)
        self.attn_vec = torch.nn.Linear(channels, 1, bias=False)
        self.classifier = torch.nn.Linear(window_dim, num_classes)

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        for enc in self.encoders.values():
            enc.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
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
        # per-block temporal (h, c) state, identity-keyed, carried across windows.
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

            for block, (h_state, c_state) in zip(self.blocks, states):
                x = block(x, incidences, adjacencies, keys, h_state, c_state)
                # inner update_func / LSTM already nonlinear; just regularize here.
                x = {r: self.dropout(x[r]) for r in ranks}

            window_embeddings.append(
                torch.cat([self._pool_cells(x[r]) for r in ranks], dim=-1)
            )

        z = self._aggregate_windows(torch.stack(window_embeddings, dim=0))
        return self.classifier(z).unsqueeze(0)
