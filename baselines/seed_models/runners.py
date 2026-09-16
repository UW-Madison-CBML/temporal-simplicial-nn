"""Model runners for SEED-VII per-trial temporal baselines."""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

from data_loaders.seed_vii.bundle import SeedVIIBundle
from data_loaders.seed_vii.constants import NUM_EEG_NODES
from modules.decoder import NodePredictor
from modules.emb_module import GraphAttentionEmbedding
from seed_models.base import SeedTrainConfig
from seed_models.dyglib_trainer import run_seed_dyglib_model
from seed_models.pyg_trainer import SeedTrialTrainer

SEED_MODELS = [
    "tgn",
    "dyrep",
    "tgat",
    "jodie",
    "graphmixer",
    "dygformer",
    "navis",
]


def _run_tgn(bundle: SeedVIIBundle, config: SeedTrainConfig, device: torch.device):
    msg_dim = bundle.msg_dim
    memory = TGNMemory(
        NUM_EEG_NODES,
        msg_dim,
        config.memory_dim,
        config.time_dim,
        message_module=IdentityMessage(msg_dim, config.memory_dim, config.time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
    gnn = GraphAttentionEmbedding(
        in_channels=config.memory_dim,
        out_channels=config.embedding_dim,
        msg_dim=msg_dim,
        time_enc=memory.time_enc,
    ).to(device).float()
    node_pred = NodePredictor(config.embedding_dim, bundle.num_classes).to(device)
    trainer = SeedTrialTrainer(bundle, memory, gnn, node_pred, config, device)
    return trainer.fit()


def _run_dyrep(bundle: SeedVIIBundle, config: SeedTrainConfig, device: torch.device):
    from modules.memory_module import DyRepMemory
    from modules.msg_agg import LastAggregator
    from modules.msg_func import IdentityMessage

    msg_dim = bundle.msg_dim
    memory = DyRepMemory(
        NUM_EEG_NODES,
        msg_dim,
        config.memory_dim,
        config.time_dim,
        message_module=IdentityMessage(msg_dim, config.memory_dim, config.time_dim),
        aggregator_module=LastAggregator(),
        memory_updater_type="rnn",
        use_src_emb_in_msg=False,
        use_dst_emb_in_msg=True,
    ).to(device)
    gnn = GraphAttentionEmbedding(
        in_channels=config.memory_dim,
        out_channels=config.embedding_dim,
        msg_dim=msg_dim,
        time_enc=memory.time_enc,
    ).to(device).float()
    node_pred = NodePredictor(config.embedding_dim, bundle.num_classes).to(device)
    trainer = SeedTrialTrainer(bundle, memory, gnn, node_pred, config, device)
    return trainer.fit()


def _run_navis(bundle: SeedVIIBundle, config: SeedTrainConfig, device: torch.device):
    from seed_models.navis_trainer import run_seed_navis

    return run_seed_navis(bundle, config, device)


_RUNNERS: Dict[str, Callable] = {
    "tgn": _run_tgn,
    "dyrep": _run_dyrep,
    "tgat": lambda b, c, d: run_seed_dyglib_model("tgat", b, c, d),
    "jodie": lambda b, c, d: run_seed_dyglib_model("jodie", b, c, d),
    "graphmixer": lambda b, c, d: run_seed_dyglib_model("graphmixer", b, c, d),
    "dygformer": lambda b, c, d: run_seed_dyglib_model("dygformer", b, c, d),
    "navis": _run_navis,
}


def run_seed_model(
    model_name: str,
    bundle: SeedVIIBundle,
    config: SeedTrainConfig,
    device: torch.device | None = None,
):
    key = model_name.lower()
    if key not in _RUNNERS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {SEED_MODELS}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _RUNNERS[key](bundle, config, device)
