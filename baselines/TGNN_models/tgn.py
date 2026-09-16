"""TGN baseline for dynamic node property prediction."""

from __future__ import annotations

import torch
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

from datasets import NodeClassificationBundle
from modules.decoder import NodePredictor
from modules.emb_module import GraphAttentionEmbedding
from TGNN_models.base import NodePropertyTrainer, TrainConfig


def build_tgn_trainer(
    bundle: NodeClassificationBundle,
    config: TrainConfig,
    device: torch.device,
) -> NodePropertyTrainer:
    data = bundle.data
    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        config.memory_dim,
        config.time_dim,
        message_module=IdentityMessage(
            data.msg.size(-1), config.memory_dim, config.time_dim
        ),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=config.memory_dim,
        out_channels=config.embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device).float()

    node_pred = NodePredictor(
        in_dim=config.embedding_dim,
        out_dim=bundle.num_classes,
    ).to(device)

    return NodePropertyTrainer(
        bundle=bundle,
        memory=memory,
        gnn=gnn,
        node_pred=node_pred,
        config=config,
        device=device,
    )


def run(bundle: NodeClassificationBundle, config: TrainConfig, device: torch.device):
    trainer = build_tgn_trainer(bundle, config, device)
    return trainer.fit()
