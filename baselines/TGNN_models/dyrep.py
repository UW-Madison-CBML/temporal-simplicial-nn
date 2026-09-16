"""DyRep baseline for dynamic node property prediction."""

from __future__ import annotations

import torch

from datasets import NodeClassificationBundle
from modules.decoder import NodePredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.memory_module import DyRepMemory
from modules.msg_agg import LastAggregator
from modules.msg_func import IdentityMessage
from TGNN_models.base import NodePropertyTrainer, TrainConfig


def build_dyrep_trainer(
    bundle: NodeClassificationBundle,
    config: TrainConfig,
    device: torch.device,
) -> NodePropertyTrainer:
    data = bundle.data
    memory = DyRepMemory(
        data.num_nodes,
        data.msg.size(-1),
        config.memory_dim,
        config.time_dim,
        message_module=IdentityMessage(
            data.msg.size(-1), config.memory_dim, config.time_dim
        ),
        aggregator_module=LastAggregator(),
        memory_updater_type="rnn",
        use_src_emb_in_msg=False,
        use_dst_emb_in_msg=True,
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=config.memory_dim,
        out_channels=config.embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

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
    trainer = build_dyrep_trainer(bundle, config, device)
    return trainer.fit()
