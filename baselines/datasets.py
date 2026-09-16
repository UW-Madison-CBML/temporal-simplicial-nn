"""Node classification / node property prediction datasets (TGB)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import TemporalDataLoader

from config import DATA_ROOT, ensure_data_root
from dyglib.utils.DataLoader import Data, get_idx_data_loader, get_node_classification_tgb_data
from dyglib.utils.utils import get_neighbor_sampler
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
from tgb.utils.info import PROJ_DIR


NODE_CLASSIFICATION_DATASETS: List[str] = [
    "tgbn-genre",
    "tgbn-reddit",
    "tgbn-trade",
    "tgbn-token",
]


@dataclass
class NodeClassificationBundle:
    name: str
    dataset: PyGNodePropPredDataset
    data: torch.Tensor
    train_loader: TemporalDataLoader
    val_loader: TemporalDataLoader
    test_loader: TemporalDataLoader
    evaluator: Evaluator
    num_classes: int
    eval_metric: str
    # Shared DyGLib-format arrays used by TGAT / DyGFormer / GraphMixer / JODIE / NAVIS
    node_raw_features: Optional[np.ndarray] = None
    edge_raw_features: Optional[np.ndarray] = None
    full_data: Optional[Data] = None
    train_data: Optional[Data] = None
    val_data: Optional[Data] = None
    test_data: Optional[Data] = None
    train_neighbor_sampler: Any = None
    full_neighbor_sampler: Any = None
    train_idx_loader: Any = None
    val_idx_loader: Any = None
    test_idx_loader: Any = None


def _tgb_root_arg(data_root: str) -> str:
    data_root = os.path.abspath(data_root)
    proj_dir = os.path.abspath(PROJ_DIR)
    return os.path.relpath(data_root, proj_dir)


def _attach_dyglib_fields(
    bundle: NodeClassificationBundle,
    data_root: str,
    batch_size: int,
) -> NodeClassificationBundle:
    (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        _,
        _,
    ) = get_node_classification_tgb_data(bundle.name, data_root=data_root)

    train_neighbor_sampler = get_neighbor_sampler(train_data, sample_neighbor_strategy="recent")
    full_neighbor_sampler = get_neighbor_sampler(full_data, sample_neighbor_strategy="recent")

    bundle.node_raw_features = node_raw_features
    bundle.edge_raw_features = edge_raw_features
    bundle.full_data = full_data
    bundle.train_data = train_data
    bundle.val_data = val_data
    bundle.test_data = test_data
    bundle.train_neighbor_sampler = train_neighbor_sampler
    bundle.full_neighbor_sampler = full_neighbor_sampler
    bundle.train_idx_loader = get_idx_data_loader(
        list(range(len(train_data.src_node_ids))), batch_size=batch_size, shuffle=False
    )
    bundle.val_idx_loader = get_idx_data_loader(
        list(range(len(val_data.src_node_ids))), batch_size=batch_size, shuffle=False
    )
    bundle.test_idx_loader = get_idx_data_loader(
        list(range(len(test_data.src_node_ids))), batch_size=batch_size, shuffle=False
    )
    return bundle


def get_node_classification_dataset(
    name: str,
    data_root: str = DATA_ROOT,
    batch_size: int = 200,
    device: torch.device | None = None,
    download: bool = True,
) -> NodeClassificationBundle:
    if name not in NODE_CLASSIFICATION_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {NODE_CLASSIFICATION_DATASETS}"
        )

    data_root = ensure_data_root() if data_root == DATA_ROOT else os.path.abspath(data_root)
    os.makedirs(data_root, exist_ok=True)
    tgb_root = _tgb_root_arg(data_root)

    dataset = PyGNodePropPredDataset(name=name, root=tgb_root, download=download)
    data = dataset.get_TemporalData()
    if device is not None:
        data = data.to(device)

    train_data = data[dataset.train_mask]
    val_data = data[dataset.val_mask]
    test_data = data[dataset.test_mask]

    bundle = NodeClassificationBundle(
        name=name,
        dataset=dataset,
        data=data,
        train_loader=TemporalDataLoader(train_data, batch_size=batch_size, shuffle=False),
        val_loader=TemporalDataLoader(val_data, batch_size=batch_size, shuffle=False),
        test_loader=TemporalDataLoader(test_data, batch_size=batch_size, shuffle=False),
        evaluator=Evaluator(name=name),
        num_classes=dataset.num_classes,
        eval_metric=dataset.eval_metric,
    )
    return _attach_dyglib_fields(bundle, data_root=data_root, batch_size=batch_size)


def list_node_classification_datasets() -> Dict[str, str]:
    return {
        "tgbn-genre": "Artist genre prediction on Last.fm (513 classes, NDCG)",
        "tgbn-reddit": "Subreddit participation forecasting (698 classes, NDCG)",
        "tgbn-trade": "Country trade volume forecasting (255 classes, NDCG)",
        "tgbn-token": "Cryptocurrency token price forecasting (1001 classes, NDCG)",
    }
