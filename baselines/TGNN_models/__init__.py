"""
Temporal GNN baselines for TGB node property prediction (node classification).
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

from datasets import NodeClassificationBundle
from TGNN_models import (
    dygformer,
    dyrep,
    graphmixer,
    jodie,
    navis,
    tgat,
    tgn,
)
from TGNN_models.base import TrainConfig

NODE_CLASSIFICATION_MODELS = [
    "tgn",
    "dyrep",
    "tgat",
    "jodie",
    "graphmixer",
    "dygformer",
    "navis",
]

_MODEL_RUNNERS: Dict[str, Callable] = {
    "tgn": tgn.run,
    "dyrep": dyrep.run,
    "tgat": tgat.run,
    "jodie": jodie.run,
    "graphmixer": graphmixer.run,
    "dygformer": dygformer.run,
    "navis": navis.run,
}


def get_model_runner(name: str) -> Callable:
  key = name.lower()
  if key not in _MODEL_RUNNERS:
    raise ValueError(
        f"Unknown model '{name}'. Available: {NODE_CLASSIFICATION_MODELS}"
    )
  return _MODEL_RUNNERS[key]


def run_model(
    model_name: str,
    bundle: NodeClassificationBundle,
    config: TrainConfig,
    device: torch.device | None = None,
):
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  runner = get_model_runner(model_name)
  return runner(bundle, config, device)
