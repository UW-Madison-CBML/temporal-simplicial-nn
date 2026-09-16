"""Build DyGLib backbones and train node-classification models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import NodeClassificationBundle
from dyglib.models.DyGFormer import DyGFormer
from dyglib.models.GraphMixer import GraphMixer
from dyglib.models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from dyglib.models.TGAT import TGAT
from dyglib.models.modules import MLPClassifier
from dyglib.node_eval import evaluate_model_node_classification
from dyglib.utils.DataLoader import get_idx_data_loader
from dyglib.utils.utils import convert_to_gpu, create_optimizer, get_neighbor_sampler
from TGNN_models.base import TrainConfig


MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "tgat": {"num_neighbors": 20, "num_layers": 2, "num_heads": 2, "dropout": 0.2},
    "jodie": {"num_neighbors": 10, "num_layers": 1, "num_heads": 2, "dropout": 0.1},
    "graphmixer": {"num_neighbors": 30, "num_layers": 2, "dropout": 0.1, "time_gap": 2000},
    "dygformer": {
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "patch_size": 8,
        "max_input_sequence_length": 64,
        "channel_embedding_dim": 50,
    },
}


@dataclass
class DyGLibModelConfig(TrainConfig):
  output_dim: int = 100
  time_feat_dim: int = 100
  num_layers: int = 2
  num_heads: int = 2
  dropout: float = 0.1
  time_gap: int = 2000
  patch_size: int = 8
  max_input_sequence_length: int = 64
  channel_embedding_dim: int = 50
  max_batches: int | None = None


def _dyglib_name(model_name: str) -> str:
  mapping = {
      "tgat": "TGAT",
      "jodie": "JODIE",
      "graphmixer": "GraphMixer",
      "dygformer": "DyGFormer",
  }
  key = model_name.lower()
  if key not in mapping:
    raise ValueError(f"Unknown DyGLib model: {model_name}")
  return mapping[key]


def build_dyglib_model(
    model_name: str,
    bundle: NodeClassificationBundle,
    config: DyGLibModelConfig,
    device: torch.device,
) -> Tuple[nn.Sequential, str]:
  dyglib_name = _dyglib_name(model_name)
  defaults = MODEL_DEFAULTS.get(model_name.lower(), {})
  num_neighbors = config.num_neighbors or defaults.get("num_neighbors", 20)
  num_layers = config.num_layers or defaults.get("num_layers", 2)
  num_heads = config.num_heads or defaults.get("num_heads", 2)
  dropout = config.dropout if config.dropout is not None else defaults.get("dropout", 0.1)

  node_raw_features = bundle.node_raw_features
  edge_raw_features = bundle.edge_raw_features
  neighbor_sampler = bundle.train_neighbor_sampler

  if dyglib_name == "TGAT":
    backbone = TGAT(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        time_feat_dim=config.time_feat_dim,
        output_dim=config.output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        device=device,
    )
  elif dyglib_name == "GraphMixer":
    backbone = GraphMixer(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        time_feat_dim=config.time_feat_dim,
        output_dim=config.output_dim,
        num_tokens=num_neighbors,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )
  elif dyglib_name == "DyGFormer":
    backbone = DyGFormer(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        time_feat_dim=config.time_feat_dim,
        channel_embedding_dim=config.channel_embedding_dim,
        output_dim=config.output_dim,
        patch_size=config.patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_input_sequence_length=config.max_input_sequence_length,
        device=device,
    )
  elif dyglib_name == "JODIE":
    shifts = compute_src_dst_node_time_shifts(
        bundle.train_data.src_node_ids,
        bundle.train_data.dst_node_ids,
        bundle.train_data.node_interact_times,
    )
    backbone = MemoryModel(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        time_feat_dim=config.time_feat_dim,
        output_dim=config.output_dim,
        model_name="JODIE",
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        src_node_mean_time_shift=shifts[0],
        src_node_std_time_shift=shifts[1],
        dst_node_mean_time_shift_dst=shifts[2],
        dst_node_std_time_shift=shifts[3],
        device=device,
    )
  else:
    raise ValueError(dyglib_name)

  classifier = MLPClassifier(
      input_dim=config.output_dim,
      output_dim=bundle.num_classes,
      dropout=dropout,
  )
  model = nn.Sequential(backbone, classifier)
  return convert_to_gpu(model, device=device), dyglib_name


def _compute_embeddings(
    model_name: str,
    backbone: nn.Module,
    batch_src_node_ids,
    batch_dst_node_ids,
    batch_node_interact_times,
    batch_edge_ids,
    num_neighbors: int,
    time_gap: int,
):
  if model_name in ["JODIE", "DyRep", "TGN"]:
    return backbone.compute_src_dst_node_temporal_embeddings(
        src_node_ids=batch_src_node_ids,
        dst_node_ids=batch_dst_node_ids,
        node_interact_times=batch_node_interact_times,
        edge_ids=batch_edge_ids,
        edges_are_positive=True,
        num_neighbors=num_neighbors,
    )
  if model_name == "GraphMixer":
    return backbone.compute_src_dst_node_temporal_embeddings(
        src_node_ids=batch_src_node_ids,
        dst_node_ids=batch_dst_node_ids,
        node_interact_times=batch_node_interact_times,
        num_neighbors=num_neighbors,
        time_gap=time_gap,
    )
  if model_name == "DyGFormer":
    return backbone.compute_src_dst_node_temporal_embeddings(
        src_node_ids=batch_src_node_ids,
        dst_node_ids=batch_dst_node_ids,
        node_interact_times=batch_node_interact_times,
    )
  return backbone.compute_src_dst_node_temporal_embeddings(
      src_node_ids=batch_src_node_ids,
      dst_node_ids=batch_dst_node_ids,
      node_interact_times=batch_node_interact_times,
      num_neighbors=num_neighbors,
  )


def run_dyglib_model(
    model_name: str,
    bundle: NodeClassificationBundle,
    config: DyGLibModelConfig,
    device: torch.device,
) -> Dict[str, Any]:
  model, dyglib_name = build_dyglib_model(model_name, bundle, config, device)
  defaults = MODEL_DEFAULTS.get(model_name.lower(), {})
  num_neighbors = config.num_neighbors or defaults.get("num_neighbors", 20)
  time_gap = config.time_gap or defaults.get("time_gap", 2000)

  optimizer = create_optimizer(
      model=model[1],
      optimizer_name="Adam",
      learning_rate=config.lr,
      weight_decay=0.0,
  )
  loss_func = nn.CrossEntropyLoss()
  evaluator = bundle.evaluator
  eval_metric = bundle.eval_metric

  train_curve, val_curve, test_curve = [], [], []
  max_val_score = 0.0
  best_test_idx = 0

  for epoch in range(1, config.epochs + 1):
    model.train()
    if dyglib_name in ["DyRep", "TGAT", "TGN", "GraphMixer", "DyGFormer"]:
      model[0].set_neighbor_sampler(bundle.train_neighbor_sampler)
    if dyglib_name in ["JODIE", "DyRep", "TGN"]:
      model[0].memory_bank.__init_memory_bank__()

    train_losses, train_metrics = [], []
    train_predicts, train_labels = defaultdict(list), defaultdict(list)

    for batch_idx, train_data_indices in enumerate(bundle.train_idx_loader):
      if config.max_batches is not None and batch_idx >= config.max_batches:
        break
      train_data_indices = train_data_indices.numpy()
      train_data = bundle.train_data
      batch_src = train_data.src_node_ids[train_data_indices]
      batch_dst = train_data.dst_node_ids[train_data_indices]
      batch_t = train_data.node_interact_times[train_data_indices]
      batch_edge_ids = train_data.edge_ids[train_data_indices]
      batch_labels = train_data.labels[train_data_indices]
      batch_types = train_data.interact_types[train_data_indices]
      batch_label_times = train_data.node_label_times[train_data_indices]

      train_idx = torch.tensor(np.where(batch_types == "train")[0])

      if dyglib_name in ["JODIE", "DyRep", "TGN"]:
        batch_src_emb, _ = _compute_embeddings(
            dyglib_name, model[0], batch_src, batch_dst, batch_t, batch_edge_ids,
            num_neighbors, time_gap,
        )
      elif len(train_idx) > 0:
        batch_src_emb, _ = _compute_embeddings(
            dyglib_name, model[0], batch_src, batch_dst, batch_t, batch_edge_ids,
            num_neighbors, time_gap,
        )
      else:
        batch_src_emb = None

      if len(train_idx) > 0:
        predicts = model[1](x=batch_src_emb).squeeze(dim=-1)
        labels = torch.from_numpy(batch_labels).float().to(predicts.device)
        loss = loss_func(predicts[train_idx], labels[train_idx])
        train_losses.append(loss.item())
        for idx in train_idx:
          train_predicts[batch_label_times[idx]].append(
              predicts[idx].softmax(dim=0).cpu().detach().numpy()
          )
          train_labels[batch_label_times[idx]].append(labels[idx].cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      if dyglib_name in ["JODIE", "DyRep", "TGN"]:
        model[0].memory_bank.detach_memory_bank()

    for time_slot in train_predicts:
      result = evaluator.eval({
          "y_true": np.stack(train_labels[time_slot], axis=0),
          "y_pred": np.stack(train_predicts[time_slot], axis=0),
          "eval_metric": [eval_metric],
      })
      train_metrics.append(result)

    _, val_metrics = evaluate_model_node_classification(
        model_name=dyglib_name,
        model=model,
        neighbor_sampler=bundle.full_neighbor_sampler,
        evaluate_idx_data_loader=bundle.val_idx_loader,
        evaluate_data=bundle.val_data,
        eval_stage="val",
        eval_metric_name=eval_metric,
        evaluator=evaluator,
        loss_func=loss_func,
        num_neighbors=num_neighbors,
        time_gap=time_gap,
    )
    _, test_metrics = evaluate_model_node_classification(
        model_name=dyglib_name,
        model=model,
        neighbor_sampler=bundle.full_neighbor_sampler,
        evaluate_idx_data_loader=bundle.test_idx_loader,
        evaluate_data=bundle.test_data,
        eval_stage="test",
        eval_metric_name=eval_metric,
        evaluator=evaluator,
        loss_func=loss_func,
        num_neighbors=num_neighbors,
        time_gap=time_gap,
    )

    train_score = np.mean([m[eval_metric] for m in train_metrics]) if train_metrics else 0.0
    val_score = np.mean([m[eval_metric] for m in val_metrics]) if val_metrics else 0.0
    test_score = np.mean([m[eval_metric] for m in test_metrics]) if test_metrics else 0.0
    train_curve.append(train_score)
    val_curve.append(val_score)
    test_curve.append(test_score)

    if val_score > max_val_score:
      max_val_score = val_score
      best_test_idx = epoch - 1

    print("-" * 40)
    print(f"Epoch {epoch:02d}")
    print(f"train {eval_metric}: {train_score:.4f}")
    print(f"val   {eval_metric}: {val_score:.4f}")
    print(f"test  {eval_metric}: {test_score:.4f}")

  summary = {
      "best_val_score": max_val_score,
      "best_val_epoch": best_test_idx + 1,
      "best_test_score": test_curve[best_test_idx],
      "train_curve": train_curve,
      "val_curve": val_curve,
      "test_curve": test_curve,
  }
  print("=" * 40)
  print(f"Best val {eval_metric}: {max_val_score:.4f}")
  print(f"Test {eval_metric} at best val: {summary['best_test_score']:.4f}")
  return summary
