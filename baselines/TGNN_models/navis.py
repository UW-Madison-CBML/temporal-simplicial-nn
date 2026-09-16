"""NAVIS-style node affinity model (simplified, no pytorchltr dependency)."""

from __future__ import annotations

import timeit
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import NodeClassificationBundle
from TGNN_models.base import TrainConfig


class SimpleRNNCell(nn.Module):
  def __init__(self, num_classes: int):
    super().__init__()
    self.linx = nn.Linear(num_classes, 1)
    self.linh = nn.Linear(num_classes, 1)
    self.ling = nn.Linear(num_classes, 1)
    self.linxg = nn.Linear(num_classes, 1)
    self.linhg = nn.Linear(num_classes, 1)
    self.s1 = nn.Sigmoid()
    self.s2 = nn.Sigmoid()

  def forward(self, h: torch.Tensor, x: torch.Tensor, g: torch.Tensor):
    z1 = self.s1(self.linx(x) + self.linh(h))
    h_tild = z1 * h + (1 - z1) * x
    z2 = self.s2(self.ling(g) + self.linxg(x) + self.linhg(h))
    return z2 * h_tild + (1 - z2) * x, h_tild


class NAVISModel(nn.Module):
  """Learnable moving-average state-space model from NAVIS (arxiv:2510.06940)."""

  def __init__(self, num_nodes: int, num_classes: int):
    super().__init__()
    self.num_nodes = num_nodes
    self.num_classes = num_classes
    self.node_history = nn.Parameter(torch.zeros(num_nodes, num_classes), requires_grad=False)
    self.node_prev_label = nn.Parameter(torch.zeros(num_nodes, num_classes), requires_grad=False)
    self.semi_labels = nn.Parameter(torch.zeros(num_nodes, num_classes), requires_grad=False)
    self.prev_global_label = nn.Parameter(torch.zeros(1, num_classes), requires_grad=False)
    self.rnn = SimpleRNNCell(num_classes)
    self.out = nn.Linear(num_classes, num_classes)

  def reset(self):
    self.node_history.data.zero_()
    self.node_prev_label.data.zero_()
    self.semi_labels.data.zero_()
    self.prev_global_label.data.zero_()

  def update_semi_labels(self, src: torch.Tensor, dst: torch.Tensor, msg: torch.Tensor):
    vals = msg[:, 0:1].abs().reshape(-1)
    self.semi_labels.data[src, dst] += vals

  def get_semilabels(self, node_ids: torch.Tensor) -> torch.Tensor:
    logits = self.semi_labels[node_ids].clone()
    scale = logits.sum(dim=1, keepdim=True)
    logits = torch.where(scale > 0, logits / scale, logits)
    self.semi_labels.data[node_ids] = 0
    return logits

  def forward(self, node_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    unique_ids = []
    unique_labels = []
  # process unique nodes in order
    for node_id, label in zip(node_ids.tolist(), labels):
      unique_ids.append(node_id)
      unique_labels.append(label)
    node_ids_t = torch.tensor(unique_ids, device=labels.device, dtype=torch.long)
    labels_t = torch.stack(unique_labels)

    semi = self.get_semilabels(node_ids_t) if labels_t.ndim == 2 else labels_t
    state = self.node_history[node_ids_t]
    h_new, h_tild = self.rnn(state, semi, self.prev_global_label.expand(len(node_ids_t), -1))
    self.node_history.data[node_ids_t] = h_tild
    self.node_prev_label.data[node_ids_t] = labels_t
    self.prev_global_label.data = labels_t[-1:].detach()
    return self.out(h_new)


class NAVISTrainer:
  def __init__(self, bundle: NodeClassificationBundle, config: TrainConfig, device: torch.device):
    self.bundle = bundle
    self.dataset = bundle.dataset
    self.train_loader = bundle.train_loader
    self.val_loader = bundle.val_loader
    self.test_loader = bundle.test_loader
    self.evaluator = bundle.evaluator
    self.eval_metric = bundle.eval_metric
    self.config = config
    self.device = device
    self.model = NAVISModel(bundle.data.num_nodes, bundle.num_classes).to(device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

  def _step_loader(self, loader, train: bool):
    if train:
      self.model.train()
      self.model.reset()
    else:
      self.model.eval()

    total_score = 0.0
    num_label_ts = 0
    label_t = self.dataset.get_label_time()

    with torch.set_grad_enabled(train):
      for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        batch = batch.to(self.device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        query_t = batch.t[-1]

        if query_t > label_t:
          label_tuple = self.dataset.get_node_label(query_t)
          if label_tuple is None:
            break
          label_ts, label_srcs, labels = label_tuple
          label_t = self.dataset.get_label_time()
          label_srcs = label_srcs.to(self.device)
          labels = labels.to(self.device)

          previous_day_mask = batch.t < label_t
          self.model.update_semi_labels(
              src[previous_day_mask], dst[previous_day_mask], msg[previous_day_mask]
          )
          src, dst, t, msg = (
              src[~previous_day_mask], dst[~previous_day_mask],
              t[~previous_day_mask], msg[~previous_day_mask],
          )

          if train:
            self.optimizer.zero_grad()
          pred = self.model(label_srcs, labels)
          loss = self.criterion(pred, labels)
          if train:
            loss.backward()
            self.optimizer.step()

          result = self.evaluator.eval({
              "y_true": labels.cpu().detach().numpy(),
              "y_pred": pred.cpu().detach().numpy(),
              "eval_metric": [self.eval_metric],
          })
          total_score += result[self.eval_metric]
          num_label_ts += 1

        self.model.update_semi_labels(src, dst, msg)

    return {self.eval_metric: total_score / max(num_label_ts, 1)}

  def fit(self) -> Dict[str, Any]:
    train_curve, val_curve, test_curve = [], [], []
    max_val_score = 0.0
    best_test_idx = 0

    for epoch in range(1, self.config.epochs + 1):
      start = timeit.default_timer()
      train_metrics = self._step_loader(self.train_loader, train=True)
      train_time = timeit.default_timer() - start
      self.dataset.reset_label_time()

      start = timeit.default_timer()
      val_metrics = self._step_loader(self.val_loader, train=False)
      val_time = timeit.default_timer() - start
      self.dataset.reset_label_time()

      if val_metrics[self.eval_metric] > max_val_score:
        max_val_score = val_metrics[self.eval_metric]
        best_test_idx = epoch - 1

      start = timeit.default_timer()
      test_metrics = self._step_loader(self.test_loader, train=False)
      test_time = timeit.default_timer() - start
      self.dataset.reset_label_time()

      train_curve.append(train_metrics[self.eval_metric])
      val_curve.append(val_metrics[self.eval_metric])
      test_curve.append(test_metrics[self.eval_metric])

      print("-" * 40)
      print(f"Epoch {epoch:02d}")
      print(f"train: {train_metrics} ({train_time:.1f}s)")
      print(f"val:   {val_metrics} ({val_time:.1f}s)")
      print(f"test:  {test_metrics} ({test_time:.1f}s)")

    return {
        "best_val_score": max_val_score,
        "best_val_epoch": best_test_idx + 1,
        "best_test_score": test_curve[best_test_idx],
        "train_curve": train_curve,
        "val_curve": val_curve,
        "test_curve": test_curve,
    }


def run(bundle: NodeClassificationBundle, config: TrainConfig, device: torch.device):
  return NAVISTrainer(bundle, config, device).fit()
