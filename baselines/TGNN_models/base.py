"""Shared training loop for TGB node property prediction models."""

from __future__ import annotations

import timeit
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import NodeClassificationBundle


@dataclass
class TrainConfig:
    lr: float = 1e-4
    epochs: int = 50
    memory_dim: int = 100
    time_dim: int = 100
    embedding_dim: int = 100
    num_neighbors: int = 10
    seed: int = 1
    max_batches: int | None = None


class NodePropertyTrainer:
    """Temporal GNN trainer for dynamic node property prediction."""

    def __init__(
        self,
        bundle: NodeClassificationBundle,
        memory: nn.Module,
        gnn: nn.Module,
        node_pred: nn.Module,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.bundle = bundle
        self.dataset = bundle.dataset
        self.data = bundle.data
        self.train_loader = bundle.train_loader
        self.val_loader = bundle.val_loader
        self.test_loader = bundle.test_loader
        self.evaluator = bundle.evaluator
        self.eval_metric = bundle.eval_metric

        self.memory = memory
        self.gnn = gnn
        self.node_pred = node_pred
        self.config = config
        self.device = device

        self.neighbor_loader = self._build_neighbor_loader()
        self.assoc = torch.empty(self.data.num_nodes, dtype=torch.long, device=device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            set(self.memory.parameters())
            | set(self.gnn.parameters())
            | set(self.node_pred.parameters()),
            lr=config.lr,
        )

    def _build_neighbor_loader(self):
        from modules.neighbor_loader import LastNeighborLoader

        return LastNeighborLoader(
            self.data.num_nodes,
            size=self.config.num_neighbors,
            device=self.device,
        )

    def process_edges(self, src, dst, t, msg) -> None:
        if src.nelement() > 0:
            self.memory.update_state(src, dst, t, msg)
            self.neighbor_loader.insert(src, dst)

    def _predict_nodes(self, label_srcs: torch.Tensor):
        n_id = label_srcs
        n_id_neighbors, mem_edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id_neighbors] = torch.arange(
            n_id_neighbors.size(0), device=self.device
        )

        z, last_update = self.memory(n_id_neighbors)
        z = self.gnn(
            z,
            last_update,
            mem_edge_index,
            self.data.t[e_id].to(self.device),
            self.data.msg[e_id].to(self.device),
        )
        z = z[self.assoc[n_id]]
        return self.node_pred(z)

    def train_epoch(self) -> Dict[str, float]:
        self.memory.train()
        self.gnn.train()
        self.node_pred.train()
        self.memory.reset_state()
        self.neighbor_loader.reset_state()

        total_loss = 0.0
        total_score = 0.0
        num_label_ts = 0
        label_t = self.dataset.get_label_time()

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="train", leave=False)):
            if self.config.max_batches is not None and batch_idx >= self.config.max_batches:
                break
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            query_t = batch.t[-1]

            if query_t > label_t:
                label_tuple = self.dataset.get_node_label(query_t)
                label_ts, label_srcs, labels = label_tuple
                label_t = self.dataset.get_label_time()
                label_srcs = label_srcs.to(self.device)

                previous_day_mask = batch.t < label_t
                self.process_edges(
                    src[previous_day_mask],
                    dst[previous_day_mask],
                    t[previous_day_mask],
                    msg[previous_day_mask],
                )
                src, dst, t, msg = (
                    src[~previous_day_mask],
                    dst[~previous_day_mask],
                    t[~previous_day_mask],
                    msg[~previous_day_mask],
                )

                pred = self._predict_nodes(label_srcs)
                labels = labels.to(self.device)
                loss = self.criterion(pred, labels)

                result = self.evaluator.eval(
                    {
                        "y_true": labels.cpu().detach().numpy(),
                        "y_pred": pred.cpu().detach().numpy(),
                        "eval_metric": [self.eval_metric],
                    }
                )
                total_score += result[self.eval_metric]
                num_label_ts += 1

                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach())

            self.process_edges(src, dst, t, msg)
            self.memory.detach()

        metrics = {"ce": total_loss / max(num_label_ts, 1)}
        metrics[self.eval_metric] = total_score / max(num_label_ts, 1)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.memory.eval()
        self.gnn.eval()
        self.node_pred.eval()

        total_score = 0.0
        num_label_ts = 0
        label_t = self.dataset.get_label_time()

        for batch in tqdm(loader, desc="eval", leave=False):
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

                previous_day_mask = batch.t < label_t
                self.process_edges(
                    src[previous_day_mask],
                    dst[previous_day_mask],
                    t[previous_day_mask],
                    msg[previous_day_mask],
                )
                src, dst, t, msg = (
                    src[~previous_day_mask],
                    dst[~previous_day_mask],
                    t[~previous_day_mask],
                    msg[~previous_day_mask],
                )

                pred = self._predict_nodes(label_srcs)
                result = self.evaluator.eval(
                    {
                        "y_true": labels.cpu().detach().numpy(),
                        "y_pred": pred.cpu().detach().numpy(),
                        "eval_metric": [self.eval_metric],
                    }
                )
                total_score += result[self.eval_metric]
                num_label_ts += 1

            self.process_edges(src, dst, t, msg)

        return {self.eval_metric: total_score / max(num_label_ts, 1)}

    def fit(self) -> Dict[str, Any]:
        train_curve = []
        val_curve = []
        test_curve = []
        max_val_score = 0.0
        best_test_idx = 0

        for epoch in range(1, self.config.epochs + 1):
            start = timeit.default_timer()
            train_metrics = self.train_epoch()
            train_time = timeit.default_timer() - start

            start = timeit.default_timer()
            val_metrics = self.evaluate(self.val_loader)
            val_time = timeit.default_timer() - start

            if val_metrics[self.eval_metric] > max_val_score:
                max_val_score = val_metrics[self.eval_metric]
                best_test_idx = epoch - 1

            start = timeit.default_timer()
            test_metrics = self.evaluate(self.test_loader)
            test_time = timeit.default_timer() - start

            train_curve.append(train_metrics[self.eval_metric])
            val_curve.append(val_metrics[self.eval_metric])
            test_curve.append(test_metrics[self.eval_metric])

            print("-" * 40)
            print(f"Epoch {epoch:02d}")
            print(f"train: {train_metrics} ({train_time:.1f}s)")
            print(f"val:   {val_metrics} ({val_time:.1f}s)")
            print(f"test:  {test_metrics} ({test_time:.1f}s)")

            self.dataset.reset_label_time()

        best_test_score = test_curve[best_test_idx]
        summary = {
            "best_val_score": max_val_score,
            "best_val_epoch": best_test_idx + 1,
            "best_test_score": best_test_score,
            "train_curve": train_curve,
            "val_curve": val_curve,
            "test_curve": test_curve,
        }
        print("=" * 40)
        print(f"Best val {self.eval_metric}: {max_val_score:.4f} (epoch {best_test_idx + 1})")
        print(f"Test {self.eval_metric} at best val: {best_test_score:.4f}")
        return summary
