"""Per-trial PyG trainer for SEED-VII (TGN / DyRep).

Each trial is an independent temporal sequence:
  reset memory → stream window graphs in order → predict emotion.
"""

from __future__ import annotations

import timeit
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loaders.seed_vii.bundle import SeedVIIBundle
from data_loaders.seed_vii.constants import NUM_EEG_NODES
from data_loaders.seed_vii.trials import TrialSequence, window_edge_tensors
from seed_models.base import SeedTrainConfig


class SeedTrialTrainer:
    def __init__(
        self,
        bundle: SeedVIIBundle,
        memory: nn.Module,
        gnn: nn.Module,
        node_pred: nn.Module,
        config: SeedTrainConfig,
        device: torch.device,
    ) -> None:
        self.bundle = bundle
        self.config = config
        self.device = device
        self.memory = memory
        self.gnn = gnn
        self.node_pred = node_pred
        self.evaluator = bundle.evaluator

        from modules.neighbor_loader import LastNeighborLoader

        self.neighbor_loader = LastNeighborLoader(
            NUM_EEG_NODES, size=config.num_neighbors, device=device
        )
        self.assoc = torch.empty(NUM_EEG_NODES, dtype=torch.long, device=device)
        self.all_nodes = torch.arange(NUM_EEG_NODES, dtype=torch.long, device=device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            set(self.memory.parameters())
            | set(self.gnn.parameters())
            | set(self.node_pred.parameters()),
            lr=config.lr,
        )
        # Store last-seen messages for GNN edge attributes within a trial.
        self._trial_t: List[torch.Tensor] = []
        self._trial_msg: List[torch.Tensor] = []

    def _reset_trial_state(self) -> None:
        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        self._trial_t = []
        self._trial_msg = []

    def _process_edges(self, src, dst, t, msg) -> None:
        if src.nelement() == 0:
            return
        self.memory.update_state(src, dst, t, msg)
        self.neighbor_loader.insert(src, dst)
        self._trial_t.append(t.detach())
        self._trial_msg.append(msg.detach())

    def _edge_lookup(self, e_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._trial_t:
            raise RuntimeError("No edges stored for this trial yet")
        t_cat = torch.cat(self._trial_t, dim=0)
        msg_cat = torch.cat(self._trial_msg, dim=0)
        return t_cat[e_id], msg_cat[e_id]

    def _embed_nodes(self) -> torch.Tensor:
        n_id = self.all_nodes
        n_id_neighbors, mem_edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id_neighbors] = torch.arange(
            n_id_neighbors.size(0), device=self.device
        )
        z, last_update = self.memory(n_id_neighbors)
        edge_t, edge_msg = self._edge_lookup(e_id)
        z = self.gnn(z, last_update, mem_edge_index, edge_t, edge_msg)
        return z[self.assoc[n_id]]

    def _run_trials(
        self,
        trials: List[TrialSequence],
        train: bool,
    ) -> Dict[str, float]:
        if train:
            self.memory.train()
            self.gnn.train()
            self.node_pred.train()
        else:
            self.memory.eval()
            self.gnn.eval()
            self.node_pred.eval()

        trial_preds: List[np.ndarray] = []
        trial_labels: List[int] = []
        window_preds: List[np.ndarray] = []
        window_labels: List[int] = []
        total_loss = 0.0
        n_loss = 0

        max_trials = self.config.max_batches
        iterable = trials if max_trials is None else trials[:max_trials]
        if train:
            # See dyglib_trainer: trials are stored sorted, so a fixed order makes
            # the gradient stream identical and emotion-blocked every epoch.
            iterable = [iterable[i] for i in torch.randperm(len(iterable)).tolist()]
        desc = "train" if train else "eval"

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for trial in tqdm(iterable, desc=desc, leave=False):
                self._reset_trial_state()
                emotion = torch.tensor([trial.emotion_idx], dtype=torch.long, device=self.device)
                window_logits: List[torch.Tensor] = []
                n_windows = int(sum(1 for r in trial.windows if r.edges))
                if train:
                    self.optimizer.zero_grad()

                for t, rec in enumerate(trial.windows):
                    if not rec.edges:
                        continue
                    src, dst, t_vec, msg = window_edge_tensors(rec, t, device=self.device)
                    self._process_edges(src, dst, t_vec, msg)

                    node_emb = self._embed_nodes()
                    graph_emb = node_emb.mean(dim=0, keepdim=True)
                    logits = self.node_pred(graph_emb)
                    window_logits.append(logits)

                    window_preds.append(logits.detach().cpu().numpy()[0])
                    window_labels.append(trial.emotion_idx)

                    if train:
                        # One step per TRIAL, not per window: all of a trial's windows
                        # share a label, so per-window steps were correlated
                        # single-sample updates. Scaling by the window count makes the
                        # accumulated gradient the trial mean.
                        loss = self.criterion(logits, emotion) / max(n_windows, 1)
                        loss.backward()
                        total_loss += float(loss.detach())
                        self.memory.detach()
                        # Detach stored msgs/times so graph does not grow across windows.
                        self._trial_t = [x.detach() for x in self._trial_t]
                        self._trial_msg = [x.detach() for x in self._trial_msg]

                if train and n_windows:
                    self.optimizer.step()
                    n_loss += 1

                if not window_logits:
                    continue
                trial_logit = torch.stack(window_logits, dim=0).mean(dim=0)
                trial_preds.append(trial_logit.detach().cpu().numpy()[0])
                trial_labels.append(trial.emotion_idx)

        metrics = self._metrics(trial_preds, trial_labels, window_preds, window_labels)
        if train and n_loss:
            metrics["loss"] = total_loss / n_loss
        return metrics

    def _metrics(
        self,
        trial_preds: List[np.ndarray],
        trial_labels: List[int],
        window_preds: List[np.ndarray],
        window_labels: List[int],
    ) -> Dict[str, float]:
        if not trial_preds:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "window_accuracy": 0.0, "window_f1": 0.0}
        trial_m = self.evaluator.eval({
            "y_true": np.asarray(trial_labels),
            "y_pred": np.stack(trial_preds, axis=0),
            "eval_metric": ["accuracy", "precision", "recall", "f1"],
        })
        window_m = self.evaluator.eval({
            "y_true": np.asarray(window_labels),
            "y_pred": np.stack(window_preds, axis=0),
            "eval_metric": ["accuracy", "precision", "recall", "f1"],
        })
        return {
            "accuracy": trial_m["accuracy"],
            "precision": trial_m["precision"],
            "recall": trial_m["recall"],
            "f1": trial_m["f1"],
            "window_accuracy": window_m["accuracy"],
            "window_f1": window_m["f1"],
        }

    def fit(self) -> Dict[str, Any]:
        train_curve: List[Dict[str, float]] = []
        val_curve: List[Dict[str, float]] = []
        test_curve: List[Dict[str, float]] = []
        best_val = -1.0
        best_idx = 0

        for epoch in range(1, self.config.epochs + 1):
            t0 = timeit.default_timer()
            train_m = self._run_trials(self.bundle.train_trials, train=True)
            train_t = timeit.default_timer() - t0

            t0 = timeit.default_timer()
            val_m = self._run_trials(self.bundle.val_trials, train=False)
            val_t = timeit.default_timer() - t0

            t0 = timeit.default_timer()
            test_m = self._run_trials(self.bundle.test_trials, train=False)
            test_t = timeit.default_timer() - t0

            train_curve.append(train_m)
            val_curve.append(val_m)
            test_curve.append(test_m)
            if val_m["accuracy"] > best_val:
                best_val = val_m["accuracy"]
                best_idx = epoch - 1

            print("-" * 40)
            print(f"Epoch {epoch:02d}")
            print(
                f"train: trial_acc={train_m['accuracy']:.4f} "
                f"win_acc={train_m['window_accuracy']:.4f} ({train_t:.1f}s)"
            )
            print(
                f"val:   trial_acc={val_m['accuracy']:.4f} "
                f"win_acc={val_m['window_accuracy']:.4f} ({val_t:.1f}s)"
            )
            print(
                f"test:  trial_acc={test_m['accuracy']:.4f} "
                f"win_acc={test_m['window_accuracy']:.4f} ({test_t:.1f}s)"
            )
            print(f"train_loss: {train_m.get('loss', float('nan')):.4f}")
            print(f"epoch_total: {train_t + val_t + test_t:.1f}s")

        best_test = test_curve[best_idx]
        return {
            "best_val_accuracy": best_val,
            "best_val_epoch": best_idx + 1,
            "best_test_accuracy": best_test["accuracy"],
            "best_test_precision": best_test["precision"],
            "best_test_recall": best_test["recall"],
            "best_test_f1": best_test["f1"],
            "train_curve": train_curve,
            "val_curve": val_curve,
            "test_curve": test_curve,
            "final_train_accuracy": train_curve[-1]["accuracy"],
            "final_train_f1": train_curve[-1]["f1"],
            "final_test_accuracy": test_curve[-1]["accuracy"],
            "final_test_precision": test_curve[-1]["precision"],
            "final_test_recall": test_curve[-1]["recall"],
            "final_test_f1": test_curve[-1]["f1"],
        }
