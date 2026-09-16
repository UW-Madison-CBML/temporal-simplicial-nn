"""Shared training config for SEED-VII baselines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeedTrainConfig:
    lr: float = 1e-4
    epochs: int = 50
    memory_dim: int = 100
    time_dim: int = 100
    embedding_dim: int = 100
    num_neighbors: int = 10
    seed: int = 42
    # For SEED per-trial trainers this caps the number of trials (not edge batches).
    max_batches: int | None = None
