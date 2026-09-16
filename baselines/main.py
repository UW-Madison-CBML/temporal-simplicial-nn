#!/usr/bin/env python3
"""
Train temporal GNN baselines for TGB node property prediction (node classification).

Examples:
    python main.py --model tgn --dataset tgbn-trade --epochs 50 --gpu 0
    python main.py --model dygformer --dataset tgbn-genre --epochs 50 --gpu 0
    python main.py --dry_run                    # 1 epoch, 3 batches, all models
    python main.py --dry_run --model tgat       # dry run single model
"""

from __future__ import annotations

import argparse
import sys

import torch

from config import DATA_ROOT, EPOCH_RESULTS_CSV, RUN_SUMMARY_CSV
from datasets import (
    NODE_CLASSIFICATION_DATASETS,
    get_node_classification_dataset,
    list_node_classification_datasets,
)
from results_logger import RunContext, save_results
from TGNN_models import NODE_CLASSIFICATION_MODELS, run_model
from TGNN_models.base import TrainConfig
from tgb.utils.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal GNN node classification on TGB datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tgn",
        choices=NODE_CLASSIFICATION_MODELS,
        help="Temporal GNN model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tgbn-genre",
        choices=NODE_CLASSIFICATION_DATASETS,
        help="TGB node property prediction dataset (tgbn-trade is smallest)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--memory_dim", type=int, default=100)
    parser.add_argument("--time_dim", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id (-1 for CPU)")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Quick smoke test: 1 epoch, 3 batches, small batch size",
    )
    parser.add_argument(
        "--run_all_models",
        action="store_true",
        help="Run every model sequentially (use with --dry_run for smoke test)",
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="Print available node classification datasets and exit",
    )
    return parser.parse_args()


def _device_from_args(args) -> torch.device:
    if args.gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def _train_one(model_name: str, args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    print("=" * 60)
    print(f"Model:      {model_name}")
    print(f"Dataset:    {args.dataset}")
    print(f"Data root:  {DATA_ROOT}")
    print(f"Results:    {EPOCH_RESULTS_CSV}")
    print(f"Device:     {device}")

    bundle = get_node_classification_dataset(
        name=args.dataset,
        data_root=DATA_ROOT,
        batch_size=args.batch_size,
        device=device,
    )

    config = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        embedding_dim=args.embedding_dim,
        num_neighbors=args.num_neighbors,
        seed=args.seed,
        max_batches=args.max_batches,
    )

    summary = run_model(model_name, bundle, config, device)

    ctx = RunContext(
        model_name=model_name,
        dataset_name=args.dataset,
        eval_metric=bundle.eval_metric,
        num_classes=bundle.num_classes,
        total_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        num_neighbors=args.num_neighbors,
        seed=args.seed,
        device=str(device),
        dry_run=args.dry_run,
        max_batches=args.max_batches,
        data_root=DATA_ROOT,
    )
    paths = save_results(ctx, summary)
    print(f"Results saved:")
    print(f"  epoch log:   {paths['epoch_results_csv']}")
    print(f"  run summary: {paths['run_summary_csv']}")
    return summary


def main() -> None:
    args = parse_args()

    if args.list_datasets:
        for name, desc in list_node_classification_datasets().items():
            print(f"  {name}: {desc}")
        return

    if args.dry_run:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 50)
        args.max_batches = 3
    else:
        args.max_batches = None

    device = _device_from_args(args)

    models = NODE_CLASSIFICATION_MODELS if args.run_all_models else [args.model]
    failures = []

    for model_name in models:
        try:
            _train_one(model_name, args, device)
        except Exception as exc:
            failures.append((model_name, exc))
            print(f"FAILED {model_name}: {exc}", file=sys.stderr)

    if failures:
        print("\nDry run failures:")
        for name, exc in failures:
            print(f"  {name}: {exc}")
        sys.exit(1)

    if args.run_all_models:
        print("\nAll models completed successfully.")


if __name__ == "__main__":
    main()
