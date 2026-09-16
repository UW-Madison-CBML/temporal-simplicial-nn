#!/usr/bin/env python3
"""
SEED-VII emotion classification with per-trial temporal GNN baselines.

Each trial is one temporal sample:
  window graphs (changing edges) → temporal model → emotion label.

Examples:
    python main_seed.py --model tgn --band alpha --split trial --fold 0
    python main_seed.py --dry_run --run_all_models --band alpha
    python main_seed.py --dry_run --run_all_models --run_all_bands --split trial
"""

from __future__ import annotations

import argparse
import sys

import torch

from config import SEED_DATA_ROOT
from data_loaders.seed_vii.bundle import build_seed_bundle, get_seed_fold_splits, list_seed_bands
from seed_models.base import SeedTrainConfig
from seed_models.runners import SEED_MODELS, run_seed_model
from seed_results_logger import SeedRunContext, build_table_iii, save_seed_run_result
from tgb.utils.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEED-VII temporal GNN emotion classification")
    parser.add_argument("--model", type=str, default="tgn", choices=SEED_MODELS)
    parser.add_argument("--band", type=str, default="alpha", choices=list_seed_bands())
    parser.add_argument(
        "--split",
        type=str,
        default="trial",
        choices=["trial", "subject"],
        help="CV split strategy (default: trial-wise over 1600 trials)",
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Fold id, 0..n_folds-1 (-1 = all folds)",
    )
    parser.add_argument(
        "--n_folds", type=int, default=4,
        help="Number of CV folds. 4 for the trial split; 20 with --split subject "
        "gives leave-one-subject-out (one held-out subject per fold).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--memory_dim", type=int, default=100)
    parser.add_argument("--time_dim", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=-1, help="CUDA device id (-1 for CPU)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_all_models", action="store_true")
    parser.add_argument("--run_all_bands", action="store_true")
    parser.add_argument("--build_table", action="store_true", help="Build Table III CSV after runs")
    return parser.parse_args()


def _device_from_args(args) -> torch.device:
    if args.gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def _train_one(
    model_name: str,
    band: str,
    fold_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    folds = get_seed_fold_splits(
        band=band,
        split_strategy=args.split,
        data_root=SEED_DATA_ROOT,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    fold = folds[fold_id]

    bundle = build_seed_bundle(
        fold=fold,
        band=band,
        split_strategy=args.split,
        batch_size=args.batch_size,
        device=device,
    )

    print("=" * 60)
    print(f"Model:   {model_name}")
    print(f"Band:    {band}")
    print(f"Split:   {args.split}")
    print(f"Fold:    {fold_id}")
    print(
        f"Train:   {len(bundle.train_trials)} trials "
        f"({len(fold.train_records)} windows)"
    )
    print(
        f"Val:     {len(bundle.val_trials)} trials "
        f"({len(fold.val_records)} windows)"
    )
    print(
        f"Test:    {len(bundle.test_trials)} trials "
        f"({len(fold.test_records)} windows)"
    )
    print(f"Msg dim: {bundle.msg_dim}  (edge + src/dst node feats)")
    print(f"Device:  {device}")
    print("Protocol: per-trial temporal sequences (memory reset each trial)")

    config = SeedTrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        embedding_dim=args.embedding_dim,
        num_neighbors=args.num_neighbors,
        seed=args.seed,
        max_batches=args.max_batches,  # dry-run: max trials
    )

    summary = run_seed_model(model_name, bundle, config, device)

    ctx = SeedRunContext(
        model_name=model_name,
        band=band,
        split_strategy=args.split,
        fold=fold_id,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        device=str(device),
        dry_run=args.dry_run,
        max_batches=args.max_batches,
        data_root=SEED_DATA_ROOT,
    )
    path = save_seed_run_result(ctx, summary)
    print(f"Results saved to {path}")
    print(
        f"Final train acc={summary['final_train_accuracy']:.4f} "
        f"test acc={summary['final_test_accuracy']:.4f}"
    )
    return summary


def main() -> None:
    args = parse_args()

    if args.dry_run:
        args.epochs = 1
        # Cap number of trials per split for a quick smoke test.
        args.max_batches = 8
    else:
        args.max_batches = None

    device = _device_from_args(args)
    bands = list_seed_bands() if args.run_all_bands else [args.band]
    models = SEED_MODELS if args.run_all_models else [args.model]
    folds = list(range(args.n_folds)) if args.fold < 0 else [args.fold]

    failures = []
    for band in bands:
        for model_name in models:
            for fold_id in folds:
                try:
                    _train_one(model_name, band, fold_id, args, device)
                except Exception as exc:
                    failures.append((band, model_name, fold_id, exc))
                    print(f"FAILED {band}/{model_name}/fold{fold_id}: {exc}", file=sys.stderr)

    if failures:
        print("\nFailures:")
        for band, model, fold_id, exc in failures:
            print(f"  {band} {model} fold{fold_id}: {exc}")
        sys.exit(1)

    if args.build_table or args.run_all_bands:
        table_path = build_table_iii(split_strategy=args.split)
        print(f"Table III saved to {table_path}")

    print("\nAll SEED-VII runs completed successfully.")


if __name__ == "__main__":
    main()
