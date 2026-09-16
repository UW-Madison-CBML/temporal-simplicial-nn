"""Cross-validation splits for SEED-VII."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Set, Tuple

import numpy as np

from data_loaders.seed_vii.constants import DEFAULT_N_FOLDS, DEFAULT_VAL_RATIO, NUM_SUBJECTS
from data_loaders.seed_vii.dataset import TrialKey, WindowRecord, group_records_by_trial, unique_subjects

SplitStrategy = Literal["trial", "subject"]


@dataclass
class FoldSplit:
    fold_id: int
    strategy: SplitStrategy
    train_records: List[WindowRecord]
    val_records: List[WindowRecord]
    test_records: List[WindowRecord]
    train_trials: List[TrialKey]
    val_trials: List[TrialKey]
    test_trials: List[TrialKey]
    train_subjects: List[int]
    val_subjects: List[int]
    test_subjects: List[int]


def _trial_keys(records: Sequence[WindowRecord]) -> List[TrialKey]:
    keys = sorted({r.trial_key for r in records}, key=lambda k: (k.subject, k.trial))
    return keys


def _records_for_trials(
    records: Sequence[WindowRecord],
    trial_keys: Set[TrialKey],
) -> List[WindowRecord]:
    out = [r for r in records if r.trial_key in trial_keys]
    out.sort(key=lambda r: (r.subject, r.trial, r.window_idx))
    return out


def _records_for_subjects(
    records: Sequence[WindowRecord],
    subject_ids: Set[int],
) -> List[WindowRecord]:
    out = [r for r in records if r.subject in subject_ids]
    out.sort(key=lambda r: (r.subject, r.trial, r.window_idx))
    return out


def _split_train_val_trials(
    train_trial_keys: List[TrialKey],
    val_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[TrialKey], List[TrialKey]]:
    keys = list(train_trial_keys)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_ratio)))
    val_keys = keys[:n_val]
    train_keys = keys[n_val:]
    return train_keys, val_keys


def _split_train_val_subjects(
    train_subject_ids: List[int],
    val_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int]]:
    subjects = list(train_subject_ids)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val_subjects = subjects[:n_val]
    train_subjects = subjects[n_val:]
    return train_subjects, val_subjects


def make_fold_splits(
    records: Sequence[WindowRecord],
    strategy: SplitStrategy = "trial",
    n_folds: int = DEFAULT_N_FOLDS,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = 42,
) -> List[FoldSplit]:
    """Build n-fold CV splits.

    trial:  split over 1600 (subject, trial) units (default, SEED-VII Table III style)
    subject: split over 20 subjects (cross-subject generalization)
    """
    if not records:
        raise ValueError("No records provided for splitting")
    rng = np.random.default_rng(seed)
    folds: List[FoldSplit] = []

    if strategy == "trial":
        all_trials = _trial_keys(records)
        perm = np.array(all_trials)
        rng.shuffle(perm)
        fold_sizes = [len(perm) // n_folds] * n_folds
        for i in range(len(perm) % n_folds):
            fold_sizes[i] += 1

        start = 0
        fold_trial_groups: List[List[TrialKey]] = []
        for size in fold_sizes:
            fold_trial_groups.append(list(perm[start : start + size]))
            start += size

        for fold_id in range(n_folds):
            test_trials = set(fold_trial_groups[fold_id])
            trainval_trials = [t for t in all_trials if t not in test_trials]
            train_trial_keys, val_trial_keys = _split_train_val_trials(
                trainval_trials, val_ratio=val_ratio, rng=rng
            )
            train_records = _records_for_trials(records, set(train_trial_keys))
            val_records = _records_for_trials(records, set(val_trial_keys))
            test_records = _records_for_trials(records, test_trials)
            folds.append(
                FoldSplit(
                    fold_id=fold_id,
                    strategy=strategy,
                    train_records=train_records,
                    val_records=val_records,
                    test_records=test_records,
                    train_trials=train_trial_keys,
                    val_trials=val_trial_keys,
                    test_trials=sorted(test_trials, key=lambda k: (k.subject, k.trial)),
                    train_subjects=unique_subjects(train_records),
                    val_subjects=unique_subjects(val_records),
                    test_subjects=unique_subjects(test_records),
                )
            )
        return folds

    if strategy == "subject":
        all_subjects = sorted(unique_subjects(records))
        if len(all_subjects) != NUM_SUBJECTS:
            raise ValueError(f"Expected {NUM_SUBJECTS} subjects, found {len(all_subjects)}")
        perm = np.array(all_subjects)
        rng.shuffle(perm)
        fold_sizes = [len(perm) // n_folds] * n_folds
        for i in range(len(perm) % n_folds):
            fold_sizes[i] += 1

        start = 0
        fold_subject_groups: List[List[int]] = []
        for size in fold_sizes:
            fold_subject_groups.append(list(perm[start : start + size].astype(int)))
            start += size

        for fold_id in range(n_folds):
            test_subjects = set(fold_subject_groups[fold_id])
            trainval_subjects = [s for s in all_subjects if s not in test_subjects]
            train_subject_ids, val_subject_ids = _split_train_val_subjects(
                trainval_subjects, val_ratio=val_ratio, rng=rng
            )
            train_records = _records_for_subjects(records, set(train_subject_ids))
            val_records = _records_for_subjects(records, set(val_subject_ids))
            test_records = _records_for_subjects(records, test_subjects)
            folds.append(
                FoldSplit(
                    fold_id=fold_id,
                    strategy=strategy,
                    train_records=train_records,
                    val_records=val_records,
                    test_records=test_records,
                    train_trials=_trial_keys(train_records),
                    val_trials=_trial_keys(val_records),
                    test_trials=_trial_keys(test_records),
                    train_subjects=train_subject_ids,
                    val_subjects=val_subject_ids,
                    test_subjects=sorted(test_subjects),
                )
            )
        return folds

    raise ValueError(f"Unknown split strategy: {strategy}")
