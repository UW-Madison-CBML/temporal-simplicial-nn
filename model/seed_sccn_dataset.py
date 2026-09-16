"""SEED-VII trial sequences for SCCNTemporalClassifier (portable bundle).

Extracted clique/neighbor/VR pickles already store edges, triangles, features, and
weights. This dataset converts each window to SCCN tensors **once** (CPU cache,
optionally on disk) and reuses those tensors across all epochs — it does not
recompute simplices or features every epoch.
"""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
_MODEL_ROOT = _ROOT / "models" / "temporal-simplicial-net"
for p in (str(_SCRIPTS), str(_MODEL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from seed_records import load_records  # noqa: E402
from seed_to_sccn_window import records_to_sccn_windows  # noqa: E402

EMOTION_ORDER = (
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
)
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTION_ORDER)}

DEFAULT_WEIGHTED = _ROOT / "data" / "all_band"
DEFAULT_CACHE = _ROOT / "data" / "all_band" / "sccn_tensor_cache"


@dataclass(frozen=True)
class TrialIndex:
    subject: int
    trial: int
    emotion: str
    emotion_id: int
    n_windows: int


def resolve_method_dir(data_root: Path, method: str) -> Path:
    d = Path(data_root) / method
    if not d.is_dir():
        raise FileNotFoundError(f"Method directory not found: {d}")
    return d


def list_available_subjects(method_dir: Path) -> list[int]:
    ids = []
    for p in sorted(Path(method_dir).glob("subject_*_windows.pkl")):
        stem = p.name.replace("subject_", "").replace("_windows.pkl", "")
        ids.append(int(stem))
    return ids


def group_records_by_trial(records) -> dict[tuple[int, int], list]:
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for r in records:
        groups[(r.subject, r.trial)].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r.window_idx)
    return groups


# Bump when the cached window-dict structure changes, so stale caches from an older
# layout are never silently reused. v2 added `cell_nodes` / `t_frac` per window;
# v3 added `local_ids` / `n_cells` for the dense temporal state; v4 added `eye`.
_CACHE_VERSION = 4


def _subject_cache_path(
    cache_dir: Path,
    method: str,
    subject: int,
    *,
    max_rank: int,
    feat_dim: int,
    max_windows_per_trial: int | None,
) -> Path:
    mw = "all" if max_windows_per_trial is None else str(max_windows_per_trial)
    return (
        cache_dir
        / method
        / f"subject_{subject:02d}_v{_CACHE_VERSION}"
        f"_rank{max_rank}_feat{feat_dim}_mw{mw}.pt"
    )


def node_feature_dim(records) -> int | None:
    """Width of the per-node feature vectors actually stored in ``records``."""
    for r in records:
        if r.node_features:
            row = r.node_features[0]
            try:
                return len(row)
            except TypeError:  # a bare scalar
                return 1
    return None


def _check_feat_dim(records, feat_dim: int, method_dir: Path) -> None:
    """Fail loudly when the requested ``feat_dim`` does not match the data.

    Single-band extracts store ``[62 x 1]`` node features while the all-band extract
    stores ``[62 x 5]``. Without this check a mismatch is silent: ``_as_tuple_feat``
    repeats a length-1 vector up to ``feat_dim``, so a band run with the default
    ``feat_dim=5`` would train on five identical copies of one number and still look
    perfectly healthy.
    """
    actual = node_feature_dim(records)
    if actual is None or actual == feat_dim:
        return
    raise ValueError(
        f"--feat-dim {feat_dim} does not match the data in {method_dir}, which stores "
        f"{actual} value(s) per node. Single-band extracts need --feat-dim 1; the "
        f"all-band extract needs --feat-dim 5. Continuing would silently "
        f"{'repeat' if actual == 1 else 'truncate'} the features."
    )


def _build_subject_trials(
    method_dir: Path,
    subject: int,
    *,
    max_rank: int,
    feat_dim: int,
    max_trials_per_subject: int | None,
    max_windows_per_trial: int | None,
    dtype,
) -> tuple[list[TrialIndex], list[list[dict]], list[torch.Tensor]]:
    """Convert extracted pickle records → SCCN window tensors (CPU, once)."""
    recs = load_records(method_dir / f"subject_{subject:02d}_windows.pkl")
    _check_feat_dim(recs, feat_dim, method_dir)
    groups = group_records_by_trial(recs)
    trial_keys = sorted(groups.keys(), key=lambda k: k[1])
    if max_trials_per_subject is not None:
        trial_keys = trial_keys[:max_trials_per_subject]

    trials: list[TrialIndex] = []
    windows_list: list[list[dict]] = []
    labels: list[torch.Tensor] = []
    for _subj, trial in trial_keys:
        wins = groups[(_subj, trial)]
        if max_windows_per_trial is not None:
            wins = wins[:max_windows_per_trial]
        emo = wins[0].emotion
        windows = records_to_sccn_windows(
            wins,
            max_rank=max_rank,
            feat_dim=feat_dim,
            device=None,  # CPU tensors
            dtype=dtype,
        )
        trials.append(
            TrialIndex(
                subject=_subj,
                trial=trial,
                emotion=emo,
                emotion_id=EMOTION_TO_ID[emo],
                n_windows=len(wins),
            )
        )
        windows_list.append(windows)
        labels.append(torch.tensor([EMOTION_TO_ID[emo]], dtype=torch.long))
    return trials, windows_list, labels


class SeedTrialSCCNDataset(Dataset):
    def __init__(
        self,
        data_root: Path | None = None,
        *,
        method: str = "clique",
        subjects: list[int] | None = None,
        max_rank: int = 2,
        feat_dim: int = 5,
        max_trials_per_subject: int | None = None,
        max_windows_per_trial: int | None = None,
        device=None,
        dtype=torch.float32,
        prefer_weighted: bool = True,
        cache_dir: Path | None = None,
        rebuild_cache: bool = False,
    ) -> None:
        del prefer_weighted
        if data_root is None:
            data_root = DEFAULT_WEIGHTED
        self.data_root = Path(data_root)
        self.method = method
        self.method_dir = resolve_method_dir(self.data_root, method)
        self.max_rank = max_rank
        self.feat_dim = feat_dim
        self.device = device  # kept for API compat; tensors stay on CPU
        self.dtype = dtype
        self.max_windows_per_trial = max_windows_per_trial
        # Cache inside the extract itself. Different bands share method/feat_dim
        # (alpha and beta are both clique/feat1), so a single shared cache directory
        # would have one band silently reusing another's tensors.
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else self.data_root / "sccn_tensor_cache"
        )

        available = list_available_subjects(self.method_dir)
        if subjects is None:
            subjects = available
        else:
            missing = sorted(set(subjects) - set(available))
            if missing:
                raise FileNotFoundError(
                    f"No pickles for subjects {missing} under {self.method_dir}"
                )
        self.subjects = list(subjects)

        self.trials: list[TrialIndex] = []
        self._windows: list[list[dict]] = []
        self._labels: list[torch.Tensor] = []

        t0 = time.time()
        for sid in self.subjects:
            cache_path = _subject_cache_path(
                self.cache_dir,
                method,
                sid,
                max_rank=max_rank,
                feat_dim=feat_dim,
                max_windows_per_trial=max_windows_per_trial,
            )
            if cache_path.is_file() and not rebuild_cache:
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
                trials = [TrialIndex(**row) for row in blob["trials"]]
                windows_list = blob["windows"]
                labels = blob["labels"]
                src = f"cache {cache_path.name}"
            else:
                trials, windows_list, labels = _build_subject_trials(
                    self.method_dir,
                    sid,
                    max_rank=max_rank,
                    feat_dim=feat_dim,
                    max_trials_per_subject=max_trials_per_subject,
                    max_windows_per_trial=max_windows_per_trial,
                    dtype=dtype,
                )
                if max_trials_per_subject is None:
                    # only persist full-subject caches
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "trials": [t.__dict__ for t in trials],
                            "windows": windows_list,
                            "labels": labels,
                        },
                        cache_path,
                    )
                    src = f"built+saved {cache_path.name}"
                else:
                    src = "built (no disk cache; max_trials set)"

            if max_trials_per_subject is not None:
                trials = trials[:max_trials_per_subject]
                windows_list = windows_list[:max_trials_per_subject]
                labels = labels[:max_trials_per_subject]

            print(
                f"[dataset] subject {sid:02d}: {len(trials)} trials from {src}",
                flush=True,
            )
            self.trials.extend(trials)
            self._windows.extend(windows_list)
            self._labels.extend(labels)

        print(
            f"[dataset] ready: {len(self.trials)} trials "
            f"({len(self.subjects)} subjects) in {time.time() - t0:.1f}s "
            f"(tensors cached; no per-epoch rebuild)",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int):
        # Return precomputed tensors as-is (CPU). Trainer moves them to GPU.
        return self._windows[idx], self._labels[idx], self.trials[idx]


def loso_splits(subjects: list[int], test_subject: int) -> tuple[list[int], list[int]]:
    if test_subject not in subjects:
        raise ValueError(f"test_subject {test_subject} not in {subjects}")
    return [s for s in subjects if s != test_subject], [test_subject]


def parse_subjects(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _split_counts(n: int, train_frac: float, val_frac: float) -> tuple[int, int]:
    """Return ``(n_train, n_val)`` out of ``n`` items; the rest are test.

    Rounding is deliberately floor-then-clamp rather than ``round``: with ``round`` a
    small group can hand every item to train/val and leave the test split empty
    (n=3 -> 2/1/0, n=4 -> 3/1/0, n=8 -> 6/2/0), which silently produces ``nan``
    metrics. Here train and val each yield a slot so that every split gets at least
    one item whenever ``n`` allows it:

    - ``n <= 1``  -> everything to train (nothing can be held out)
    - ``n == 2``  -> one train, one test (still too few for a validation split)
    - ``n >= 3``  -> at least one item in each of train / val / test

    ``val_frac == 0`` is honoured exactly (no validation split is forced).
    """
    if n <= 0:
        return 0, 0
    if n == 1:
        return 1, 0
    want_val = val_frac > 0
    if n == 2:
        return 1, 0
    # reserve one slot for test, plus one for val when a validation split is wanted
    reserved = 2 if want_val else 1
    n_train = max(1, min(int(n * train_frac), n - reserved))
    n_val = max(1, min(int(n * val_frac), n - n_train - 1)) if want_val else 0
    return n_train, n_val


def auto_subject_split(
    subjects: list[int],
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split subject ids into disjoint (train, val, test) buckets by count.

    Cross-subject generalization split: every trial of a given subject lands in exactly
    one bucket. Shuffled deterministically by ``seed`` before slicing.
    """
    shuffled = sorted(subjects)
    random.Random(seed).shuffle(shuffled)
    n_train, n_val = _split_counts(len(shuffled), train_frac, val_frac)
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val :])
    return train, val, test


def _carve_val(rest: list, val_frac: float, rng: random.Random) -> tuple[list, list]:
    """Split a fold's non-test remainder into (train, val)."""
    pool = list(rest)
    rng.shuffle(pool)
    n_val = max(1, round(len(pool) * val_frac)) if val_frac > 0 else 0
    n_val = min(n_val, max(len(pool) - 1, 0))
    return sorted(pool[n_val:]), sorted(pool[:n_val])


def subject_cv_folds(
    subjects: list[int],
    n_folds: int = 4,
    val_frac: float = 0.2,
    seed: int = 42,
) -> list[tuple[list[int], list[int], list[int]]]:
    """K-fold cross-validation over *subjects* (cross-subject generalization).

    Every subject serves as test exactly once across the folds. The validation set is
    carved out of each fold's remaining subjects, so train / val / test never share a
    subject within a fold.

    Returns one ``(train, val, test)`` subject-id triple per fold.
    """
    shuffled = sorted(subjects)
    random.Random(seed).shuffle(shuffled)
    # Round-robin dealing keeps fold sizes within one of each other for any n_folds.
    bins: list[list[int]] = [[] for _ in range(n_folds)]
    for i, s in enumerate(shuffled):
        bins[i % n_folds].append(s)

    folds = []
    for k in range(n_folds):
        test = sorted(bins[k])
        rest = [s for j, b in enumerate(bins) if j != k for s in b]
        train, val = _carve_val(rest, val_frac, random.Random(seed + 1000 + k))
        folds.append((train, val, test))
    return folds


def trial_cv_folds(
    trials: list[TrialIndex],
    n_folds: int = 4,
    val_frac: float = 0.2,
    seed: int = 42,
) -> list[tuple[list[int], list[int], list[int]]]:
    """K-fold cross-validation over *trials*, stratified by emotion.

    Ignores subject identity: trials from one subject may land in different folds.
    Dealing each emotion's trials round-robin across folds keeps every fold's class
    balance close to the overall distribution.

    Returns one ``(train_idx, val_idx, test_idx)`` triple of dataset indices per fold.
    """
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(trials):
        by_label[t.emotion_id].append(i)

    rng = random.Random(seed)
    bins: list[list[int]] = [[] for _ in range(n_folds)]
    for label in sorted(by_label):
        idxs = by_label[label][:]
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            bins[i % n_folds].append(idx)

    folds = []
    for k in range(n_folds):
        test = sorted(bins[k])
        rest = [i for j, b in enumerate(bins) if j != k for i in b]
        train, val = _carve_val(rest, val_frac, random.Random(seed + 1000 + k))
        folds.append((train, val, test))
    return folds


def trial_split_indices(
    trials: list[TrialIndex],
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split trial indices into (train, val, test), stratified by emotion label.

    Unlike :func:`auto_subject_split`, this ignores subject identity entirely -- trials
    from the same subject can land in different splits. Stratifying by emotion keeps the
    class distribution balanced across splits instead of a flat shuffle, which can starve
    val/test of rare classes.
    """
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(trials):
        by_label[t.emotion_id].append(i)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(by_label):
        idxs = by_label[label][:]
        rng.shuffle(idxs)
        n_train, n_val = _split_counts(len(idxs), train_frac, val_frac)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)
