# T-SNN: Temporal Simplicial Neural Network for EEG Emotion Recognition

Code and reproduction scripts for emotion recognition from EEG modelled as a
**sequence of simplicial complexes**. Each trial is split into overlapping
windows; every window is lifted to a clique complex over the 62 EEG channels
(nodes, edges, triangles), and a recurrent simplicial network passes messages
within a window while carrying state across windows.

The repository also contains the temporal-GNN baselines used for comparison, so
every number in the paper can be reproduced from one place.

---

## 1. Requirements

- Python 3.9+
- PyTorch 2.0+ (a GPU is strongly recommended; CPU works but is far slower)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The proposed model needs only `torch`, `numpy` and `scikit-learn`. The baselines
additionally need `torch-geometric`, `torch-scatter` and `py-tgb`. `torch-scatter`
has no PyPI wheel and must come from the PyG index matched to your torch build:

```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

## 2. Data

The raw SEED-VII recordings are **not** redistributed here; they require their own
access agreement from the dataset authors. What is hosted is the derived extract
the training code consumes: per-window simplicial complexes with their features,
adjacency and incidence operators.

```bash
./scripts/download_data.sh
```

Set the download location first, either by editing `DATA_URL` in
`scripts/download_data.sh` or by passing it in:

```bash
DATA_URL=https://osf.io/<file-id>/download ./scripts/download_data.sh
```

All extracts ship as one archive (~2.5 GB download and unpacked). If you fetched
it by hand, unpack it with `tar -xzf seedvii_simplicial_extracts.tar.gz -C data/`.

Extracts land in `data/<name>/clique/`:

| Extract | Contents | Size |
|---|---|---|
| `all_band` | 5 band-power values per channel | ~630 MB |
| `delta`, `theta`, `alpha`, `beta`, `gamma` | 1 value per channel | ~250 MB each |
| `all_band_eye` | `all_band` plus per-window eye-movement vectors | ~650 MB |

To regenerate an extract from raw SEED-VII instead of downloading it, see
`model/seed_to_sccn_window.py`.

## 3. Reproducing the results

Each script runs every fold in sequence and writes one JSON per fold. Runtime is
roughly 1.5–2 h per fold on a single modern GPU.

```bash
./scripts/run_band_ablation.sh     # Table I  - per band, trial split, 4-fold CV
./scripts/run_eye_fusion.sh        # Table II - EEG only vs EEG + eye movements
./scripts/run_cross_subject.sh     # Table III - leave-one-subject-out, 20 folds
./scripts/run_baselines.sh         # Table I  - temporal-GNN baselines
```

Then print mean ± standard deviation over folds:

```bash
./scripts/aggregate.sh             # proposed model
./scripts/aggregate_baselines.sh   # baselines
```

Useful overrides (all scripts read them from the environment):

| Variable | Default | Meaning |
|---|---|---|
| `DEVICE` | `auto` | `cuda`, `cpu`, or auto-detect |
| `EPOCHS` | `100` | training epochs |
| `CHANNELS` | `512` | hidden width `d` |
| `LR` | `3e-4` | learning rate |
| `MICRO_BATCH` | `64` | trials per forward pass; **lower this if you run out of memory** |
| `DATA_ROOT` | `./data` | where extracts live |

Subsets can be run directly, which is the easiest way to parallelise across
machines — each fold is independent:

```bash
./scripts/run_band_ablation.sh alpha theta    # selected bands
./scripts/run_cross_subject.sh 3 7            # selected folds
./scripts/run_baselines.sh --models tgn --bands all_band
```

`MICRO_BATCH` caps how many trials share a forward pass. It bounds activation
memory **without** changing the optimizer step: sub-chunk losses are re-weighted
to the batch mean, so results are unchanged.

## 4. Layout

```
model/                     proposed model
  sccn_temporal.py           classifier: encoders, priors, readout, eye fusion
  sccn_temporal_layer.py     temporal SCCN blocks (three spatio-temporal variants)
  topomodelx/                vendored simplicial message-passing layers
  train_sccn_temporal_seed.py  training / cross-validation driver
  seed_sccn_dataset.py       dataset and fold construction
  seed_to_sccn_window.py     raw SEED-VII -> windowed simplicial extract
  merge_eye_into_extract.py  attach eye-movement vectors to an EEG extract
  aggregate_folds.py         mean +/- std over folds
baselines/                 temporal-GNN baselines
  main_seed.py               entry point
  seed_models/               per-backend trainers
  data_loaders/seed_vii/     per-trial sequence construction
  aggregate_seed_folds.py    mean +/- std over folds
scripts/                   portable runners (plain bash, no scheduler assumed)
results/                   metrics from the reported runs
data/                      extracts land here after download
```

## 5. Protocol

- **Trial split** — 4-fold cross-validation over the 1600 (subject, trial) pairs,
  stratified by emotion.
- **Cross-subject split** — leave-one-subject-out, 20 folds, each holding out one
  subject entirely.
- Validation is carved from the training folds only. Eye-movement standardisation
  statistics are computed on training folds alone.
- The reported epoch is the one with the best **validation** macro-F1. Macro
  averaging is used throughout: it exposes collapse onto a single class, which
  accuracy alone hides on seven classes.
- Every table reports mean and standard deviation across folds.

### A note on the baselines

The baseline training loop was modified in one respect: it originally took one
optimizer step per *window*, and since all windows of a trial share a label, this
produced long runs of identical-label single-sample updates and the models did not
learn (training accuracy stayed flat for 100 epochs). The loop now accumulates over
a trial's windows and takes **one step per trial**, with trial order shuffled each
epoch. Everything else — architecture, features, and the graph encoding — is
unchanged from the original implementations.

## 6. Results

`results/` holds the per-fold metrics behind the reported tables, so the numbers
can be checked without rerunning anything:

```bash
./scripts/aggregate.sh
./scripts/aggregate_baselines.sh
```

Trained checkpoints are not included; rerunning a fold regenerates them.
