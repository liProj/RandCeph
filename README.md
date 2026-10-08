# RandCeph

**RandCeph: Cross-Dataset Cephalometric Landmark Detection with Coordinate-Consistent Data Augmentation**

Research code for detecting the 19 shared cephalometric landmarks in Aariz and ISBI 2015. RandCeph combines a ResNet-34 U-Net detector and soft-argmax coordinate decoding with geometric and intensity augmentation. One affine mapping links image sampling, transformed landmark supervision, and recovery of predictions in original-image coordinates.

## What is included

- Data loaders and image-cache preparation.
- Training, deterministic evaluation, same-backbone controls, and architectural/initialization ablations.
- Cross-dataset, joint-training, and leave-one-device-out experiments.
- MRE/SDR aggregation, device-scale audits, target-offset calibration, and angular analyses.
- Figure-generation code and optional adapters for the published benchmark.

This release contains source code. Datasets, trained weights, per-image predictions, and the original experiment's result directories are not bundled. The reported paper results are not a new execution of this release. Earlier manuscript-generation and literature-screening utilities are excluded.

## Setup

Use Python 3.10 or newer and a CUDA-enabled PyTorch/torchvision installation compatible with your GPU. Install the matching PyTorch and torchvision builds using the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The original training implementation stores complete image caches on the GPU and uses CUDA directly. Training is GPU-only. Default mixed precision uses bfloat16; use `--amp 0` if your GPU does not support it. Dependency versions in this release describe required packages, not a recovered lockfile from the original training machine.

## Prepare the datasets

See [data layout and annotation conventions](docs/DATA.md). After placing both datasets in `data/`:

```bash
python code/prep.py
python code/audit.py
```

Paths default to this repository. Set `RANDCEPH_ROOT` to keep datasets and experiment outputs elsewhere. `RANDCEPH_AARIZ` and `RANDCEPH_ISBI` can override the two dataset locations individually.

## Train RandCeph

Run both source datasets with three seeds:

```bash
python scripts/run_experiments.py --suite main
```

Preview commands without training:

```bash
python scripts/run_experiments.py --suite main --dry-run
```

Equivalent single-run command:

```bash
python code/train_lm.py --tag s1_letter --train aariz --view letter --seed 0 --aug_scale 0.6 --aug_rot 12 --aug_shift 0.10 --aug_photo 1
```

The legacy flag `--view letter` selects aspect-preserving letterboxing. The augmentation flags control scale, rotation, translation, and intensity changes. The tag `s1_letter` is retained so that saved files remain compatible with the analysis code.

Each run writes `model.pt`, `metrics.json`, and `preds.npz` under `results/runs/<tag>/<source>_s<seed>/`. A completed run is skipped when its `metrics.json` already exists. Use a new tag for a new configuration.

## Controls and extended experiments

```bash
python scripts/run_experiments.py --suite controls
python scripts/run_experiments.py --suite ablations
python scripts/run_experiments.py --suite joint
python scripts/run_experiments.py --suite lodo
```

See [experiment protocols](docs/EXPERIMENTS.md) for the settings and supervision used in each experiment. The launcher defaults to seeds 0, 1, and 2. Individual configurations can also be run through `code/train_lm.py`.

## Analyse saved predictions

After preparing the data and finishing the relevant experiment suites:

```bash
python code/evaluate.py
python code/analysis.py
python code/figures.py
```

Evaluation writes summary CSVs to `results/`; secondary analyses write to `results/analysis/`; figures write to `paper/figures/`. The full secondary analysis expects the main, control, joint, and leave-one-device-out outputs. Figure generation reports missing inputs. These plotting scripts reproduce the underlying experimental plots; final manuscript typography and figure numbering were edited separately.

## External methods and attribution

HRNet, Cascade, ViT, and ResNet50 identify the configurations from Özcan and Allaf's dual-dataset benchmark. Published comparison constants in `code/common.py` are explicitly marked as literature values. Optional scripts for running the authors' supplementary implementation are described in [references and provenance](docs/REFERENCES.md).

The repository title follows the current RandCeph manuscript. Research findings and protocol details should be read together with that manuscript.
