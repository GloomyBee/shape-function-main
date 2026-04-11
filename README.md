# shape-function

Phase A+B scaffold for a structure-preserving meshfree shape function generator.

## Environment

The original plan targeted Python 3.11. The current machine exposes Python 3.9.12
at `D:\Anaconda\python.exe`, so the local `.venv` is created with Python 3.9.12.
All code in this phase is written to remain compatible with Python 3.9+.

Because of that interpreter constraint, the local environment uses a Python 3.9
compatible scientific stack instead of the original NumPy/SciPy upper range.

## Scope

This repository currently implements:

- a 2D fixed-`k=16` patch sampler
- a NumPy patch-level max-ent teacher solver
- a structure-preserving output head
- a kernel-integral backbone
- an MLP baseline
- a minimal training and evaluation loop

## Implemented vs Planned

Implemented in the current repository:

- 2D only
- fixed `k=16`
- patch-level max-ent teacher solver with `gaussian` and `quartic_spline` priors
- patch families:
  - `uniform`
  - `mildly_perturbed`
  - `highly_random`
  - `clustered`
  - `boundary_truncated`
  - `anisotropic`
  - `sparse_dense_transition`
- structure-preserving head
- `kernel_operator` backbone
- `mlp_baseline` backbone
- minimal train/eval loop and unit tests

Planned next, but not implemented yet:

- variable-`k` handling and masking through the full training pipeline
- 3D extension
- `DeepSets` / `Set Transformer-lite`
- `Transolver-lite`
- OOD beta study and full paper-grade experiment suite
- solver embedding for large-deformation meshfree analysis

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## CLI Quick Start

The repository now exposes a first-party training CLI.

Run via module:

```powershell
.\.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs/data.yaml `
  --train-config configs/train_kernel_operator.yaml
```

Run via console script after `pip install -e .`:

```powershell
shape-function train `
  --data-config configs/data.yaml `
  --train-config configs/train_kernel_operator.yaml
```

Configuration responsibilities:

- `configs/data.yaml`: patch sampling, dataset sizes, `feature_mode`, `k_neighbors`, `patch_types`, `beta_range`
- `configs/train_*.yaml`: backbone selection and training hyperparameters

CLI behavior:

- only the `train` subcommand is available in v1
- optional overrides: `--run-name`, `--device`, `--seed`
- outputs are written to `runs/<run_name>/`
- when `--run-name` is omitted, a readable name is generated from backbone, feature mode, `k_neighbors`, `seed`, and a timestamp

Saved artifacts per run:

- `metrics.json`
- `summary.txt`
- `curves.npz`
- `checkpoint.pt`
- `config_snapshot.yaml`
- `eval_metrics.json`
