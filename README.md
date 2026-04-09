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
