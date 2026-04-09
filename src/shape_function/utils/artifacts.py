from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunArtifacts:
    root_dir: Path
    figures_dir: Path
    diagnostics_dir: Path


def ensure_run_artifacts(repo_root: Path, *parts: str) -> RunArtifacts:
    root_dir = repo_root.joinpath("runs", *parts)
    figures_dir = root_dir / "figures"
    diagnostics_dir = figures_dir / "diagnostics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(root_dir=root_dir, figures_dir=figures_dir, diagnostics_dir=diagnostics_dir)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_summary(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)
