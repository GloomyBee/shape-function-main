from __future__ import annotations

from pathlib import Path

import numpy as np

from shape_function.eval.visualize_run import (
    REL_ERROR_CLIP,
    REL_ERROR_FLOOR,
    build_clipped_relative_error,
    resolve_basis_node_index,
    save_basis_field_visualization,
    save_patch_field_visualization,
    save_patch_visualization,
    save_training_curves_visualization,
    select_patch_indices,
)


def test_select_patch_indices_spread_covers_error_range() -> None:
    scores = [0.05, 0.10, 0.20, 0.30, 0.40, 0.80]
    indices = select_patch_indices(scores, num_patches=4, strategy="spread")
    assert len(indices) == 4
    assert indices[0] == 0
    assert indices[-1] == 5
    assert len(set(indices)) == 4


def test_save_patch_visualization_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "patch.png"
    patch = {
        "X": np.asarray([[0.0, 0.0], [0.7, 0.0], [0.0, 0.8]], dtype=np.float64),
        "x_q": np.asarray([0.1, 0.1], dtype=np.float64),
        "phi_ref": np.asarray([0.55, 0.30, 0.15], dtype=np.float64),
        "beta": 1.25,
        "r_max": 0.9,
        "patch_type": "uniform",
    }
    prediction = {
        "phi_pred": np.asarray([0.50, 0.34, 0.16], dtype=np.float64),
        "phi_base": np.asarray([0.48, 0.33, 0.19], dtype=np.float64),
        "relative_l2": 0.08,
        "global_linf": 0.05,
        "neg_fraction": 0.0,
        "max_negative_magnitude": 0.0,
        "cond_M": 12.0,
        "support_radius": 0.945,
    }
    saved_path = save_patch_visualization(output_path, patch, prediction, title_prefix="sample")
    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_patch_field_visualization_writes_png_for_contour_and_hybrid3d(tmp_path: Path) -> None:
    patch = {
        "X": np.asarray([[0.0, 0.0], [0.7, 0.0], [0.0, 0.8], [0.8, 0.7]], dtype=np.float64),
        "x_q": np.asarray([0.1, 0.1], dtype=np.float64),
        "phi_ref": np.asarray([0.55, 0.20, 0.15, 0.10], dtype=np.float64),
        "beta": 1.25,
        "r_max": 0.95,
        "patch_type": "uniform",
    }
    prediction = {
        "phi_pred": np.asarray([0.50, 0.25, 0.15, 0.10], dtype=np.float64),
        "phi_base": np.asarray([0.48, 0.24, 0.17, 0.11], dtype=np.float64),
        "relative_l2": 0.09,
        "global_linf": 0.05,
        "neg_fraction": 0.0,
        "max_negative_magnitude": 0.0,
        "cond_M": 15.0,
        "support_radius": 1.0,
    }
    contour_path = tmp_path / "patch_contour.png"
    hybrid3d_path = tmp_path / "patch_hybrid3d.png"
    for output_path, view in ((contour_path, "contour"), (hybrid3d_path, "hybrid3d")):
        saved_path = save_patch_field_visualization(output_path, patch, prediction, view=view, title_prefix="sample")
        assert saved_path == output_path
        assert output_path.is_file()
        assert output_path.stat().st_size > 0


def test_resolve_basis_node_index_defaults_to_max_reference_weight() -> None:
    patch = {
        "phi_ref": np.asarray([0.05, 0.40, 0.20, 0.35], dtype=np.float64),
    }
    assert resolve_basis_node_index(patch, node_index=None) == 1
    assert resolve_basis_node_index(patch, node_index=2) == 2


def test_save_basis_field_visualization_writes_png_for_contour_and_hybrid3d(tmp_path: Path) -> None:
    output_contour = tmp_path / "basis_contour.png"
    output_hybrid3d = tmp_path / "basis_hybrid3d.png"
    patch = {
        "X": np.asarray([[0.0, 0.0], [0.7, 0.0], [0.0, 0.8], [0.8, 0.7]], dtype=np.float64),
        "phi_ref": np.asarray([0.55, 0.20, 0.15, 0.10], dtype=np.float64),
        "beta": 1.25,
        "patch_type": "uniform",
    }
    field = {
        "grid_x": np.asarray([[0.0, 0.5], [0.0, 0.5]], dtype=np.float64),
        "grid_y": np.asarray([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64),
        "teacher_field": np.asarray([[0.5, 0.3], [0.2, 0.1]], dtype=np.float64),
        "pred_field": np.asarray([[0.45, 0.28], [0.22, 0.12]], dtype=np.float64),
        "abs_error": np.asarray([[0.05, 0.02], [0.02, 0.02]], dtype=np.float64),
        "rel_error": np.asarray([[0.10, 0.0667], [0.10, 0.20]], dtype=np.float64),
        "valid_mask": np.asarray([[True, True], [True, True]]),
        "node_index": 1,
        "node_coord": np.asarray([0.7, 0.0], dtype=np.float64),
        "teacher_success_ratio": 1.0,
    }
    for output_path, view in ((output_contour, "basis_contour"), (output_hybrid3d, "basis_hybrid3d")):
        saved_path = save_basis_field_visualization(output_path, patch, field, view=view, title_prefix="basis")
        assert saved_path == output_path
        assert output_path.is_file()
        assert output_path.stat().st_size > 0


def test_build_clipped_relative_error_masks_small_reference_and_clips_large_values() -> None:
    phi_pred = np.asarray([0.8, 1.0, 0.5], dtype=np.float64)
    phi_ref = np.asarray([0.0, 1.0e-4, 0.1], dtype=np.float64)
    values, masked = build_clipped_relative_error(phi_pred, phi_ref)
    assert bool(masked[0])
    assert bool(masked[1])
    assert np.isnan(values[0])
    assert np.isnan(values[1])
    assert not bool(masked[2])
    assert abs(values[2] - REL_ERROR_CLIP) < 1.0e-12
    assert REL_ERROR_FLOOR == 1.0e-3


def test_save_training_curves_visualization_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "training_curves.png"
    curves = {
        "epoch": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        "lr": np.asarray([1.0e-3, 5.0e-4, 1.0e-5], dtype=np.float64),
        "train_total": np.asarray([1.5, 1.0, 0.7], dtype=np.float64),
        "val_total": np.asarray([1.6, 1.1, 0.8], dtype=np.float64),
        "train_relative_l2": np.asarray([0.4, 0.3, 0.2], dtype=np.float64),
        "val_relative_l2": np.asarray([0.42, 0.31, 0.22], dtype=np.float64),
        "train_loss_data": np.asarray([1.2, 0.8, 0.5], dtype=np.float64),
        "val_loss_data": np.asarray([1.3, 0.9, 0.6], dtype=np.float64),
    }
    saved_path = save_training_curves_visualization(output_path, curves)
    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
