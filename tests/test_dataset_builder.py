from __future__ import annotations

from shape_function.data.dataset_builder import build_dataset


def test_build_dataset_consumes_patch_types_and_beta_range() -> None:
    dataset = build_dataset(
        num_patches=4,
        seed=5,
        feature_mode="minimal",
        k_neighbors=16,
        patch_types=("uniform",),
        beta_range=(1.25, 1.25),
    )
    assert len(dataset) == 4
    assert {item["patch_type"] for item in dataset} == {"uniform"}
    assert all(abs(item["beta"] - 1.25) < 1.0e-12 for item in dataset)
    assert all("phi_ref" not in item for item in dataset)


def test_build_dataset_teacher_mode_still_produces_phi_ref() -> None:
    dataset = build_dataset(
        num_patches=2,
        seed=7,
        feature_mode="minimal",
        k_neighbors=16,
        patch_types=("uniform",),
        beta_range=(1.0, 1.0),
        supervision_mode="teacher",
    )
    assert len(dataset) == 2
    assert all("phi_ref" in item for item in dataset)
