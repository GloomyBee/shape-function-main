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
