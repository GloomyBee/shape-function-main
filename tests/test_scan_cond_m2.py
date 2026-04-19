from __future__ import annotations

from shape_function.eval.scan_cond_m2 import scan_cond_m2


def test_scan_cond_m2_returns_overall_and_grouped_stats() -> None:
    payload = scan_cond_m2(
        num_patches=4,
        seed=3,
        k_neighbors=16,
        beta_range=(1.0, 1.0),
        kappa_max=1.0e8,
        patch_types=("uniform", "mildly_perturbed"),
    )
    assert payload["overall"]["mean_cond_M2"] > 0.0
    assert set(payload["by_patch_type"]) == {"uniform", "mildly_perturbed"}
    assert "fallback_rate" in payload["overall"]
    assert "suggested_kappa_max" in payload
