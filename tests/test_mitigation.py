"""Unit tests for the mitigation grid module."""

from __future__ import annotations

import pytest

from siam_vqe.mitigation import MitigationSpec, make_default_grid


def test_mitigation_spec_is_frozen_dataclass() -> None:
    spec = MitigationSpec(
        name="test", m3=True, zne=False,
        zne_extrapolator=None, zne_noise_factors=None,
    )
    with pytest.raises(AttributeError):
        spec.name = "mutated"  # frozen dataclass must reject this


def test_mitigation_spec_defaults_shots_to_8192() -> None:
    spec = MitigationSpec(
        name="x", m3=False, zne=False, zne_extrapolator=None, zne_noise_factors=None,
    )
    assert spec.shots == 8192


def test_default_grid_has_8_configs() -> None:
    grid = make_default_grid()
    assert len(grid) == 8


def test_default_grid_first_spec_is_no_mit() -> None:
    grid = make_default_grid()
    assert grid[0].name == "no_mit"
    assert grid[0].m3 is False
    assert grid[0].zne is False


def test_default_grid_unique_names() -> None:
    grid = make_default_grid()
    names = [s.name for s in grid]
    assert len(set(names)) == len(names), f"duplicate names in grid: {names}"


def test_default_grid_includes_all_8_specs() -> None:
    grid = make_default_grid()
    names = {s.name for s in grid}
    assert names == {
        "no_mit",
        "m3_only",
        "zne_lin_135",
        "m3_zne_lin_135",
        "m3_zne_exp_135",
        "m3_zne_poly3_135",
        "m3_zne_lin_123",
        "m3_zne_lin_12345",
    }


def test_zne_configs_have_factor_tuple() -> None:
    grid = make_default_grid()
    for spec in grid:
        if spec.zne:
            assert isinstance(spec.zne_noise_factors, tuple)
            assert all(isinstance(x, (int, float)) for x in spec.zne_noise_factors)
            assert spec.zne_noise_factors[0] == 1.0 or spec.zne_noise_factors[0] == 1
            assert spec.zne_extrapolator is not None


def test_non_zne_configs_have_no_factor_tuple() -> None:
    grid = make_default_grid()
    for spec in grid:
        if not spec.zne:
            assert spec.zne_noise_factors is None
            assert spec.zne_extrapolator is None
