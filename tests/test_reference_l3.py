"""Tests for siam_vqe.reference_l3."""
from __future__ import annotations

import numpy as np
import pytest

from siam_vqe.hamiltonian_l3 import L3Params
from siam_vqe.reference_l3 import (
    L3Reference,
    compute_l3_reference,
    inspect_multiplets,
    load_l3_reference,
    save_l3_reference,
)


def test_compute_l3_reference_returns_L3Reference():
    p = L3Params()
    ref = compute_l3_reference(p, k_states=6)
    assert isinstance(ref, L3Reference)


def test_l3_reference_ground_energy_finite():
    p = L3Params()
    ref = compute_l3_reference(p, k_states=6)
    assert np.isfinite(ref.ground_energy)


def test_l3_reference_first_excited_above_ground():
    p = L3Params()
    ref = compute_l3_reference(p, k_states=6)
    assert ref.excited_energies[0] > ref.ground_energy


def test_l3_reference_sector_dim_is_100():
    p = L3Params()
    ref = compute_l3_reference(p, k_states=3)
    assert ref.sector_dim == 100
    assert ref.ground_vector.shape == (100,)


def test_l3_reference_ground_vector_normalized():
    p = L3Params()
    ref = compute_l3_reference(p, k_states=3)
    assert abs(np.linalg.norm(ref.ground_vector) - 1.0) < 1e-10


def test_l3_reference_npz_roundtrip(tmp_path):
    p = L3Params()
    ref = compute_l3_reference(p, k_states=4)
    path = tmp_path / "nio_l3_reference.npz"
    save_l3_reference(ref, path)
    loaded = load_l3_reference(path)

    assert loaded.ground_energy == pytest.approx(ref.ground_energy)
    assert loaded.sector_dim == ref.sector_dim
    assert loaded.sector == ref.sector
    np.testing.assert_array_equal(loaded.basis, ref.basis)
    np.testing.assert_allclose(loaded.ground_vector, ref.ground_vector)
    np.testing.assert_allclose(loaded.excited_energies, ref.excited_energies)


def test_inspect_multiplets_returns_table_rows():
    p = L3Params()
    ref = compute_l3_reference(p, k_states=5)
    table = inspect_multiplets(ref)
    assert isinstance(table, list)
    assert len(table) == 5
    for row in table:
        assert "index" in row
        assert "energy_eV" in row
        assert "gap_eV" in row
    assert table[0]["gap_eV"] == pytest.approx(0.0)


def test_compute_l3_xas_reference_returns_valid_in_sector():
    from siam_vqe.core_hole import CoreHoleParams
    from siam_vqe.hamiltonian_l3 import L3Params
    from siam_vqe.reference_l3 import compute_l3_xas_reference

    params = L3Params()
    ch = CoreHoleParams()
    ref = compute_l3_xas_reference(params, ch, num_particles=(10, 9), k_states=5)
    assert ref.num_particles == (10, 9)
    assert len(ref.energies) == 5
    # Lowest eigenvalue should be ~10 × U_dc below the bare H GS (rough bound)
    # — at least, it should be < bare-H GS at (10, 9)
    from siam_vqe.reference_l3 import compute_l3_reference
    ref_bare = compute_l3_reference(params, num_particles=(10, 9), k_states=1)
    assert ref.ground_energy < ref_bare.ground_energy
