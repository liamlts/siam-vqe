"""Tests for siam_vqe.observables_l3."""
from __future__ import annotations

import pytest
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.hamiltonian_l3 import L3Params
from siam_vqe.observables_l3 import (
    compute_reference_observables,
    evaluate_observable_on_sector,
    make_l3_observables,
)
from siam_vqe.reference_l3 import compute_l3_reference


def test_make_l3_observables_returns_required_keys():
    obs = make_l3_observables()
    required = {"n_d", "n_p", "S2", "n_d_3z2", "n_d_x2y2",
                "n_d_xz", "n_d_yz", "n_d_xy"}
    assert required.issubset(set(obs.keys()))


def test_n_d_observable_is_fermionic_op():
    obs = make_l3_observables()
    assert isinstance(obs["n_d"], FermionicOp)


def test_n_d_observable_is_sum_of_d_mode_number_ops():
    """n_d = Σ_{α∈d, σ} n_{α σ}. Confirm the 10 d-mode number ops are exactly the labels."""
    obs = make_l3_observables()
    n_d = obs["n_d"].simplify()
    expected_keys = set()
    for spin in (0, 1):
        for alpha in range(5):
            mode = alpha + spin * 10  # n_s = 10
            expected_keys.add(f"+_{mode} -_{mode}")
    actual_keys = {k for k, _ in n_d.items()}
    assert actual_keys == expected_keys


def test_n_p_observable_is_sum_of_bath_mode_number_ops():
    obs = make_l3_observables()
    n_p = obs["n_p"].simplify()
    expected_keys = set()
    for spin in (0, 1):
        for beta in range(5):
            mode = 5 + beta + spin * 10
            expected_keys.add(f"+_{mode} -_{mode}")
    actual_keys = {k for k, _ in n_p.items()}
    assert actual_keys == expected_keys


def test_n_d_plus_n_p_total_is_20():
    """⟨n_d⟩ + ⟨n_p⟩ summed over all 20 modes (coefficient sum, before sector projection) is 20."""
    obs = make_l3_observables()
    total = (obs["n_d"] + obs["n_p"]).simplify()
    total_sum = sum(c for _, c in total.items())
    assert total_sum == pytest.approx(20.0)


def test_evaluate_n_d_on_l3_ground_is_near_8():
    """For NiO d⁸ ground state, ⟨n_d⟩ should be ~8 (with some charge transfer)."""
    p = L3Params()
    ref = compute_l3_reference(p, k_states=2)
    obs = make_l3_observables()
    n_d = evaluate_observable_on_sector(obs["n_d"], ref.ground_vector,
                                         basis=ref.basis,
                                         num_spin_orbitals=20)
    assert 7.5 <= n_d <= 9.0, f"⟨n_d⟩ = {n_d}, outside expected [7.5, 9.0]"


def test_evaluate_n_d_plus_n_p_equals_18():
    p = L3Params()
    ref = compute_l3_reference(p, k_states=2)
    obs = make_l3_observables()
    n_d = evaluate_observable_on_sector(obs["n_d"], ref.ground_vector,
                                         basis=ref.basis, num_spin_orbitals=20)
    n_p = evaluate_observable_on_sector(obs["n_p"], ref.ground_vector,
                                         basis=ref.basis, num_spin_orbitals=20)
    assert (n_d + n_p) == pytest.approx(18.0, abs=1e-8)


def test_evaluate_S2_on_l3_ground_is_near_2():
    """For ³A_2g (S=1) ground state, ⟨S²⟩ = S(S+1) = 2."""
    p = L3Params()
    ref = compute_l3_reference(p, k_states=2)
    obs = make_l3_observables()
    s2 = evaluate_observable_on_sector(obs["S2"], ref.ground_vector,
                                         basis=ref.basis, num_spin_orbitals=20)
    assert s2 == pytest.approx(2.0, abs=0.1)


def test_compute_reference_observables_fills_dict():
    p = L3Params()
    ref = compute_l3_reference(p, k_states=2)
    obs_dict = compute_reference_observables(ref)
    required = {"n_d", "n_p", "S2"}
    assert required.issubset(obs_dict.keys())
    assert obs_dict["n_d"] + obs_dict["n_p"] == pytest.approx(18.0, abs=1e-8)
