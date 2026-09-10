"""L1 core-hole Hamiltonian helpers — atomic-limit gates.

The L1 model (`nio_l1_anderson`) uses Phase 1's all-up-then-all-down mode
layout: 0=d↑, 1=p↑, 2=d↓, 3=p↓. V_core acts only on the impurity (d) modes
0 and 2; the ligand p modes are core-hole-blind.
"""
from __future__ import annotations

import numpy as np


def test_v_core_l1_zero_when_udc_zero():
    """V_core(U_dc=0) is the zero FermionicOp on L1 modes."""
    from siam_vqe.core_hole import CoreHoleParams, v_core_operator_l1

    ch = CoreHoleParams(U_dc=0.0)
    v = v_core_operator_l1(ch)
    assert len(v.simplify()) == 0


def test_v_core_l1_hermitian():
    """V_core is Hermitian for any U_dc."""
    from siam_vqe.core_hole import CoreHoleParams, v_core_operator_l1

    ch = CoreHoleParams(U_dc=8.5)
    v = v_core_operator_l1(ch)
    v_dag = v.adjoint().simplify()
    assert v.simplify() == v_dag


def test_v_core_l1_acts_only_on_impurity_modes():
    """V_core_L1 touches modes 0 and 2 only (impurity d↑, d↓)."""
    from siam_vqe.core_hole import CoreHoleParams, v_core_operator_l1
    from siam_vqe.hamiltonian import _L1_IMPURITY_MODES

    ch = CoreHoleParams(U_dc=8.5)
    v = v_core_operator_l1(ch)
    impurity_modes = set(_L1_IMPURITY_MODES)
    for label, _ in v.terms():
        modes_in_term = {mode for _, mode in label}
        assert modes_in_term <= impurity_modes, (
            f"V_core_L1 touches non-impurity mode in {label}"
        )


def test_v_core_l1_coefficient_is_negative_udc():
    """Each n_{d_σ} carries coefficient -U_dc; expect 2 such terms."""
    from siam_vqe.core_hole import CoreHoleParams, v_core_operator_l1

    ch = CoreHoleParams(U_dc=8.5)
    v = v_core_operator_l1(ch)
    coeffs = [c for _, c in v.terms()]
    assert len(coeffs) == 2
    for c in coeffs:
        assert c == -8.5


def test_l1_h_prime_equals_h_when_udc_zero():
    """With U_dc=0, H' should equal H_L1 exactly."""
    from siam_vqe.core_hole import CoreHoleParams, nio_l1_core_hole_hamiltonian
    from siam_vqe.hamiltonian import nio_l1_anderson

    ch = CoreHoleParams(U_dc=0.0)
    h_l1 = nio_l1_anderson(U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5)
    h_prime = nio_l1_core_hole_hamiltonian(
        ch, U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5
    )
    diff = (h_prime - h_l1).simplify()
    assert len(diff) == 0


def test_l1_h_prime_lower_min_energy_for_positive_udc():
    """V_core ≤ 0 (operator inequality) so H' ≤ H, hence min eig(H') < min eig(H)
    whenever the H ground state has nonzero d-occupation (it does at half-filling).
    """
    from siam_vqe.core_hole import CoreHoleParams, nio_l1_core_hole_hamiltonian
    from siam_vqe.hamiltonian import nio_l1_anderson
    from siam_vqe.mappings import to_qubit_op

    h_l1 = nio_l1_anderson(U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5)
    ch = CoreHoleParams(U_dc=8.5)
    h_prime = nio_l1_core_hole_hamiltonian(
        ch, U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5
    )

    h_l1_qop = to_qubit_op(h_l1, scheme="jw")
    hp_qop = to_qubit_op(h_prime, scheme="jw")
    e_l1 = float(np.linalg.eigvalsh(h_l1_qop.to_matrix()).min())
    e_hp = float(np.linalg.eigvalsh(hp_qop.to_matrix()).min())
    assert e_hp < e_l1, (
        f"H' min={e_hp} not below H min={e_l1} for U_dc={ch.U_dc}"
    )
