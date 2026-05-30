"""Tests for dimer observables — n_d_total, S^2, double occupancy."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.hamiltonian import hubbard_dimer, nio_l1_anderson, observables_dimer, observables_l1


def _ground_state_expectation(
    op: FermionicOp, h_op: FermionicOp, n_target: int = 2
) -> float:
    """Diagonalize h_op in the N=n_target sector, take lowest eigenvector, evaluate <op>."""
    mapper = JordanWignerMapper()
    h_matrix = mapper.map(h_op).to_matrix()
    op_matrix = mapper.map(op).to_matrix()
    # Build particle-number operator and project to N=n_target sector.
    n_op = FermionicOp(
        {f"+_{i} -_{i}": 1.0 for i in range(h_op.num_spin_orbitals)},
        num_spin_orbitals=h_op.num_spin_orbitals,
    )
    n_diag = np.diag(mapper.map(n_op).to_matrix()).real
    mask = np.abs(n_diag - n_target) < 1e-8
    h_n = h_matrix[np.ix_(mask, mask)]
    op_n = op_matrix[np.ix_(mask, mask)]
    _eigvals, eigvecs = np.linalg.eigh(h_n)
    psi0 = eigvecs[:, 0]
    return complex(psi0.conj() @ op_n @ psi0).real


def test_observables_returns_dict() -> None:
    obs = observables_dimer()
    assert set(obs.keys()) == {"n_total", "n_site0", "n_site1", "S2", "double_occ"}
    for v in obs.values():
        assert isinstance(v, FermionicOp)
        assert v.num_spin_orbitals == 4


def test_dimer_total_particle_count_is_two_at_half_filling(
    dimer_params: dict[str, float],
) -> None:
    h = hubbard_dimer(**dimer_params)
    obs = observables_dimer()
    # The N=2 sector ground state of the Hubbard dimer at U=4, t=1, eps=0 is at half-filling.
    n_expect = _ground_state_expectation(obs["n_total"], h)
    assert n_expect == pytest.approx(2.0, abs=1e-10)


def test_dimer_S2_is_zero_for_singlet_ground_state(dimer_params: dict[str, float]) -> None:
    """The Hubbard dimer ground state at half-filling is the singlet (S=0, so S^2=0)."""
    h = hubbard_dimer(**dimer_params)
    obs = observables_dimer()
    s2 = _ground_state_expectation(obs["S2"], h)
    assert s2 == pytest.approx(0.0, abs=1e-10)


def test_dimer_double_occupancy_decreases_with_increasing_U() -> None:
    """As U increases, double-occupancy on either site should drop."""
    obs = observables_dimer()
    d_values = []
    for U in [0.5, 2.0, 8.0]:
        h = hubbard_dimer(U=U, t=1.0, eps=0.0)
        d = _ground_state_expectation(obs["double_occ"], h)
        d_values.append(d)
    assert d_values[0] > d_values[1] > d_values[2]


# ---------------------------------------------------------------------------
# L1 Anderson impurity model observables
# ---------------------------------------------------------------------------


def test_observables_l1_returns_dict() -> None:
    obs = observables_l1()
    assert set(obs.keys()) == {"n_d_total", "n_p_total", "S2_d", "double_occ_d"}
    for v in obs.values():
        assert isinstance(v, FermionicOp)
        assert v.num_spin_orbitals == 4


def test_observables_l1_n_d_plus_n_p_is_total_number() -> None:
    """n_d_total + n_p_total = total particle number."""
    obs = observables_l1()
    # Construct H with arbitrary nontrivial parameters and verify
    # <n_d> + <n_p> = 2 in the half-filled GS (lowest sector containing 2 electrons).
    fop = nio_l1_anderson(U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5)
    mapper = JordanWignerMapper()
    h_matrix = mapper.map(fop).to_matrix()
    eigvals, eigvecs = np.linalg.eigh(h_matrix)
    # Find lowest eigenstate with <N>=2 by scanning until we hit half-filling.
    n_total_fop = (obs["n_d_total"] + obs["n_p_total"]).simplify()
    n_total_mat = mapper.map(n_total_fop).to_matrix()
    for i in range(len(eigvals)):
        psi = eigvecs[:, i]
        n_expect = complex(psi.conj() @ n_total_mat @ psi).real
        if abs(n_expect - 2.0) < 1e-9:
            break
    else:
        raise AssertionError("no half-filled eigenstate found in spectrum")
    nd = complex(psi.conj() @ mapper.map(obs["n_d_total"]).to_matrix() @ psi).real
    np_ = complex(psi.conj() @ mapper.map(obs["n_p_total"]).to_matrix() @ psi).real
    assert (nd + np_) == pytest.approx(2.0, abs=1e-9)


def test_observables_l1_S2_d_is_physical_on_kondo_singlet() -> None:
    """For U=4, V=1, ε=0 the SIAM ground state is the Kondo singlet (total S=0). The impurity-only S²_d is nonzero because the impurity spin is entangled with the bath; for these parameters <S²_d> = 3/4 <n_d> - 3/2 <double_occ_d> ≈ 0.407."""
    obs = observables_l1()
    fop = nio_l1_anderson(U=4.0, V=1.0, eps_d=0.0, eps_p=0.0)
    mapper = JordanWignerMapper()
    h_matrix = mapper.map(fop).to_matrix()
    s2_matrix = mapper.map(obs["S2_d"]).to_matrix()
    _eigvals, eigvecs = np.linalg.eigh(h_matrix)
    psi0 = eigvecs[:, 0]
    s2_d = complex(psi0.conj() @ s2_matrix @ psi0).real
    assert 0.3 <= s2_d <= 0.6


def test_observables_l1_double_occ_decreases_with_U() -> None:
    obs = observables_l1()
    mapper = JordanWignerMapper()
    docc_matrix = mapper.map(obs["double_occ_d"]).to_matrix()
    docc_values = []
    for U in [0.0, 4.0, 16.0]:
        fop = nio_l1_anderson(U=U, V=1.0, eps_d=0.0, eps_p=0.0)
        h = mapper.map(fop).to_matrix()
        _, vecs = np.linalg.eigh(h)
        docc_values.append(complex(vecs[:, 0].conj() @ docc_matrix @ vecs[:, 0]).real)
    assert docc_values[0] > docc_values[1] > docc_values[2]
