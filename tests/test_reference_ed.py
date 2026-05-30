"""Tests for siam_vqe.reference_ed — scipy.sparse exact diagonalization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from siam_vqe.hamiltonian import hubbard_dimer
from siam_vqe.mappings import to_qubit_op
from siam_vqe.reference_ed import EDResult, exact_diag


def test_ed_result_fields(dimer_params: dict[str, float]) -> None:
    pop = to_qubit_op(hubbard_dimer(**dimer_params), scheme="jw", num_particles=(1, 1))
    res = exact_diag(pop, k=4)
    assert isinstance(res, EDResult)
    assert res.energies.shape == (4,)
    assert res.vectors.shape[1] == 4
    assert res.vectors.shape[0] == 2**pop.num_qubits
    # Energies sorted ascending.
    assert np.all(np.diff(res.energies) >= -1e-12)


def test_ed_dimer_groundstate_matches_analytic(dimer_params: dict[str, float]) -> None:
    """The lowest eigenvalue of the parity_tapered (N=2 Sz=0) Hamiltonian is
    the half-filling singlet GS, which matches the analytic formula."""
    pop = to_qubit_op(hubbard_dimer(**dimer_params), scheme="parity_tapered", num_particles=(1, 1))
    res = exact_diag(pop, k=1)
    expected = (
        dimer_params["eps"] * 2
        + dimer_params["U"] / 2
        - math.sqrt((dimer_params["U"] / 2) ** 2 + 4 * dimer_params["t"] ** 2)
    )
    assert res.energies[0] == pytest.approx(expected, abs=1e-9)


def test_ed_sparse_matches_dense(dimer_params: dict[str, float]) -> None:
    """For a 4-qubit Hamiltonian, dense and sparse should agree to numerical precision."""
    pop = to_qubit_op(hubbard_dimer(**dimer_params), scheme="jw", num_particles=(1, 1))
    sparse_res = exact_diag(pop, k=4)
    dense_eigvals = np.linalg.eigvalsh(pop.to_matrix())
    assert np.allclose(sparse_res.energies, dense_eigvals[:4], atol=1e-8)


def test_ed_groundstate_normalized(dimer_params: dict[str, float]) -> None:
    pop = to_qubit_op(hubbard_dimer(**dimer_params), scheme="jw", num_particles=(1, 1))
    res = exact_diag(pop, k=1)
    psi = res.vectors[:, 0]
    assert np.abs(np.linalg.norm(psi) - 1.0) < 1e-10
