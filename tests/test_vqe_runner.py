"""Tests for siam_vqe.vqe_runner — VQE orchestration on noiseless simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from siam_vqe.ansatz import efficient_su2_ansatz, uccsd_ansatz
from siam_vqe.hamiltonian import hubbard_dimer
from siam_vqe.mappings import to_qubit_op
from siam_vqe.vqe_runner import VQEResult, run_vqe, run_vqe_multistart


def _analytic_dimer_gs(U: float, t: float, eps: float = 0.0) -> float:
    return eps * 2 + U / 2 - math.sqrt((U / 2) ** 2 + 4 * t**2)


def test_run_vqe_returns_vqeresult(dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    circuit, x0 = efficient_su2_ansatz(num_qubits=pop.num_qubits, reps=2, seed=1)
    result = run_vqe(pop, circuit, x0, optimizer="COBYLA", maxiter=200)
    assert isinstance(result, VQEResult)
    assert isinstance(result.energy, float)
    assert result.params.shape == (circuit.num_parameters,)
    assert len(result.history) > 0
    assert result.walltime_s > 0.0
    assert result.n_qubits == pop.num_qubits


def test_run_vqe_efficient_su2_converges_to_dimer_groundstate(
    dimer_params: dict[str, float],
) -> None:
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    circuit, x0 = efficient_su2_ansatz(num_qubits=pop.num_qubits, reps=3, seed=20260524)
    result = run_vqe(pop, circuit, x0, optimizer="COBYLA", maxiter=600)
    expected = _analytic_dimer_gs(**dimer_params)
    # EfficientSU2 with reps=3 on a 2-qubit problem can reach the GS within mHa.
    assert result.energy == pytest.approx(expected, abs=1e-3)


def test_run_vqe_uccsd_reaches_dimer_groundstate(dimer_params: dict[str, float]) -> None:
    """UCCSD is physics-motivated and should reach the GS within tight tolerance.

    reps=2 (two Trotter steps) is required: reps=1 has a landscape local minimum
    at ~-0.5 Hartree that traps gradient-based optimizers starting from x0=0.
    """
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    circuit, x0 = uccsd_ansatz(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        mapper_scheme="parity_tapered",
        reps=2,
    )
    result = run_vqe(pop, circuit, x0, optimizer="SLSQP", maxiter=200)
    expected = _analytic_dimer_gs(**dimer_params)
    assert result.energy == pytest.approx(expected, abs=1e-6)


def test_run_vqe_history_monotone_nonincreasing_best(dimer_params: dict[str, float]) -> None:
    """The running BEST energy over iterations should never increase."""
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    circuit, x0 = efficient_su2_ansatz(num_qubits=pop.num_qubits, reps=2, seed=7)
    result = run_vqe(pop, circuit, x0, optimizer="COBYLA", maxiter=200)
    energies = [e for _, e in result.history]
    running_min = np.minimum.accumulate(energies)
    assert np.all(np.diff(running_min) <= 1e-12)


def test_run_vqe_multistart_returns_n_results(dimer_params: dict[str, float]) -> None:
    from siam_vqe.vqe_runner import MultistartResult
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    circuit, _ = efficient_su2_ansatz(num_qubits=pop.num_qubits, reps=2, seed=0)
    result = run_vqe_multistart(
        pop, circuit, n_starts=4, ansatz_factory_seed_base=100,
        ansatz_num_qubits=pop.num_qubits, ansatz_reps=2,
        optimizer="COBYLA", maxiter=150,
    )
    assert isinstance(result, MultistartResult)
    assert len(result.runs) == 4
    energies = [r.energy for r in result.runs]
    assert result.best.energy == min(energies)
    assert min(energies) == pytest.approx(_analytic_dimer_gs(**dimer_params), abs=5e-3)
    # Spread sanity check: all starts should find the ground state on this trivial problem.
    spread = max(energies) - min(energies)
    assert spread < 1e-1  # 100 mHa is generous for 2-qubit dimer
