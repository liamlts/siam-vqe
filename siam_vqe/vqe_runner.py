"""VQE orchestration on noiseless statevector simulator.

This module owns ONE thing: turning (Hamiltonian, ansatz, x0, optimizer) into a
frozen VQEResult. It does not construct Hamiltonians, does not pick ansatze, does
not handle noise — those live in their own modules.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA, Optimizer

OptimizerName = Literal["COBYLA", "SLSQP", "SPSA"]


@dataclass(frozen=True)
class VQEResult:
    """Frozen record of a VQE run."""

    energy: float
    params: np.ndarray
    circuit: QuantumCircuit
    history: list[tuple[int, float]]
    walltime_s: float
    optimizer_name: str
    n_qubits: int
    ansatz_name: str
    seed: int | None = None


@dataclass(frozen=True)
class MultistartResult:
    """Aggregated result from a multi-start VQE run.

    Attributes
    ----------
    runs : all individual VQE results, sorted by energy ascending.
    best : the run with the lowest energy.
    spread_best_to_median : |E_median - E_best| across all starts (eV).
        Measures how sharp the energy landscape basin is.
    """

    runs: list[VQEResult]
    best: VQEResult
    spread_best_to_median: float


def run_vqe(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    x0: np.ndarray,
    optimizer: OptimizerName = "COBYLA",
    maxiter: int = 500,
    seed: int | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> VQEResult:
    """Run noiseless VQE and return a frozen VQEResult."""
    if ansatz.num_parameters != x0.size:
        raise ValueError(
            f"x0.size={x0.size} does not match ansatz.num_parameters={ansatz.num_parameters}"
        )

    estimator = StatevectorEstimator(seed=seed)
    history: list[tuple[int, float]] = []
    eval_counter = [0]

    def cost(params: np.ndarray) -> float:
        eval_counter[0] += 1
        job = estimator.run([(ansatz, hamiltonian, params)])
        result = job.result()[0]
        energy = float(result.data.evs)
        history.append((eval_counter[0], energy))
        if callback is not None:
            callback(eval_counter[0], energy)
        return energy

    opt = _build_optimizer(optimizer, maxiter)

    t0 = time.perf_counter()
    opt_result = opt.minimize(fun=cost, x0=x0)
    walltime = time.perf_counter() - t0

    return VQEResult(
        energy=float(opt_result.fun),
        params=np.asarray(opt_result.x, dtype=float),
        circuit=ansatz,
        history=history,
        walltime_s=walltime,
        optimizer_name=optimizer,
        n_qubits=hamiltonian.num_qubits,
        ansatz_name=ansatz.name or "ansatz",
        seed=seed,
    )


def _build_optimizer(name: OptimizerName, maxiter: int) -> Optimizer:
    if name == "COBYLA":
        return COBYLA(maxiter=maxiter)
    if name == "SLSQP":
        return SLSQP(maxiter=maxiter)
    if name == "SPSA":
        return SPSA(maxiter=maxiter)
    raise ValueError(f"unknown optimizer: {name!r}")


def run_vqe_multistart(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    n_starts: int,
    ansatz_factory_seed_base: int | None = None,
    ansatz_num_qubits: int | None = None,
    ansatz_reps: int | None = None,
    optimizer: OptimizerName = "COBYLA",
    maxiter: int = 300,
    seed: int | None = None,
) -> MultistartResult:
    """Run VQE from `n_starts` independently-seeded initial points.

    Simplified call (UCCSD / any fixed ansatz with x0=0 physics start):
        run_vqe_multistart(hamiltonian, ansatz, n_starts=8,
                           optimizer='SLSQP', maxiter=200, seed=20260524)
    Each start perturbs x0 with small Gaussian noise seeded from `seed + i`.
    For UCCSD the physics-motivated x0=0 is start i=0; remaining starts
    explore the basin.

    Legacy call (EfficientSU2-style, random x0 from factory):
        run_vqe_multistart(hamiltonian, ansatz, n_starts=4,
                           ansatz_factory_seed_base=100,
                           ansatz_num_qubits=2, ansatz_reps=2,
                           optimizer='COBYLA', maxiter=150)

    Returns
    -------
    MultistartResult with .runs, .best, .spread_best_to_median. Iterate
    runs via `result.runs`, not the result itself.
    """
    rng_seed = seed if seed is not None else (ansatz_factory_seed_base if ansatz_factory_seed_base is not None else 0)

    if ansatz_factory_seed_base is not None and ansatz_num_qubits is not None and ansatz_reps is not None:
        # Legacy EfficientSU2-style path: draw random x0 from the factory for each start.
        from siam_vqe.ansatz import efficient_su2_ansatz  # local import to avoid cycle
        vqe_results: list[VQEResult] = []
        for i in range(n_starts):
            s = ansatz_factory_seed_base + i
            _, x0 = efficient_su2_ansatz(
                num_qubits=ansatz_num_qubits,
                reps=ansatz_reps,
                seed=s,
            )
            result = run_vqe(
                hamiltonian, ansatz, x0,
                optimizer=optimizer, maxiter=maxiter,
                seed=s,
            )
            vqe_results.append(result)
    else:
        # Simplified path: perturb x0=0 with small Gaussian noise per start.
        # Start 0 uses x0=0 (physics-motivated for UCCSD).
        rng = np.random.default_rng(rng_seed)
        n_params = ansatz.num_parameters
        vqe_results = []
        for i in range(n_starts):
            x0 = np.zeros(n_params) if i == 0 else rng.normal(0.0, 0.05, size=n_params)
            result = run_vqe(
                hamiltonian, ansatz, x0,
                optimizer=optimizer, maxiter=maxiter,
                seed=rng_seed + i,
            )
            vqe_results.append(result)

    energies = np.array([r.energy for r in vqe_results])
    best = vqe_results[int(np.argmin(energies))]
    spread_best_to_median = float(abs(np.median(energies) - energies.min()))
    sorted_results = sorted(vqe_results, key=lambda r: r.energy)

    return MultistartResult(
        runs=sorted_results,
        best=best,
        spread_best_to_median=spread_best_to_median,
    )
