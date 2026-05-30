"""Validation layers and plotting for VQE results.

The six validation layers from the design spec §5 live here:
    1. check_energy_match         -- |E_VQE - E_ED| within tolerance
    2. check_state_overlap        — |⟨ψ_ED | ψ_VQE⟩|² ≥ threshold
    3. check_ansatz_expressivity  — classical max-overlap probe (Task 13)
    4. check_multistart_spread    — best/median/spread across N seeded starts
    5. check_observable_agreement — observable from VQE state vs ED state
    6. check_noise_mitigation_guardrail — Phase 3, not in Phase 1
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_nature.second_q.operators import FermionicOp
from scipy.optimize import minimize

from siam_vqe.mappings import to_qubit_op as _to_qubit_op
from siam_vqe.reference_ed import EDResult
from siam_vqe.vqe_runner import MultistartResult, VQEResult


@dataclass(frozen=True)
class EnergyMatchReport:
    passed: bool
    delta: float
    tol_hartree: float
    e_vqe: float
    e_ed: float

    @property
    def delta_e(self) -> float:
        """Alias for delta (notebook API compat)."""
        return self.delta


@dataclass(frozen=True)
class OverlapReport:
    passed: bool
    overlap: float
    threshold: float

    @property
    def overlap_sq(self) -> float:
        """Alias for overlap (notebook uses overlap_sq; overlap is already |⟨ψ|ψ⟩|²)."""
        return self.overlap


@dataclass(frozen=True)
class ObservableReport:
    passed: bool
    name: str
    vqe_value: float
    ed_value: float
    rel_error: float
    rel_tol: float


@dataclass(frozen=True)
class _ObservableValues:
    """Per-observable VQE / ED values for multi-observable report."""

    vqe: float
    ed: float
    rel_error: float


@dataclass(frozen=True)
class MultiObservableReport:
    """Return type for multi-observable check_observable_agreement calls.

    Attributes
    ----------
    passed : True iff ALL observables pass their individual thresholds.
    max_rel_error : largest relative error across all observables.
    values : dict of observable name → _ObservableValues(vqe, ed, rel_error).
    """

    passed: bool
    max_rel_error: float
    values: dict[str, _ObservableValues]


@dataclass(frozen=True)
class MultiStartReport:
    passed: bool
    n_starts: int
    best: float
    median: float
    spread: float
    max_spread_hartree: float

    @property
    def spread_best_to_median(self) -> float:
        """Alias for |median - best| (notebook API compat)."""
        return abs(self.median - self.best)


def check_energy_match(
    e_vqe: float,
    e_ed: float,
    tol_hartree: float = 1e-3,
    abs_tol: float | None = None,
) -> EnergyMatchReport:
    """Layer 1: |E_VQE - E_ED| < tol.

    Parameters
    ----------
    abs_tol : alias for tol_hartree (notebook API). If both are given,
        abs_tol takes precedence.
    """
    tol = abs_tol if abs_tol is not None else tol_hartree
    delta = float(e_vqe - e_ed)
    return EnergyMatchReport(
        passed=bool(abs(delta) < tol),
        delta=delta,
        tol_hartree=tol,
        e_vqe=e_vqe,
        e_ed=e_ed,
    )


def check_state_overlap(
    vqe: VQEResult,
    ed_or_pop: EDResult | SparsePauliOp,
    psi_ed_or_threshold: np.ndarray | float = 0.99,
    threshold: float = 0.99,
) -> OverlapReport:
    """Layer 2: |⟨ψ_ED | ψ_VQE⟩|².

    Two call signatures are supported:

    Original (tests):
        check_state_overlap(vqe, ed: EDResult, threshold: float = 0.99)

    Notebook API:
        check_state_overlap(vqe, pop: SparsePauliOp, psi_ed: np.ndarray,
                            threshold: float = 0.99)
        The `pop` argument is accepted but not used — psi_ed is taken directly.
    """
    bound = vqe.circuit.assign_parameters(vqe.params)
    psi_vqe = Statevector(bound).data

    if isinstance(psi_ed_or_threshold, np.ndarray):
        # Notebook call: (vqe, pop, psi_ed_vector, threshold)
        psi_ed = psi_ed_or_threshold
        _threshold = threshold
    elif isinstance(ed_or_pop, EDResult):
        # Original call: (vqe, ed: EDResult, threshold: float)
        psi_ed = ed_or_pop.vectors[:, 0]
        _threshold = float(psi_ed_or_threshold)
    else:
        raise TypeError(
            "check_state_overlap: pass either (vqe, ed: EDResult, threshold) "
            "or (vqe, pop, psi_ed: np.ndarray, threshold)"
        )

    overlap = float(np.abs(np.vdot(psi_ed, psi_vqe)) ** 2)
    # Clamp to [0, 1] to absorb floating-point noise near the boundary.
    overlap = min(overlap, 1.0)
    return OverlapReport(passed=bool(overlap >= _threshold), overlap=overlap, threshold=_threshold)


def check_observable_agreement(
    vqe: VQEResult,
    ed: EDResult,
    observable: SparsePauliOp,
    rel_tol: float = 0.05,
    name: str = "observable",
    abs_tol: float = 1e-4,
) -> ObservableReport:
    """Layer 5 (single observable): ⟨ψ | O | ψ⟩ from VQE vs ED within rel_tol.

    For observables whose ED expectation value is near zero (e.g. S² in a
    singlet sector after parity tapering), the relative error is ill-defined.
    The check passes if EITHER the relative error is below rel_tol OR the
    absolute difference is below abs_tol.
    """
    psi_ed = ed.vectors[:, 0]
    bound = vqe.circuit.assign_parameters(vqe.params)
    psi_vqe = Statevector(bound).data
    o_mat = observable.to_matrix()
    vqe_val = float((psi_vqe.conj() @ o_mat @ psi_vqe).real)
    ed_val = float((psi_ed.conj() @ o_mat @ psi_ed).real)
    abs_diff = abs(vqe_val - ed_val)
    rel_err = abs_diff / max(abs(ed_val), abs_tol)
    passed = bool(rel_err < rel_tol or abs_diff < abs_tol)
    return ObservableReport(
        passed=passed,
        name=name,
        vqe_value=vqe_val,
        ed_value=ed_val,
        rel_error=rel_err,
        rel_tol=rel_tol,
    )


def check_observable_agreement_multi(
    vqe: VQEResult,
    pop: SparsePauliOp,
    psi_ed: np.ndarray,
    obs_dict: dict[str, SparsePauliOp | FermionicOp],
    num_particles: tuple[int, int],
    rel_tol: float = 0.05,
    abs_tol: float = 1e-4,
) -> MultiObservableReport:
    """Layer 5 multi-observable check: every observable in obs_dict agrees within rel_tol.

    FermionicOp values are auto-mapped via parity_tapered using `num_particles`.
    SparsePauliOp values are used as-is.

    The `pop` argument is kept for positional compatibility with the notebook
    call site; it is not consumed internally — psi_ed is used directly (mirrors
    the analogous unused-pop convention in check_state_overlap).
    """
    bound = vqe.circuit.assign_parameters(vqe.params)
    psi_vqe = Statevector(bound).data

    qubit_obs: dict[str, SparsePauliOp] = {}
    for obs_name, obs_op in obs_dict.items():
        if isinstance(obs_op, FermionicOp):
            qubit_obs[obs_name] = _to_qubit_op(
                obs_op, scheme="parity_tapered", num_particles=num_particles
            )
        else:
            qubit_obs[obs_name] = obs_op

    per_obs: dict[str, _ObservableValues] = {}
    all_passed = True
    max_rel_err = 0.0
    for obs_name, obs_op in qubit_obs.items():
        o_mat = obs_op.to_matrix()
        vqe_val = float((psi_vqe.conj() @ o_mat @ psi_vqe).real)
        ed_val = float((psi_ed.conj() @ o_mat @ psi_ed).real)
        abs_diff = abs(vqe_val - ed_val)
        rel_err = abs_diff / max(abs(ed_val), abs_tol)
        passed_i = bool(rel_err < rel_tol or abs_diff < abs_tol)
        if not passed_i:
            all_passed = False
        if rel_err > max_rel_err:
            max_rel_err = rel_err
        per_obs[obs_name] = _ObservableValues(vqe=vqe_val, ed=ed_val, rel_error=rel_err)

    return MultiObservableReport(
        passed=all_passed,
        max_rel_error=max_rel_err,
        values=per_obs,
    )


def check_multistart_spread(
    results: list[VQEResult] | MultistartResult,
    max_spread_hartree: float = 5e-3,
    tol_mha: float | None = None,
) -> MultiStartReport:
    """Layer 4: best / median / max-min spread across seeded starts.

    Parameters
    ----------
    results : list[VQEResult] (original API) or MultistartResult (notebook API).
    max_spread_hartree : spread threshold in Hartree / eV (original kwarg).
    tol_mha : threshold in milli-Hartree (notebook API). If given, overrides
        max_spread_hartree: tol_mha mHa → tol_mha * 1e-3 internal units.
        The units match whatever units your Hamiltonian energies are in.
    """
    vqe_list = results.runs if isinstance(results, MultistartResult) else results

    tol = (tol_mha * 1e-3) if tol_mha is not None else max_spread_hartree

    energies = np.array([r.energy for r in vqe_list])
    spread = float(energies.max() - energies.min())
    return MultiStartReport(
        passed=bool(spread < tol),
        n_starts=len(vqe_list),
        best=float(energies.min()),
        median=float(np.median(energies)),
        spread=spread,
        max_spread_hartree=tol,
    )


def plot_convergence(
    vqe: VQEResult, ed_energy: float, ax: Axes | None = None
) -> Axes:
    """Plot per-iteration energy + running best, with ED reference line."""
    if ax is None:
        _, ax = plt.subplots()
    iters = [i for i, _ in vqe.history]
    energies = np.array([e for _, e in vqe.history])
    running_min = np.minimum.accumulate(energies)
    ax.plot(iters, energies, alpha=0.4, label="per-eval")
    ax.plot(iters, running_min, lw=2, label="running best")
    ax.axhline(ed_energy, color="k", ls="--", label=f"ED ({ed_energy:.5f})")
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("energy")
    ax.set_title(f"{vqe.optimizer_name} on {vqe.ansatz_name}, {vqe.n_qubits} qubits")
    ax.legend(loc="best")
    return ax


@dataclass(frozen=True)
class ExpressivityReport:
    passed: bool
    max_overlap: float
    threshold: float
    optimal_params: np.ndarray


def check_ansatz_expressivity(
    circuit: QuantumCircuit,
    target_state: np.ndarray,
    threshold: float = 0.99,
    seed: int | None = None,
    maxiter: int = 300,
    n_starts: int = 1,
) -> ExpressivityReport:
    """Layer 3: classically maximize |⟨target | ansatz(θ)⟩|² over θ.

    If even this best-case overlap falls below `threshold`, the ansatz cannot
    represent the target state — the VQE optimizer is not at fault.

    Parameters
    ----------
    n_starts : number of random restarts for the classical optimizer. Each
        restart uses seed + i. The best result (highest overlap) is returned.
        For UCCSD on 2 qubits with 3 parameters, 1 restart is usually enough
        but n_starts=8 confirms the global optimum robustly.
    """
    best_overlap = -1.0
    best_params = np.zeros(circuit.num_parameters)

    rng = np.random.default_rng(seed)

    def neg_overlap(theta: np.ndarray) -> float:
        bound = circuit.assign_parameters(theta)
        psi = Statevector(bound).data
        return -float(np.abs(np.vdot(target_state, psi)) ** 2)

    for _ in range(max(1, n_starts)):
        x0 = rng.normal(0.0, 0.1, size=circuit.num_parameters)
        res = minimize(neg_overlap, x0, method="L-BFGS-B", options={"maxiter": maxiter})
        ov = -float(res.fun)
        if ov > best_overlap:
            best_overlap = ov
            best_params = np.asarray(res.x, dtype=float)

    max_overlap = best_overlap
    return ExpressivityReport(
        passed=bool(max_overlap >= threshold),
        max_overlap=max_overlap,
        threshold=threshold,
        optimal_params=best_params,
    )


# ---------------------------------------------------------------------------
# Task 6: compare_energies bar chart + ResilienceGuardrailReport (layer 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResilienceGuardrailReport:
    """Validation layer 6 (Phase 2 spec §6): resilience-tier monotonicity check.

    Hardware-mitigation pass: |E_hw_L2 - E_ED| <= |E_hw_L1 - E_ED| <= |E_hw_L0 - E_ED|.
    Failure does NOT abort Phase 2 — the notebook reports it honestly.
    """

    passed: bool
    gap_l0: float
    gap_l1: float
    gap_l2: float
    notes: str


def compare_energies(
    results: dict[str, float],
    ed_energy: float,
    uncertainties: dict[str, float] | None = None,
    title: str = "L1 NiO SIAM — energy comparison",
) -> Figure:
    """Bar chart of energies against the ED reference.

    Parameters
    ----------
    results : ordered mapping of label -> energy. Plot uses dict insertion order.
    ed_energy : reference energy drawn as a horizontal dashed line.
    uncertainties : optional 1-sigma uncertainties keyed by the same labels as `results`.
        Missing keys default to 0.0 (no error bar).
    title : figure title.

    Returns
    -------
    matplotlib Figure. The caller is responsible for `fig.savefig(...)` or display.
    """
    labels = list(results.keys())
    energies = [results[k] for k in labels]
    if uncertainties is not None:
        errs = [uncertainties.get(k, 0.0) for k in labels]
    else:
        errs = [0.0] * len(labels)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x, energies, yerr=errs, capsize=4, color="#3a6df0")
    ax.axhline(
        ed_energy,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"ED = {ed_energy:.4f} eV",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def check_resilience_guardrail(
    e_ed: float, e_hw_l0: float, e_hw_l1: float, e_hw_l2: float
) -> ResilienceGuardrailReport:
    """Validation layer 6: check that hardware mitigation monotonically reduces the
    gap to ED as resilience level increases (0 -> 1 -> 2).

    Passes when |E_L2 - ED| <= |E_L1 - ED| <= |E_L0 - ED|.
    Fails (and notes which step regressed) when any inequality is reversed by
    more than 1e-12 (the small slack absorbs floating-point ties).
    """
    gap_l0 = abs(e_hw_l0 - e_ed)
    gap_l1 = abs(e_hw_l1 - e_ed)
    gap_l2 = abs(e_hw_l2 - e_ed)

    notes = []
    if gap_l1 > gap_l0 + 1e-12:
        notes.append("L1 worse than L0")
    if gap_l2 > gap_l1 + 1e-12:
        notes.append("L2 worse than L1")
    passed = len(notes) == 0
    return ResilienceGuardrailReport(
        passed=passed,
        gap_l0=gap_l0,
        gap_l1=gap_l1,
        gap_l2=gap_l2,
        notes="; ".join(notes) if notes else "monotone improvement L0 -> L1 -> L2",
    )
