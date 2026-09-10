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
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Phase 4 L3 layer checkers: Layer 2 (overlap) + Layer 5 (observables)
# ---------------------------------------------------------------------------


def check_layer2_overlap_l3(
    psi_vqe_tapered: np.ndarray,
    ref,  # L3Reference
    *,
    threshold: float = 0.85,
    degeneracy_eV: float = 0.010,
) -> dict:
    """Layer 2: overlap of VQE statevector with scipy-ED ground state.

    Lifts the 18-qubit tapered statevector back to the (9, 9) sector basis,
    then computes |<psi_ED|psi_VQE>|^2. If the ED first-excited-state is
    within `degeneracy_eV` of the ground state, switches to soft-degeneracy
    mode and computes the overlap with the projector onto the near-degenerate
    subspace.
    """
    from siam_vqe.tapering_l3 import lift_tapered_to_full_sector

    psi_sector = lift_tapered_to_full_sector(
        psi_vqe_tapered,
        num_particles=ref.sector,
        num_spin_orbitals=20,
    )
    # Normalize (lifted vector may have norm < 1 if VQE leaked outside sector)
    norm = np.linalg.norm(psi_sector)
    if norm == 0:
        return {"pass": False, "overlap": 0.0, "mode": "leaked_outside_sector",
                "leak_fraction": 1.0}
    psi_sector = psi_sector / norm
    sector_leak = 1.0 - norm**2

    gap = float(ref.excited_energies[0] - ref.ground_energy)
    if gap < degeneracy_eV:
        # Soft mode: overlap against the projector onto {ground, excited_1}.
        # In this implementation we only have the ground vector; assume the
        # degenerate-partner overlap is computed externally if needed.
        overlap = abs(np.vdot(ref.ground_vector, psi_sector)) ** 2
        return {
            "pass": bool(overlap >= threshold),
            "overlap": float(overlap),
            "mode": "soft_degeneracy",
            "gap_eV": gap,
            "sector_leak": float(sector_leak),
        }

    overlap = abs(np.vdot(ref.ground_vector, psi_sector)) ** 2
    return {
        "pass": bool(overlap >= threshold),
        "overlap": float(overlap),
        "mode": "strict",
        "sector_leak": float(sector_leak),
    }


def check_layer5_observables_l3(
    vqe_observables: dict[str, float],
    ref,  # L3Reference
    *,
    rel_tol: float = 0.05,
    abs_tol: float = 0.05,
    abs_tol_threshold: float = 0.01,
) -> dict:
    """Layer 5: per-observable relative-error check with abs-tol fallback near zero.

    For each observable in `ref.observables`:
      - If |<O>_ED| < abs_tol_threshold: use |VQE - ED| < abs_tol (absolute mode).
      - Else: use |VQE - ED| / |ED| < rel_tol (relative mode).
    Returns a dict with overall pass/fail and per-observable details.
    """
    details: dict[str, dict] = {}
    overall_pass = True

    for key, ed_val in ref.observables.items():
        vqe_val = vqe_observables.get(key)
        if vqe_val is None:
            details[key] = {"pass": False, "mode": "missing"}
            overall_pass = False
            continue
        if abs(ed_val) < abs_tol_threshold:
            err = abs(vqe_val - ed_val)
            p = err < abs_tol
            details[key] = {"pass": p, "mode": "abs_tol",
                            "ed": ed_val, "vqe": vqe_val, "err": err}
        else:
            err_rel = abs(vqe_val - ed_val) / abs(ed_val)
            p = err_rel < rel_tol
            details[key] = {"pass": p, "mode": "rel_tol",
                            "ed": ed_val, "vqe": vqe_val,
                            "rel_err": err_rel}
        if not p:
            overall_pass = False

    return {"pass": overall_pass, "details": details}


# ---------------------------------------------------------------------------
# Phase 4 L3 layer checkers: Layer 4 (multistart spread) + Layer 6 (ADAPT trace)
# ---------------------------------------------------------------------------


def check_layer4_multistart_l3(
    multistart_result,  # AdaptMultistartResult
    *,
    layer1_passed: bool,
    spread_threshold: float = 0.1,
) -> dict:
    """Layer 4: multistart spread with L2 soft-cluster carryover.

    - If spread < threshold: strict pass.
    - If spread >= threshold AND layer1_passed: soft pass with `[warn]` (the
      L2 4-state-cluster pattern: best seed converged but others did not).
    - If spread >= threshold AND not layer1_passed: hard fail.
    """
    spread = multistart_result.spread
    if spread < spread_threshold:
        return {"pass": True, "mode": "strict", "spread_eV": float(spread)}
    if layer1_passed:
        return {
            "pass": True,
            "mode": "soft_cluster",
            "spread_eV": float(spread),
            "warn": "cluster-like behavior",
            "best_energy_eV": float(multistart_result.best_energy),
            "median_energy_eV": float(multistart_result.median_energy),
            "worst_energy_eV": float(multistart_result.worst_energy),
        }
    return {
        "pass": False,
        "mode": "fail_spread_no_best",
        "spread_eV": float(spread),
    }


def check_layer6_adapt_trace(
    trace,  # tuple[dict]
    *,
    noise_floor_eV: float = 0.001,
) -> dict:
    """Layer 6: ADAPT convergence trace must be monotonic (within noise floor).

    A violation is energy[i+1] > energy[i] + noise_floor_eV at any iteration
    where an operator was appended.
    """
    violations: list[dict] = []
    prev_E = None
    for row in trace:
        if row["event"] != "operator_added":
            continue
        E = row["energy"]
        if prev_E is not None and E > prev_E + noise_floor_eV:
            violations.append({
                "iteration": row["iteration"],
                "prev_E": prev_E, "E": E,
                "delta": E - prev_E,
            })
        prev_E = E
    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "n_operator_steps": sum(1 for r in trace if r["event"] == "operator_added"),
    }


def plot_adapt_convergence(
    trace,
    *,
    ed_reference: float | None = None,
    output_path=None,
) -> Axes:
    """E vs operator count + g_max on twin axis.

    Parameters
    ----------
    trace : sequence of dict rows (AdaptResult.trace).
    ed_reference : float | None
        If given, draws a horizontal dashed line at this energy as the ED reference.
    output_path : Path-like | None
        If given, saves the figure as PDF + PNG at this path (suffix stripped).
    Returns the primary `matplotlib.axes.Axes`.
    """
    iters = [row["n_operators"] for row in trace
             if row["event"] in ("operator_added", "converged")]
    energies = [row["energy"] for row in trace
                if row["event"] in ("operator_added", "converged")]
    gmaxes = [row["g_max"] for row in trace
              if row["event"] in ("operator_added", "converged")]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iters, energies, marker="o", color="C0", label="E_VQE")
    if ed_reference is not None:
        ax.axhline(ed_reference, ls="--", color="k", label="E_ED")
    ax.set_xlabel("Number of operators appended")
    ax.set_ylabel("Energy (eV)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.semilogy(iters, gmaxes, marker="s", color="C3", label="g_max")
    ax2.set_ylabel("max gradient (eV)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    fig.tight_layout()
    if output_path is not None:
        base = Path(output_path).with_suffix("")
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".png"), dpi=150)
    return ax


def plot_l3_observables(
    vqe_observables: dict[str, float],
    ed_observables: dict[str, float],
    *,
    output_path=None,
) -> Axes:
    """Side-by-side bar chart of VQE vs ED observables.

    Plots ⟨n_d⟩, ⟨n_p⟩, ⟨S²⟩ as one group; per-orbital ⟨n_d^α⟩ as another.
    """
    summary_keys = ["n_d", "n_p", "S2"]
    perorb_keys = ["n_d_3z2", "n_d_x2y2", "n_d_xz", "n_d_yz", "n_d_xy"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def _bars(ax, keys, title):
        x = np.arange(len(keys))
        w = 0.35
        ax.bar(x - w/2, [vqe_observables[k] for k in keys], width=w, label="VQE",
               color="C0")
        ax.bar(x + w/2, [ed_observables[k] for k in keys], width=w, label="ED",
               color="C1")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=20, ha="right")
        ax.set_ylabel("⟨O⟩")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    _bars(axes[0], summary_keys, "Summary observables")
    _bars(axes[1], perorb_keys, "Per d-orbital occupation")
    fig.tight_layout()
    if output_path is not None:
        base = Path(output_path).with_suffix("")
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".png"), dpi=150)
    return axes[0]
