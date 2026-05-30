"""VQE for single-impurity Anderson models on Qiskit."""

from siam_vqe.analysis import (
    EnergyMatchReport,
    ExpressivityReport,
    MultiObservableReport,
    MultiStartReport,
    ObservableReport,
    OverlapReport,
    ResilienceGuardrailReport,
    check_ansatz_expressivity,
    check_energy_match,
    check_multistart_spread,
    check_observable_agreement,
    check_observable_agreement_multi,
    check_resilience_guardrail,
    check_state_overlap,
    compare_energies,
    plot_convergence,
)
from siam_vqe.ansatz import efficient_su2_ansatz, uccsd_ansatz
from siam_vqe.hamiltonian import hubbard_dimer, nio_l1_anderson, observables_dimer, observables_l1
from siam_vqe.hardware import make_runtime_estimator, pick_backend, transpile_for_backend
from siam_vqe.mappings import to_qubit_op
from siam_vqe.noise import make_noisy_estimator
from siam_vqe.reference_ed import EDResult, exact_diag
from siam_vqe.reference_edrixs import compute_l1_levels
from siam_vqe.vqe_runner import MultistartResult, VQEResult, run_vqe, run_vqe_multistart

__version__ = "0.1.0"

__all__ = [
    "EDResult",
    "EnergyMatchReport",
    "ExpressivityReport",
    "MultiObservableReport",
    "MultiStartReport",
    "MultistartResult",
    "ObservableReport",
    "OverlapReport",
    "ResilienceGuardrailReport",
    "VQEResult",
    "__version__",
    "check_ansatz_expressivity",
    "check_energy_match",
    "check_multistart_spread",
    "check_observable_agreement",
    "check_observable_agreement_multi",
    "check_resilience_guardrail",
    "check_state_overlap",
    "compare_energies",
    "compute_l1_levels",
    "efficient_su2_ansatz",
    "exact_diag",
    "hubbard_dimer",
    "make_noisy_estimator",
    "make_runtime_estimator",
    "nio_l1_anderson",
    "observables_dimer",
    "observables_l1",
    "pick_backend",
    "plot_convergence",
    "run_vqe",
    "run_vqe_multistart",
    "to_qubit_op",
    "transpile_for_backend",
    "uccsd_ansatz",
]
