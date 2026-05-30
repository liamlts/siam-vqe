"""IBM Quantum Runtime wrappers for siam_vqe.

Three pure helpers around the runtime API:
    pick_backend             — least-busy operational selection
    transpile_for_backend    — ISA transpile + matching observable layout
    make_runtime_estimator   — EstimatorV2 with validated resilience level

Tested via mocks. CI never submits to a real backend.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.ibm_backend import IBMBackend


def pick_backend(
    service: QiskitRuntimeService,
    min_qubits: int = 2,
) -> IBMBackend:
    """Return the least-busy operational, non-simulator backend with >= min_qubits."""
    return service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=min_qubits,
    )


def transpile_for_backend(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend: IBMBackend,
    optimization_level: int = 3,
    translation_method: str = "translator",
) -> tuple[QuantumCircuit, SparsePauliOp]:
    """Transpile `circuit` to ISA + project `observable` onto the result's layout.

    The returned (isa_circuit, isa_observable) pair has matching qubit counts and
    is ready for submission to EstimatorV2.

    Parameters
    ----------
    circuit : parametrized or bound circuit. If parametrized, bind parameters
        BEFORE calling this function — runtime EstimatorV2 expects ISA-bound
        circuits in Phase 2 (parameter sweep is a Phase 3+ concern).
    observable : the operator whose expectation value will be evaluated.
    backend : target hardware (or fake backend with matching ISA).
    optimization_level : transpiler optimization level (default 3 = full).
    translation_method : qiskit transpile translation method. Default
        ``'translator'`` works on every backend with no optional extras.
        ``'ibm_dynamic_circuits'`` is the plugin real IBM backends advertise
        as preferred, but it requires the ``qiskit-ibm-transpiler`` extra —
        opt in only when that's installed (Phase 3 hardware stretch).
        See ``qiskit.transpile`` docs for the full list of translation
        plugins.
    """
    isa_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        translation_method=translation_method,
    )
    isa_observable = observable.apply_layout(isa_circuit.layout)
    return isa_circuit, isa_observable


def make_runtime_estimator(
    backend: IBMBackend,
    resilience_level: int,
    default_shots: int = 8192,
) -> RuntimeEstimatorV2:
    """Build an IBM Runtime EstimatorV2 at the requested resilience level.

    Resilience-level meaning (qiskit-ibm-runtime >= 0.25):
        0 — no error mitigation
        1 — M3 readout-error mitigation only
        2 — M3 + ZNE (Zero Noise Extrapolation)
    """
    if resilience_level not in {0, 1, 2}:
        raise ValueError(
            f"resilience_level must be 0, 1, or 2; got {resilience_level}"
        )
    estimator = RuntimeEstimatorV2(mode=backend)
    estimator.options.default_shots = default_shots
    estimator.options.resilience_level = resilience_level
    return estimator
