"""Tests for siam_vqe.noise — Aer-backed noisy Estimator V2."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from siam_vqe.noise import make_noisy_estimator


def test_make_noisy_estimator_returns_estimator() -> None:
    backend = FakeMarrakesh()
    est = make_noisy_estimator(backend, shots=2048, seed=42)
    assert isinstance(est, BackendEstimatorV2)


def test_noisy_estimator_evaluates_observable() -> None:
    """A trivial circuit |0> on a noisy Aer estimator should give <Z> ≈ 1.

    The 'noise' is shot noise + readout error; the result should be close to 1
    but not exactly 1.
    """
    backend = FakeMarrakesh()
    est = make_noisy_estimator(backend, shots=8192, seed=20260524)
    circ = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    # Transpile to backend for Aer-from-FakeBackend to apply noise.
    from qiskit import transpile

    isa_circ = transpile(circ, backend=backend, optimization_level=1)
    isa_obs = obs.apply_layout(isa_circ.layout)
    job = est.run([(isa_circ, isa_obs)])
    result = job.result()
    ev = float(result[0].data.evs)
    # 0.7-1.0 is a wide guardrail that catches gross noise-model failures
    # (e.g. depolarizing channel mis-attached) without being flaky.
    assert 0.7 <= ev <= 1.0


def test_noisy_estimator_differs_from_noiseless() -> None:
    """For the same trivial <Z> on |0>, noisy estimator should NOT return
    1.0 exactly; noiseless should be 1.0 to machine precision."""
    backend = FakeMarrakesh()
    noisy = make_noisy_estimator(backend, shots=8192, seed=20260524)
    noiseless = StatevectorEstimator()
    circ = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    from qiskit import transpile

    isa_circ = transpile(circ, backend=backend, optimization_level=1)
    isa_obs = obs.apply_layout(isa_circ.layout)
    ev_noisy = float(noisy.run([(isa_circ, isa_obs)]).result()[0].data.evs)
    ev_clean = float(noiseless.run([(circ, obs)]).result()[0].data.evs)
    assert ev_clean == pytest.approx(1.0, abs=1e-12)
    assert abs(ev_noisy - 1.0) > 0.0  # at least some deviation


def test_run_manual_zne_linear_extrapolation_recovers_noiseless() -> None:
    """At c=1 the folded circuit is the original; extrapolation to 0 with linear
    fit must yield the noiseless value when the noise model is depolarizing."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from siam_vqe.noise import run_manual_zne

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), "cx")
    sim = AerSimulator(noise_model=noise)

    from qiskit.primitives import BackendEstimatorV2
    base_estimator = BackendEstimatorV2(backend=sim)

    extrapolated, raw = run_manual_zne(
        base_estimator, qc, obs,
        noise_factors=(1.0, 3.0, 5.0),
        extrapolator="linear",
        shots=8192,
    )
    # Noiseless ⟨ZZ⟩ on Bell = 1.0. Linear extrap should recover within ~10%.
    assert abs(extrapolated - 1.0) < 0.1
    assert len(raw) == 3
    # raw values should monotonically decrease with c for depolarizing noise
    assert raw[0] > raw[1] > raw[2]


def test_run_manual_zne_validates_factors() -> None:
    """noise_factors must start with 1.0 (the unfolded baseline)."""
    from qiskit import QuantumCircuit
    from qiskit.primitives import BackendEstimatorV2
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator

    from siam_vqe.noise import run_manual_zne

    qc = QuantumCircuit(2)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    base_estimator = BackendEstimatorV2(backend=AerSimulator())

    with pytest.raises(ValueError):
        run_manual_zne(
            base_estimator, qc, obs,
            noise_factors=(3.0, 5.0),  # missing 1.0
            extrapolator="linear",
            shots=1024,
        )


def test_make_mitigated_estimator_no_mit_returns_estimator() -> None:
    """The no_mit spec should produce a working estimator with no mitigation."""
    from qiskit_aer import AerSimulator

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    spec = MitigationSpec(
        name="no_mit", m3=False, zne=False,
        zne_extrapolator=None, zne_noise_factors=None, shots=1024,
    )
    sim = AerSimulator()
    est = make_mitigated_estimator(sim, spec)
    assert est is not None
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    result = est.run([(qc, obs)]).result()
    assert np.isfinite(float(result[0].data.evs))


def test_make_mitigated_estimator_validates_zne_inputs() -> None:
    """A spec with zne=True but zne_extrapolator=None should raise."""
    from qiskit_aer import AerSimulator

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    bad_spec = MitigationSpec(
        name="bad", m3=False, zne=True,
        zne_extrapolator=None, zne_noise_factors=None,
    )
    with pytest.raises(ValueError):
        make_mitigated_estimator(AerSimulator(), bad_spec)


def test_make_mitigated_estimator_zne_only_runs_and_returns_finite() -> None:
    """ZNE-only branch should produce a finite extrapolated value and surface raw values in metadata."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    spec = MitigationSpec(
        name="zne_lin_135", m3=False, zne=True,
        zne_extrapolator="linear", zne_noise_factors=(1.0, 3.0, 5.0),
        shots=2048,
    )
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), "cx")
    sim = AerSimulator(noise_model=noise)
    est = make_mitigated_estimator(sim, spec)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    result = est.run([(qc, obs)]).result()
    assert np.isfinite(float(result[0].data.evs))
    meta = result[0].metadata
    assert meta["path"] == "manual_zne"
    assert "zne_raw_values" in meta
    assert len(meta["zne_raw_values"]) == 3


def test_make_mitigated_estimator_m3_only_runs_and_returns_finite() -> None:
    """M3-only branch should calibrate against the backend and produce a finite expectation."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    spec = MitigationSpec(
        name="m3_only", m3=True, zne=False,
        zne_extrapolator=None, zne_noise_factors=None,
        shots=2048,
    )
    sim = AerSimulator()
    est = make_mitigated_estimator(sim, spec)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 1.0)])  # Bell ⟨ZZ+XX⟩ = 2
    result = est.run([(qc, obs)]).result()
    val = float(result[0].data.evs)
    assert np.isfinite(val)
    # Sanity: on a noiseless AerSimulator, expect val ≈ 2.0 within shot noise (1/sqrt(2048) ≈ 0.022)
    assert abs(val - 2.0) < 0.2
    assert result[0].metadata["path"] == "manual_m3"


def test_fold_circuit_global_c2_on_4cx_circuit_yields_8_two_qubit_gates() -> None:
    """For c=2.0 on a 4-CX circuit, the folded circuit must have 4 + 4 = 8 two-qubit gates
    (round(2.0 * 4) = 8). With 0 global folds and 4 local folds (2 extra gates each)."""
    from siam_vqe.noise import _fold_circuit_global

    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    qc.cx(1, 0)
    folded = _fold_circuit_global(qc, 2.0)
    n_2q = sum(1 for instr in folded.data if instr.operation.num_qubits == 2)
    assert n_2q == 8, f"Expected 8 2Q gates after c=2.0 fold of 4-CX circuit, got {n_2q}"


def test_fold_circuit_global_c4_on_4cx_circuit_yields_16_two_qubit_gates() -> None:
    """For c=4.0 on a 4-CX circuit, expect 16 2Q gates (1 global fold gives 12, then 2 local folds add 4 -> 16)."""
    from siam_vqe.noise import _fold_circuit_global

    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    qc.cx(1, 0)
    folded = _fold_circuit_global(qc, 4.0)
    n_2q = sum(1 for instr in folded.data if instr.operation.num_qubits == 2)
    assert n_2q == 16, f"Expected 16 2Q gates after c=4.0 fold of 4-CX circuit, got {n_2q}"


def test_fold_circuit_global_c5_is_pure_global_fold() -> None:
    """For c=5.0, n_global=2, no local remainder. 4-CX circuit -> 5*4=20 2Q gates.
    n_global=floor(4/2)=2, current_2q=(2*2+1)*4=20=target, additional=0. Pure global fold."""
    from siam_vqe.noise import _fold_circuit_global

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    qc.cx(1, 0)
    folded = _fold_circuit_global(qc, 5.0)
    n_2q = sum(1 for instr in folded.data if instr.operation.num_qubits == 2)
    assert n_2q == 20, f"Expected 20 2Q gates for c=5.0 on 4-CX, got {n_2q}"


def test_fold_circuit_global_rejects_c_below_one() -> None:
    from siam_vqe.noise import _fold_circuit_global

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    with pytest.raises(ValueError, match=r"must be >= 1\.0"):
        _fold_circuit_global(qc, 0.5)


def test_make_mitigated_estimator_m3_zne_runs_and_returns_finite() -> None:
    """The combined m3+zne branch should produce a finite extrapolation with both diagnostics in metadata."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    spec = MitigationSpec(
        name="m3_zne_lin_135", m3=True, zne=True,
        zne_extrapolator="linear", zne_noise_factors=(1.0, 3.0, 5.0),
        shots=2048,
    )
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), "cx")
    sim = AerSimulator(noise_model=noise)
    est = make_mitigated_estimator(sim, spec)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    result = est.run([(qc, obs)]).result()
    val = float(result[0].data.evs)
    assert np.isfinite(val)
    assert abs(val - 1.0) < 0.15, f"m3+zne ⟨ZZ⟩ on Bell with 3% CX depol expected ≈ 1.0, got {val}"
    meta = result[0].metadata
    assert meta["path"] == "manual_m3_plus_zne"
    assert "zne_raw_values" in meta and len(meta["zne_raw_values"]) == 3
    assert "m3_cal_shots" in meta


# ---------------------------------------------------------------------------
# Regression tests for the FakeMarrakesh-transpile path (Task 11 smoke-test bugs)
# ---------------------------------------------------------------------------
#
# When the input circuit has been transpiled against a large fake backend
# (FakeMarrakesh: 156 qubits, basis {sx, rz, cz, x, ...} — NO sxdg), two
# things went wrong in the original Task 7 wrapper:
#   1. M3 calibration ran over range(qc.num_qubits) == range(156), broadcasting
#      the wrong shape from the simulator.
#   2. ZNE global folding called circuit.inverse(), which mapped each sx to
#      sxdg; AerSimulator.from_backend(FakeMarrakesh) rejected sxdg.
# Both fixes belong inside noise.py: read the physical qubit indices from
# isa_circuit.layout, and re-translate folded / basis-rotated circuits back
# into the backend basis.


def _isa_pair(backend, qc):
    """Helper: transpile circuit to backend ISA, return (isa_circ, layout-aware obs builder).
    Returns (isa_circ, isa_obs_factory) where isa_obs_factory(obs) applies the layout."""
    from qiskit import transpile
    isa = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=0)
    return isa


def test_mitigated_estimator_m3_on_transpiled_circuit_does_not_crash() -> None:
    """M3-only on a FakeMarrakesh-ISA circuit must restrict calibration to the
    physical qubits the circuit actually touches — not all 156 backend qubits."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    backend = FakeMarrakesh()
    sim = AerSimulator.from_backend(backend)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 1.0)])  # tests both Z-only and H-rotated bases

    isa = _isa_pair(backend, qc)
    isa_obs = obs.apply_layout(isa.layout)
    assert isa.num_qubits == backend.num_qubits  # confirms the precondition

    spec = MitigationSpec(
        name="m3_only_isa", m3=True, zne=False,
        zne_extrapolator=None, zne_noise_factors=None, shots=2048,
    )
    est = make_mitigated_estimator(sim, spec)
    result = est.run([(isa, isa_obs)]).result()
    val = float(result[0].data.evs)
    assert np.isfinite(val)
    # Bell state ⟨ZZ + XX⟩ = 2 in theory; with FakeMarrakesh's noise expect 1.0-2.0.
    assert 0.5 < val < 2.5, f"Unexpected mitigated value: {val}"
    # Sanity: the wrapper should have remembered only the 2 physical qubits.
    assert est._m3_qubits is not None
    assert len(est._m3_qubits) == 2
    assert set(est._m3_qubits).issubset(set(range(backend.num_qubits)))


def test_mitigated_estimator_zne_only_on_transpiled_circuit_does_not_crash() -> None:
    """ZNE-only on a FakeMarrakesh-ISA circuit must retranslate folded circuits
    back into the backend basis — folding produces sxdg, which AerSimulator
    rejects unless we expand it via BasisTranslator."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    backend = FakeMarrakesh()
    sim = AerSimulator.from_backend(backend)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])

    isa = _isa_pair(backend, qc)
    isa_obs = obs.apply_layout(isa.layout)
    # Precondition: the ISA circuit contains sx, which when folded via inverse
    # yields sxdg — not in FakeMarrakesh's basis.
    isa_ops = {instr.operation.name for instr in isa.data}
    assert "sx" in isa_ops

    spec = MitigationSpec(
        name="zne_lin_135_isa", m3=False, zne=True,
        zne_extrapolator="linear", zne_noise_factors=(1.0, 3.0, 5.0),
        shots=2048,
    )
    est = make_mitigated_estimator(sim, spec)
    result = est.run([(isa, isa_obs)]).result()
    val = float(result[0].data.evs)
    assert np.isfinite(val)


def test_mitigated_estimator_m3_zne_on_transpiled_circuit_does_not_crash() -> None:
    """Combined M3+ZNE on a FakeMarrakesh-ISA circuit must apply both fixes."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    backend = FakeMarrakesh()
    sim = AerSimulator.from_backend(backend)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])

    isa = _isa_pair(backend, qc)
    isa_obs = obs.apply_layout(isa.layout)

    spec = MitigationSpec(
        name="m3_zne_lin_13_isa", m3=True, zne=True,
        zne_extrapolator="linear", zne_noise_factors=(1.0, 3.0),
        shots=1024,
    )
    est = make_mitigated_estimator(sim, spec)
    result = est.run([(isa, isa_obs)]).result()
    val = float(result[0].data.evs)
    assert np.isfinite(val)
    meta = result[0].metadata
    assert meta["path"] == "manual_m3_plus_zne"
    assert "zne_raw_values" in meta and len(meta["zne_raw_values"]) == 2
