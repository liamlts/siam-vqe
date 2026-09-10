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


def test_sample_counts_named_creg() -> None:
    """_sample_counts returns counts keyed by a named classical register."""
    from qiskit import ClassicalRegister, transpile

    from siam_vqe.noise import _sample_counts

    backend = FakeMarrakesh()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    creg = ClassicalRegister(2, name="m3_meas")
    qc.add_register(creg)
    qc.measure(0, creg[0])
    qc.measure(1, creg[1])
    isa = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=0)

    counts = _sample_counts(backend, isa, shots=512, creg_name="m3_meas")
    assert sum(counts.values()) == 512
    assert (counts.get("00", 0) + counts.get("11", 0)) > 300


def test_sample_counts_default_creg() -> None:
    """_sample_counts infers the sole creg name when creg_name is None."""
    from qiskit import transpile

    from siam_vqe.noise import _sample_counts

    backend = FakeMarrakesh()
    qc = QuantumCircuit(2, 1)  # default creg "c"
    qc.h(0)
    qc.measure(0, 0)
    isa = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=0)

    counts = _sample_counts(backend, isa, shots=256)
    assert sum(counts.values()) == 256
    assert set(counts.keys()).issubset({"0", "1"})


def test_expval_from_counts_zz_bell() -> None:
    """Raw counts → ⟨ZZ⟩ ≈ 1 on a noiseless Bell state."""
    from qiskit_aer import AerSimulator

    from siam_vqe.noise import (
        _build_basis_circuit,
        _expval_from_counts,
        _sample_counts,
    )

    n = 2
    qc = QuantumCircuit(n)
    qc.h(0)
    qc.cx(0, 1)
    physical_qubits = [0, 1]
    basis_label = "ZZ"
    meas = _build_basis_circuit(qc, basis_label, physical_qubits)
    counts = _sample_counts(AerSimulator(), meas, shots=4096, creg_name="m3_meas")
    ev = _expval_from_counts(counts, z_string="ZZ", physical_qubits=physical_qubits, n=n)
    assert ev == pytest.approx(1.0, abs=0.05)


def test_eval_observable_from_counts_raw_matches_statevector() -> None:
    """m3=None path: ⟨ZZ⟩ on noiseless AerSimulator ≈ statevector value."""
    from qiskit_aer import AerSimulator

    from siam_vqe.noise import _eval_observable_from_counts

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    ev = _eval_observable_from_counts(
        qc, obs, AerSimulator(), m3=None, shots=4096, physical_qubits=[0, 1]
    )
    assert ev == pytest.approx(1.0, abs=0.05)


def test_eval_observable_from_counts_identity_term() -> None:
    """m3=None path: an identity observable returns its coefficient exactly."""
    from qiskit_aer import AerSimulator

    from siam_vqe.noise import _eval_observable_from_counts

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("II", 2.5)])
    ev = _eval_observable_from_counts(
        qc, obs, AerSimulator(), m3=None, shots=512, physical_qubits=[0, 1]
    )
    assert ev == pytest.approx(2.5, abs=1e-9)


def test_mitigated_estimator_no_mit_counts_based() -> None:
    """no-mit branch returns a finite counts-based ⟨ZZ⟩ ≈ 1 on noiseless Aer."""
    from qiskit_aer import AerSimulator

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    spec = MitigationSpec(name="no_mit", m3=False, zne=False, zne_extrapolator=None, zne_noise_factors=None, shots=4096)
    est = make_mitigated_estimator(AerSimulator(), spec)
    res = est.run([(qc, obs)]).result()
    assert float(res[0].data.evs) == pytest.approx(1.0, abs=0.05)
    assert res[0].metadata["path"] == "no_mit"
    import math
    assert res[0].data.stds == pytest.approx(1.0 / math.sqrt(4096), rel=0.01)


def test_m3_backend_kwarg_stored() -> None:
    """m3_backend is stored as _m3_backend; defaults to mode when omitted."""
    from qiskit_aer import AerSimulator

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import MitigatedEstimator

    sim_a = AerSimulator()
    sim_b = AerSimulator()
    spec = MitigationSpec(name="no_mit", m3=False, zne=False, zne_extrapolator=None, zne_noise_factors=None, shots=128)

    est_default = MitigatedEstimator(mode=sim_a, spec=spec)
    assert est_default._m3_backend is sim_a

    est_explicit = MitigatedEstimator(mode=sim_a, spec=spec, m3_backend=sim_b)
    assert est_explicit._m3_backend is sim_b


def test_assert_isa_connectivity_passes_and_fails() -> None:
    """A coupled CZ passes; an off-coupling CZ raises RuntimeError naming the pair."""
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from siam_vqe.noise import _assert_isa_connectivity

    backend = FakeMarrakesh()
    edges = list(backend.coupling_map.get_edges())
    good = edges[0]  # a coupled physical pair

    qc_ok = QuantumCircuit(backend.num_qubits)
    qc_ok.cz(good[0], good[1])
    _assert_isa_connectivity(qc_ok, backend)  # must not raise

    # Find a non-edge pair (search a few candidates; the device is sparse).
    all_pairs = {tuple(e) for e in edges}
    bad = None
    for a in range(min(20, backend.num_qubits)):
        for b in range(a + 1, min(20, backend.num_qubits)):
            if (a, b) not in all_pairs and (b, a) not in all_pairs:
                bad = (a, b)
                break
        if bad:
            break
    assert bad is not None
    qc_bad = QuantumCircuit(backend.num_qubits)
    qc_bad.cz(bad[0], bad[1])
    with pytest.raises(RuntimeError, match="coupling map"):
        _assert_isa_connectivity(qc_bad, backend)


def test_assert_isa_connectivity_noop_on_bare_aer() -> None:
    """Bare AerSimulator has no coupling map → helper is a no-op (returns None)."""
    from qiskit_aer import AerSimulator

    from siam_vqe.noise import _assert_isa_connectivity

    qc = QuantumCircuit(3)
    qc.cz(0, 2)
    assert _assert_isa_connectivity(qc, AerSimulator()) is None


def _isa_qc_for_fold():
    """A small circuit transpiled to FakeMarrakesh ISA (layout + basis + routing)."""
    from qiskit import transpile
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    backend = FakeMarrakesh()
    logical = QuantumCircuit(2)
    logical.h(0)
    logical.cx(0, 1)
    isa = transpile(logical, backend=backend, optimization_level=1,
                    seed_transpiler=1234, translation_method="translator")
    return isa, backend


def test_fold_isa_factor_1_is_identity() -> None:
    from siam_vqe.noise import _fold_circuit_global_isa

    isa, backend = _isa_qc_for_fold()
    folded = _fold_circuit_global_isa(isa, 1, backend)
    # factor=1 → no extra ops beyond a possible basis re-translation; op count
    # must not exceed the input (no folding applied).
    assert len(folded.data) <= len(isa.data) + 0 or len(folded.data) == len(isa.data)


def test_fold_isa_factor_3_scales_two_qubit_count() -> None:
    from siam_vqe.noise import _fold_circuit_global_isa

    isa, backend = _isa_qc_for_fold()
    n_cz_isa = sum(1 for i in isa.data if i.operation.num_qubits == 2
                   and i.operation.name != "barrier")
    folded = _fold_circuit_global_isa(isa, 3, backend)
    n_cz_folded = sum(1 for i in folded.data if i.operation.num_qubits == 2
                      and i.operation.name != "barrier")
    # factor 3 = U U† U → 3× the 2-qubit gates.
    assert n_cz_folded == 3 * n_cz_isa


def test_fold_isa_even_factor_raises() -> None:
    from siam_vqe.noise import _fold_circuit_global_isa

    isa, backend = _isa_qc_for_fold()
    with pytest.raises(ValueError, match="odd"):
        _fold_circuit_global_isa(isa, 2, backend)


def test_fold_isa_stays_connectivity_valid() -> None:
    from siam_vqe.noise import _assert_isa_connectivity, _fold_circuit_global_isa

    isa, backend = _isa_qc_for_fold()
    folded = _fold_circuit_global_isa(isa, 5, backend)
    # Must not raise — folding preserves the routed connectivity.
    _assert_isa_connectivity(folded, backend)


def test_isa_backend_fallback_chain() -> None:
    """_isa_backend defaults to m3_backend, which defaults to mode."""
    from qiskit_aer import AerSimulator

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import MitigatedEstimator

    mode = AerSimulator()
    m3b = AerSimulator()
    isab = AerSimulator()
    spec = MitigationSpec(name="no_mit", m3=False, zne=False,
                          zne_extrapolator=None, zne_noise_factors=None, shots=128)

    assert MitigatedEstimator(mode=mode, spec=spec)._isa_backend is mode
    assert MitigatedEstimator(mode=mode, spec=spec, m3_backend=m3b)._isa_backend is m3b
    assert MitigatedEstimator(mode=mode, spec=spec, m3_backend=m3b,
                              isa_backend=isab)._isa_backend is isab


def test_run_isa_transpiles_logical_input() -> None:
    """Through a Batch (ISA-enforcing mode), run() ISA-transpiles a layout-None
    input so the Batch accepts it. A finite result with no Batch rejection proves
    the input was ISA-transpiled (the hardware path is keyed off the Batch mode)."""
    from qiskit_ibm_runtime import Batch
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    backend = FakeMarrakesh()
    spec = MitigationSpec(name="no_mit", m3=False, zne=False,
                          zne_extrapolator=None, zne_noise_factors=None, shots=512)

    qc = QuantumCircuit(2)  # logical, layout is None
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    with Batch(backend=backend) as batch:
        est = make_mitigated_estimator(
            mode=batch, spec=spec, m3_backend=backend, isa_backend=backend
        )
        res = est.run([(qc, obs)]).result()
    # |amp| of ZZ on a Bell state is ~1; noisy but finite.
    assert np.isfinite(float(res[0].data.evs))
    assert res[0].metadata["path"] == "no_mit"


def test_zne_branch_isa_finite_on_fake_target() -> None:
    """ZNE-only config through the ISA path (Batch-enforced) yields a finite
    extrapolated value and records the manual_zne path. Odd factors only, since
    the ISA fold engages on the Batch mode."""
    from qiskit_ibm_runtime import Batch
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    backend = FakeMarrakesh()
    spec = MitigationSpec(name="zne_lin_135", m3=False, zne=True,
                          zne_extrapolator="linear",
                          zne_noise_factors=(1.0, 3.0, 5.0), shots=512)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    with Batch(backend=backend) as batch:
        est = make_mitigated_estimator(
            mode=batch, spec=spec, m3_backend=backend, isa_backend=backend
        )
        res = est.run([(qc, obs)]).result()
    assert np.isfinite(float(res[0].data.evs))
    assert res[0].metadata["path"] == "manual_zne"
    assert len(res[0].metadata["zne_raw_values"]) == 3


def test_no_mit_bare_aer_regression() -> None:
    """Bare AerSimulator (no real isa_backend) keeps the original no_mit behaviour."""
    from qiskit_aer import AerSimulator

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    spec = MitigationSpec(name="no_mit", m3=False, zne=False,
                          zne_extrapolator=None, zne_noise_factors=None, shots=4096)
    est = make_mitigated_estimator(AerSimulator(), spec)
    res = est.run([(qc, obs)]).result()
    assert float(res[0].data.evs) == pytest.approx(1.0, abs=0.05)
    assert res[0].metadata["path"] == "no_mit"


def test_driver_select_specs_by_name() -> None:
    """run_l1_xas_hw._select_specs returns MitigationSpecs for the requested names."""
    import importlib.util
    from pathlib import Path

    spec_path = (Path(__file__).resolve().parent.parent
                 / "scripts" / "run_l1_xas_hw.py")
    mod_spec = importlib.util.spec_from_file_location("run_l1_xas_hw", spec_path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)

    specs = mod._select_specs(["no_mit", "zne_lin_135"])
    assert [s.name for s in specs] == ["no_mit", "zne_lin_135"]
    with pytest.raises(SystemExit, match="unknown config"):
        mod._select_specs(["does_not_exist"])


def test_driver_make_mode_cm_per_config_batch() -> None:
    """_make_mode_cm returns a fresh Batch when real_backend is set (per-config
    session isolation) and a nullcontext sim path when it is None."""
    import importlib.util
    from contextlib import nullcontext
    from pathlib import Path

    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import Batch
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    spec_path = (Path(__file__).resolve().parent.parent
                 / "scripts" / "run_l1_xas_hw.py")
    mod_spec = importlib.util.spec_from_file_location("run_l1_xas_hw", spec_path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)

    sim = AerSimulator()
    # Sim path: real_backend None -> nullcontext yielding the sim backend.
    cm_sim = mod._make_mode_cm(sim, None)
    assert isinstance(cm_sim, nullcontext)
    with cm_sim as mode:
        assert mode is sim

    # Hardware/fake path: a fresh Batch each call (distinct objects).
    fake = FakeMarrakesh()
    b1 = mod._make_mode_cm(sim, fake)
    b2 = mod._make_mode_cm(sim, fake)
    assert isinstance(b1, Batch) and isinstance(b2, Batch)
    assert b1 is not b2
    b1.close()
    b2.close()


def test_zne_sim_path_even_factor_preserved() -> None:
    """The bare-Aer SIM path still supports even ZNE factors via the original
    _fold_circuit_global (the ISA odd-only restriction must NOT leak to sim)."""
    from qiskit_aer import AerSimulator

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    spec = MitigationSpec(name="zne_even", m3=False, zne=True,
                          zne_extrapolator="linear",
                          zne_noise_factors=(1.0, 2.0, 3.0), shots=1024)
    est = make_mitigated_estimator(AerSimulator(), spec)  # bare Aer -> sim path
    res = est.run([(qc, obs)]).result()
    assert np.isfinite(float(res[0].data.evs))
    assert res[0].metadata["path"] == "manual_zne"


# ---------------------------------------------------------------------------
# Integration test: FakeMarrakesh Batch — ISA-correctness rehearsal layer
# ---------------------------------------------------------------------------
#
# Batch(FakeMarrakesh()) enforces ISA submission identically to a real IBM
# Batch: the underlying backend validates basis gates, qubit routing, and
# circuit width.  Running all four mitigation configs through this path proves
# that the Tasks 1-4 ISA-transpile + ISA-fold rewiring is correct without
# touching real hardware or requiring credentials.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        ("no_mit", False, False, None, None),
        ("m3_only", True, False, None, None),
        ("zne_lin_135", False, True, "linear", (1.0, 3.0, 5.0)),
        ("m3_zne_poly3_135", True, True, "polynomial_degree_3", (1.0, 3.0, 5.0)),
    ],
)
def test_mitigated_estimator_fakemarrakesh_batch_all_configs(config) -> None:
    """Each of the 4 configs dispatches through a Batch(FakeMarrakesh) and returns
    a finite ⟨Z_anc⟩. Batch(FakeMarrakesh) enforces ISA, so this proves ISA-correctness
    without touching real hardware."""
    from qiskit_ibm_runtime import Batch
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import make_mitigated_estimator

    name, m3, zne, extrap, factors = config
    backend = FakeMarrakesh()
    spec = MitigationSpec(name=name, m3=m3, zne=zne,
                          zne_extrapolator=extrap, zne_noise_factors=factors, shots=256)

    # A 3-qubit logical circuit standing in for a Hadamard-test circuit (ancilla = q0).
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    z_anc = SparsePauliOp.from_list([("IIZ", 1.0)])

    with Batch(backend=backend) as batch:
        est = make_mitigated_estimator(
            mode=batch, spec=spec, m3_backend=backend, isa_backend=backend
        )
        res = est.run([(qc, z_anc)]).result()

    assert np.isfinite(float(res[0].data.evs))
    assert np.isfinite(float(res[0].data.stds))


def test_m3_multiterm_hadamard_batch_layout_pinned() -> None:
    """M3 through compute_hadamard_test_mitigated with a MULTI-TERM operator on a
    Batch must not trip the M3 calibration-qubit guard.

    compute_hadamard_test_mitigated calls MitigatedEstimator.run() once per
    (Pauli term, Re/Im). On the ISA path each distinct Hadamard circuit is
    transpiled; without layout pinning they route to different physical qubits
    and the second run() raises (calibrated-for-X-but-got-Y). Layout pinning keeps
    physical_qubits constant across terms, so the single M3 calibration stays
    valid. The 1-term integration test above cannot catch this.
    """
    from qiskit_ibm_runtime import Batch
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import (
        compute_hadamard_test_mitigated,
        make_mitigated_estimator,
    )

    backend = FakeMarrakesh()
    spec = MitigationSpec(name="m3_zne_poly3_135", m3=True, zne=True,
                          zne_extrapolator="polynomial_degree_3",
                          zne_noise_factors=(1.0, 3.0, 5.0), shots=512)
    qc_left = QuantumCircuit(1)
    qc_left.x(0)
    qc_right = QuantumCircuit(1)  # |0>
    op = SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)])  # 2 distinct terms

    with Batch(backend=backend) as batch:
        est = make_mitigated_estimator(
            mode=batch, spec=spec, m3_backend=backend, isa_backend=backend
        )
        re, im, _e_re, _e_im = compute_hadamard_test_mitigated(
            qc_left=qc_left, qc_right=qc_right, operator=op,
            mitigated_estimator=est,
        )
    assert np.isfinite(re) and np.isfinite(im)


# ---------------------------------------------------------------------------
# Regression tests for _retranslate_to_basis Qiskit-2.0 cleanliness
# ---------------------------------------------------------------------------
#
# _retranslate_to_basis derives its basis from ``backend.target.operation_names``.
# On a bare AerSimulator that list contains non-standard instruction names
# (save_*, set_*, mcx_gray, ...). Passing those through ``transpile(...,
# basis_gates=...)`` triggers a Qiskit 1.3 DeprecationWarning on every call and
# becomes a hard ERROR in Qiskit 2.0. The fix filters the basis to Qiskit's
# allowed gate set; it must NOT introduce layout/routing, since M3 readout
# calibration relies on the circuit staying mapped to the same physical qubits.


def test_retranslate_to_basis_emits_no_deprecation_warning() -> None:
    """Re-translating against a bare AerSimulator must not emit a
    DeprecationWarning — its target carries non-standard names (save_*, mcx_gray)
    that ``basis_gates=`` deprecates in 1.3 and rejects in 2.0."""
    import warnings

    from qiskit_aer import AerSimulator

    from siam_vqe.noise import _retranslate_to_basis

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _retranslate_to_basis(qc, AerSimulator())

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, (
        "expected no DeprecationWarning from _retranslate_to_basis; got: "
        + "; ".join(str(w.message)[:120] for w in deprecations)
    )


def test_retranslate_to_basis_preserves_final_index_layout() -> None:
    """For a circuit already transpiled to FakeMarrakesh, re-translation must
    leave the physical-qubit mapping intact (no routing). This guards M3
    calibration, which calibrates on physical_qubits derived from
    ``final_index_layout()``."""
    from qiskit import transpile

    from siam_vqe.noise import _retranslate_to_basis

    backend = FakeMarrakesh()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    isa = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=0)
    assert isa.layout is not None
    expected_layout = isa.layout.final_index_layout()

    out = _retranslate_to_basis(isa, backend)

    assert out.layout is not None, "re-translation dropped the transpile layout"
    assert out.layout.final_index_layout() == expected_layout


def test_retranslate_to_basis_does_not_reroute_two_qubit_gates() -> None:
    """The underlying routing invariant: re-translation never moves a gate to a
    different physical-qubit pair (no coupling map => no routing pass)."""
    from qiskit import transpile

    from siam_vqe.noise import _retranslate_to_basis

    backend = FakeMarrakesh()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    isa = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=0)

    def two_qubit_pairs(circ: QuantumCircuit) -> list[tuple[int, ...]]:
        return sorted(
            tuple(circ.find_bit(q).index for q in instr.qubits)
            for instr in circ.data
            if instr.operation.num_qubits == 2
        )

    out = _retranslate_to_basis(isa, backend)
    assert two_qubit_pairs(out) == two_qubit_pairs(isa)
