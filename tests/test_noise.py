"""Tests for siam_vqe.noise — Aer-backed noisy Estimator V2."""

from __future__ import annotations

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
