"""Tests for siam_vqe.hardware — IBM Quantum Runtime wrappers (mocked)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
from qiskit import QuantumCircuit, transpile as _real_transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

import siam_vqe.hardware as _hw
from siam_vqe.hardware import (
    make_runtime_estimator,
    pick_backend,
    transpile_for_backend,
)


def test_pick_backend_calls_least_busy_with_expected_filters() -> None:
    service = MagicMock()
    fake = FakeMarrakesh()
    service.least_busy.return_value = fake
    result = pick_backend(service, min_qubits=2)
    service.least_busy.assert_called_once_with(
        operational=True, simulator=False, min_num_qubits=2
    )
    assert result is fake


def test_transpile_for_backend_returns_isa_circuit_and_observable() -> None:
    """Build a trivial 2-qubit circuit + observable, transpile against FakeMarrakesh,
    and verify (a) the circuit's num_qubits matches the backend's, (b) the observable's
    num_qubits matches the transpiled circuit's."""
    fake = FakeMarrakesh()
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5)])
    isa_circ, isa_obs = transpile_for_backend(circ, obs, fake, optimization_level=1)
    assert isa_circ.num_qubits == fake.num_qubits
    assert isa_obs.num_qubits == fake.num_qubits


def test_transpile_for_backend_default_translation_method() -> None:
    """Default translation_method must remain 'translator' (Phase 2 behaviour)."""
    sig = inspect.signature(transpile_for_backend)
    assert "translation_method" in sig.parameters
    assert sig.parameters["translation_method"].default == "translator"


def test_transpile_for_backend_threads_translation_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify translation_method is actually passed to qiskit.transpile, not
    silently dropped or hardcoded to 'translator'.

    The previous override test only re-passed the default value, so it could
    not detect a regression where the body ignored the kwarg. This spy patches
    the `transpile` symbol bound on siam_vqe.hardware and asserts the kwarg
    reaches qiskit.transpile with the caller-supplied value. We use
    ``'synthesis'`` for the override path — a non-default qiskit translation
    method — so the assertion would fail if the body silently hardcoded
    ``'translator'``.
    """
    captured: list[dict] = []

    def spy(circuit, **kwargs):
        captured.append(dict(kwargs))
        return _real_transpile(circuit, **kwargs)

    monkeypatch.setattr(_hw, "transpile", spy)

    fake = FakeMarrakesh()
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])

    # Default path — no translation_method passed; should default to 'translator'.
    transpile_for_backend(circ, obs, fake, optimization_level=1)  # opt=1 for speed
    assert captured[-1].get("translation_method") == "translator", (
        f"Default call should pass translation_method='translator'; got {captured[-1]}"
    )

    # Override path — pass a distinct, valid qiskit translation method.
    # 'synthesis' is documented in qiskit.transpile alongside 'translator'.
    captured.clear()
    transpile_for_backend(
        circ, obs, fake, optimization_level=1, translation_method="synthesis"
    )
    assert captured[-1].get("translation_method") == "synthesis", (
        f"Explicit call should pass translation_method='synthesis'; got {captured[-1]}"
    )


def test_make_runtime_estimator_validates_resilience_level() -> None:
    backend = FakeMarrakesh()
    for rl in (0, 1, 2):
        # Should not raise. We do not assert on the returned object's type
        # because RuntimeEstimatorV2 is a thin wrapper — its behaviour is
        # tested by the IBM-side smoke tests in the notebook.
        est = make_runtime_estimator(backend, resilience_level=rl)
        assert est is not None
    with pytest.raises(ValueError, match="resilience_level must be 0, 1, or 2"):
        make_runtime_estimator(backend, resilience_level=3)
    with pytest.raises(ValueError, match="resilience_level must be 0, 1, or 2"):
        make_runtime_estimator(backend, resilience_level=-1)
