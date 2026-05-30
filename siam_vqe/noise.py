"""Noisy Estimator V2 wrappers for siam_vqe.

Owns the simulator-side noise plumbing for Phase 2. Phase 3 will extend this
module with M3 readout error mitigation and ZNE; Phase 2 only ships the
FakeBackend-derived Aer Estimator.

This module does NOT build hand-rolled noise models. It relies on Aer's
``AerSimulator.from_backend()`` which extracts calibration / readout / pulse-
error data from the supplied (fake or real) backend.

API note (Qiskit ≥1.2 / Aer ≥0.15)
------------------------------------
``BackendEstimatorV2`` derives shot count from ``default_precision`` via
shots = ⌈1 / precision²⌉. There is no ``default_shots`` constructor kwarg.
The seed is forwarded to the AerSimulator through the estimator's own
``seed_simulator`` option (the estimator's ``_run_pubs`` explicitly passes it
to the backend's ``run()`` call). Setting ``seed_simulator`` on the AerSimulator
itself is therefore redundant, but harmless — we set it on the estimator side.
"""

from __future__ import annotations

import math

from qiskit.primitives import BackendEstimatorV2
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator


def make_noisy_estimator(
    fake_backend: BackendV2,
    shots: int = 8192,
    seed: int = 20260524,
) -> BackendEstimatorV2:
    """Build a BackendEstimatorV2 backed by AerSimulator.from_backend(fake_backend).

    Parameters
    ----------
    fake_backend:
        Any ``BackendV2`` whose noise model can be extracted by
        ``AerSimulator.from_backend`` (Aer fake backends, real IBM backends, or
        anything implementing the BackendV2 + Target interface with calibration
        data attached).
    shots:
        Target per-circuit shot count.  ``BackendEstimatorV2`` computes the
        actual shot count from ``default_precision`` as
        shots = ⌈1 / precision²⌉, so the realised count may differ from
        *shots* by at most 1.
    seed:
        RNG seed forwarded to the AerSimulator through the estimator's
        ``seed_simulator`` option, ensuring shot-noise reproducibility.

    Returns
    -------
    BackendEstimatorV2
        Configured with a noisy AerSimulator as its backend.
    """
    sim = AerSimulator.from_backend(fake_backend)
    precision = 1.0 / math.sqrt(shots)
    estimator = BackendEstimatorV2(
        backend=sim,
        options={"default_precision": precision, "seed_simulator": seed},
    )
    return estimator
