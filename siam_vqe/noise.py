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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from siam_vqe.mitigation import MitigationSpec


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


def run_manual_zne(
    base_estimator: BackendEstimatorV2,
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    noise_factors: Sequence[float],
    extrapolator: str = "linear",
    shots: int = 8192,
    backend: BackendV2 | AerSimulator | None = None,
) -> tuple[float, list[float]]:
    """Manual digital ZNE via global circuit folding.

    For each noise factor c in {1, 3, 5, ...}, build a folded circuit by
    inserting (c-1)/2 pairs of (U, U†) after the full circuit, measure
    <observable>, then extrapolate the {c -> <H>(c)} pairs to c=0.

    Parameters
    ----------
    base_estimator:
        BackendEstimatorV2 (any backend; typically AerSimulator).
    circuit:
        The bound (parameter-free) circuit to evaluate.
    observable:
        SparsePauliOp matching circuit's qubit count.
    noise_factors:
        Sequence of odd-integer (or 1.0) folding factors. Must start
        with 1.0 (the unfolded baseline).
    extrapolator:
        "linear", "exponential", or "polynomial_degree_3".
    shots:
        Per-config shot count.
    backend:
        If provided, re-translate each folded circuit to the backend's
        basis_gates before evaluating. Required when ``circuit`` is in an
        ISA whose gate inverses fall outside that basis (e.g. ``sx`` on
        IBM hardware, where ``sx.inverse() = sxdg`` is not in the basis).

    Returns
    -------
    tuple[float, list[float]]
        (extrapolated_value, raw_values): the zero-noise extrapolation plus
        the per-noise-factor <observable> values in the order of noise_factors.
    """
    if not (len(noise_factors) >= 2 and noise_factors[0] == 1.0):
        raise ValueError(
            f"noise_factors must start with 1.0 and contain >= 2 values; got {noise_factors}"
        )

    raw_values: list[float] = []
    for c in noise_factors:
        folded = _fold_circuit_global(circuit, c)
        if backend is not None:
            folded = _retranslate_to_basis(folded, backend)
        base_estimator.options.default_precision = 1.0 / np.sqrt(shots)
        result = base_estimator.run([(folded, observable)]).result()
        raw_values.append(float(result[0].data.evs))

    extrapolated = _extrapolate_zne(noise_factors, raw_values, extrapolator)
    return extrapolated, raw_values


def _fold_circuit_global(circuit: QuantumCircuit, c: float) -> QuantumCircuit:
    """Digital ZNE folding (global + local remainder) for noise factor c >= 1.

    Algorithm:
      1. Apply ``n_global = (c-1) // 2`` full global folds (each appends U† U after U).
      2. Compute the remaining number of 2Q gates needed to reach round(c * n_2q)
         effective 2Q gates, and apply that many local folds (G -> G G† G) on the
         *last* 2Q gates of the globally-folded circuit.

    Supports arbitrary c >= 1.0 (odd integers reduce to pure global folding;
    even integers and non-integers use the local-fold remainder).
    """
    if c < 1.0:
        raise ValueError(f"noise factor c must be >= 1.0, got {c}")
    if c == 1.0:
        return circuit.copy()

    n_global = int((c - 1) // 2)
    # Step 1: n_global full global folds
    folded = circuit.copy()
    inv = circuit.inverse()
    for _ in range(n_global):
        folded = folded.compose(inv).compose(circuit)

    n_2q_original = sum(
        1 for instr in circuit.data if instr.operation.num_qubits == 2
    )
    target_2q = round(c * n_2q_original)
    current_2q = (2 * n_global + 1) * n_2q_original
    additional = (target_2q - current_2q) // 2

    if additional == 0:
        return folded

    if n_2q_original == 0:
        raise ValueError(
            f"Cannot apply local folding for c={c}: circuit has no 2-qubit gates"
        )

    indices_2q = [
        i
        for i, instr in enumerate(folded.data)
        if instr.operation.num_qubits == 2
    ]
    if additional > len(indices_2q):
        raise ValueError(
            f"Local-fold count {additional} exceeds available 2Q gates "
            f"{len(indices_2q)} after global folding for c={c}"
        )
    fold_positions = set(indices_2q[-int(additional):])

    result = QuantumCircuit(*folded.qregs, *folded.cregs)
    for i, instr in enumerate(folded.data):
        result.append(instr.operation, instr.qubits, instr.clbits)
        if i in fold_positions:
            result.append(instr.operation.inverse(), instr.qubits, instr.clbits)
            result.append(instr.operation, instr.qubits, instr.clbits)
    return result


def _extrapolate_zne(
    factors: Sequence[float], values: Sequence[float], method: str
) -> float:
    """Extrapolate <observable>(c) to c=0."""
    x = np.array(factors, dtype=float)
    y = np.array(values, dtype=float)
    if method == "linear":
        coeffs = np.polyfit(x, y, 1)
        return float(np.polyval(coeffs, 0.0))
    if method == "polynomial_degree_3":
        deg = min(3, len(x) - 1)
        coeffs = np.polyfit(x, y, deg)
        return float(np.polyval(coeffs, 0.0))
    if method == "exponential":
        from scipy.optimize import curve_fit

        def expdecay(xx: np.ndarray, a: float, b: float, k: float) -> np.ndarray:
            return a + b * np.exp(-k * xx)

        try:
            popt, _ = curve_fit(
                expdecay, x, y, p0=[y[-1], y[0] - y[-1], 0.5], maxfev=5000
            )
            return float(expdecay(np.array(0.0), *popt))
        except (RuntimeError, ValueError):
            # curve_fit raises RuntimeError on convergence failure and ValueError
            # on bad inputs; fall back to linear extrapolation in either case.
            coeffs = np.polyfit(x, y, 1)
            return float(np.polyval(coeffs, 0.0))
    raise ValueError(f"Unknown extrapolator: {method!r}")


# ---------------------------------------------------------------------------
# Task 7 — MitigatedEstimator: manual M3 + ZNE wrapper
# ---------------------------------------------------------------------------

_M3_CAL_SHOTS: int = 4096


@dataclass
class _MitigatedData:
    evs: float
    stds: float


@dataclass
class _MitigatedPubResult:
    data: _MitigatedData
    metadata: dict[str, Any]


@dataclass
class _MitigatedPrimitiveResult:
    _items: list[_MitigatedPubResult] = field(default_factory=list)

    def __getitem__(self, i: int) -> _MitigatedPubResult:
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class _MitigatedJob:
    _result: _MitigatedPrimitiveResult

    def result(self) -> _MitigatedPrimitiveResult:
        return self._result


def _physical_qubits_from_layout(qc: QuantumCircuit) -> list[int]:
    """Physical qubit indices a transpiled circuit's logical qubits map to.

    For a circuit transpiled against a backend, ``qc.num_qubits`` equals the
    backend's full qubit count, but only a subset of those physical qubits
    carries gates from the original logical circuit. This helper returns that
    subset (in logical-qubit order), so M3 calibration and measurement can be
    restricted to the qubits the observable actually touches.

    Fallback: if ``qc.layout`` is None (untranspiled circuit), returns
    ``list(range(qc.num_qubits))`` — every qubit is "physical" in that case.
    """
    if qc.layout is None:
        return list(range(qc.num_qubits))
    return list(qc.layout.final_index_layout())


def _retranslate_to_basis(
    circuit: QuantumCircuit,
    backend: BackendV2 | AerSimulator,
) -> QuantumCircuit:
    """Translate ``circuit``'s gates into ``backend``'s basis, leaving the
    qubit layout untouched.

    Two situations require this:
      * Digital ZNE global folding calls ``circuit.inverse()``, which on an
        IBM-basis ISA circuit (sx, rz, cz) emits ``sxdg`` — not in the basis.
      * Per-Pauli measurement-basis rotations append ``H`` and ``S†`` to the
        ISA circuit, also outside the IBM basis.

    Using ``transpile(..., basis_gates=..., optimization_level=0)`` runs
    ``BasisTranslator`` only — no layout or routing passes — so the
    folded/rotated circuit stays mapped to the same physical qubits.
    """
    target = getattr(backend, "target", None)
    if target is not None and getattr(target, "operation_names", None):
        basis = list(target.operation_names)
    elif hasattr(backend, "configuration"):
        basis = list(backend.configuration().basis_gates)
    else:
        # No way to introspect — return unchanged; caller will surface any
        # downstream "unknown instruction" error with the original gate name.
        return circuit
    return transpile(circuit, basis_gates=basis, optimization_level=0)


def _pauli_basis_groups(
    obs: SparsePauliOp,
) -> list[tuple[str, list[tuple[str, complex]]]]:
    """Group SparsePauliOp terms by measurement basis.

    Returns a list of (basis_label, [(z_string, coeff), ...]) pairs.
    basis_label is a Pauli string with X/Y/Z (no I); it defines which qubits
    need a rotation before Z-basis measurement.  Terms whose basis is all-I
    (i.e. the identity term) are returned as a single group with label "".
    """
    groups: dict[str, list[tuple[str, complex]]] = {}
    for pauli, coeff in zip(obs.paulis, obs.coeffs, strict=True):
        label: str = pauli.to_label()  # e.g. "IXYZ"
        # Derive the measurement-basis label by replacing I -> I (no rotation
        # needed, but we keep the full-length string for qubit indexing).
        # Two terms share a basis iff their X/Y/Z pattern is identical.
        basis = label  # full n-qubit string; used as group key
        if basis not in groups:
            groups[basis] = []
        groups[basis].append((label, complex(coeff)))
    return list(groups.items())


def _build_basis_circuit(
    qc: QuantumCircuit,
    basis_label: str,
    physical_qubits: list[int],
) -> QuantumCircuit:
    """Append measurement-basis rotations and per-qubit measurements on
    ``physical_qubits`` only, into a fresh ``ClassicalRegister`` named "m3_meas".

    basis_label is a Pauli string of length ``qc.num_qubits`` (Qiskit
    big-endian: ``basis_label[n-1-q]`` is the Pauli on physical qubit ``q``).
    For each physical qubit ``q`` in ``physical_qubits``:
      X -> H gate before measurement
      Y -> S†, H before measurement
      Z or I -> no rotation

    Measurements go to classical bits in the order of ``physical_qubits``:
    ``physical_qubits[i]`` is measured into ``c[i]``. The resulting counts
    bitstrings therefore have length ``len(physical_qubits)`` with the
    Qiskit/M3 convention "leftmost bit = highest classical bit = the qubit at
    ``physical_qubits[-1]``".
    """
    n = qc.num_qubits
    meas = qc.copy()
    creg = ClassicalRegister(len(physical_qubits), name="m3_meas")
    meas.add_register(creg)
    for i, q in enumerate(physical_qubits):
        char = basis_label[n - 1 - q]
        if char == "X":
            meas.h(q)
        elif char == "Y":
            meas.sdg(q)
            meas.h(q)
        # Z or I: no rotation needed
        meas.measure(q, creg[i])
    return meas


def _eval_observable_with_m3(
    qc: QuantumCircuit,
    obs: SparsePauliOp,
    backend: BackendV2 | AerSimulator,
    m3: Any,  # mthree.M3Mitigation
    shots: int,
    physical_qubits: list[int],
) -> float:
    """Evaluate ⟨obs⟩ using M3-corrected quasi-distributions on ``physical_qubits``.

    Decomposes ``obs`` into per-basis measurement circuits (each measuring
    only ``physical_qubits``), runs them via BackendSamplerV2 after a basis
    re-translation step, applies M3 correction, and sums per-Pauli
    contributions.

    ``obs`` has ``qc.num_qubits`` qubits; identity-only positions outside
    ``physical_qubits`` are skipped implicitly (they contribute +1 to every
    bitstring's parity, so dropping them is exact).
    """
    if not np.allclose(obs.coeffs.imag, 0, atol=1e-12):
        raise ValueError(
            f"Observable coefficients must be Hermitian (real); "
            f"max imag = {np.max(np.abs(obs.coeffs.imag)):.3e}"
        )
    sampler = BackendSamplerV2(backend=backend)
    groups = _pauli_basis_groups(obs)
    total: float = 0.0
    n = qc.num_qubits

    for basis_label, terms in groups:
        # Pure-identity (across the full ISA width) terms contribute trivially.
        if all(c == "I" for c in basis_label):
            for _z_string, coeff in terms:
                total += coeff.real
            continue

        meas_circuit = _build_basis_circuit(qc, basis_label, physical_qubits)
        meas_circuit = _retranslate_to_basis(meas_circuit, backend)
        result = sampler.run([(meas_circuit,)], shots=shots).result()
        counts = result[0].data.m3_meas.get_counts()
        qd = m3.apply_correction(counts, qubits=physical_qubits)

        for z_string, coeff in terms:
            # Build the mthree expval string (length n_meas, M3/Qiskit
            # big-endian: leftmost char = qubits[-1]). For each physical qubit
            # q at position i in ``physical_qubits``, the Pauli on q lives at
            # z_string[n-1-q]; we replace X/Y with Z because the basis
            # rotation already mapped them into the Z basis.
            chars = [
                z_string[n - 1 - q].replace("X", "Z").replace("Y", "Z")
                for q in reversed(physical_qubits)
            ]
            mthree_string = "".join(chars)
            ev: float = float(qd.expval(mthree_string))
            total += coeff.real * ev

    return total


class MitigatedEstimator:
    """Manual M3 + ZNE wrapper around BackendEstimatorV2 / BackendSamplerV2.

    Quacks like a minimal EstimatorV2: exposes .run(pubs) -> _MitigatedJob
    whose .result() returns a list-indexable _MitigatedPrimitiveResult with
    .data.evs, .data.stds, and .metadata on each pub result.

    Construct via make_mitigated_estimator() — do not instantiate directly.
    """

    def __init__(
        self,
        backend: BackendV2 | AerSimulator,
        spec: MitigationSpec,
    ) -> None:
        self._backend = backend
        self._spec = spec
        self._m3: Any = None  # lazy; set on first run() if spec.m3
        self._m3_qubits: list[int] | None = None  # qubit list at calibration time

    def _calibrate_m3(self, physical_qubits: list[int]) -> None:
        """Calibrate M3 on the first .run() call (lazy, cached).

        ``physical_qubits`` is the subset of the backend's qubits the logical
        circuit maps to (derived from ``isa_circuit.layout.final_index_layout()``).
        Restricting calibration to just these qubits — rather than the
        backend's full width — is required: M3's full-system calibration on
        large backends like FakeMarrakesh (156 qubits) fails the sampler-side
        result aggregation with a shape mismatch.
        """
        import mthree  # lazy import — callers not needing M3 pay no cost

        m3 = mthree.M3Mitigation(system=self._backend)
        try:
            m3.cals_from_system(
                qubits=physical_qubits,
                shots=_M3_CAL_SHOTS,
                async_cal=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"mthree calibration failed for qubits {physical_qubits}: {exc}"
            ) from exc
        self._m3 = m3
        self._m3_qubits = list(physical_qubits)

    def run(
        self,
        pubs: Sequence[tuple[QuantumCircuit, SparsePauliOp] | tuple[QuantumCircuit, SparsePauliOp, Any]],
    ) -> _MitigatedJob:
        """Evaluate all pubs and return a resolved _MitigatedJob."""
        spec = self._spec
        results: list[_MitigatedPubResult] = []

        for pub in pubs:
            if len(pub) == 3:
                raise NotImplementedError(
                    "Parameterized PUBs (qc, obs, params) are not yet supported "
                    "by MitigatedEstimator. Bind parameters before calling .run()."
                )
            qc, obs = pub[0], pub[1]
            physical_qubits = _physical_qubits_from_layout(qc)

            if spec.m3:
                if self._m3 is None:
                    self._calibrate_m3(physical_qubits)
                elif self._m3_qubits is not None and self._m3_qubits != physical_qubits:
                    raise RuntimeError(
                        f"MitigatedEstimator was calibrated for qubits {self._m3_qubits} "
                        f"but received a pub on qubits {physical_qubits}. Construct a new "
                        f"estimator per circuit layout."
                    )

            if not spec.m3 and not spec.zne:
                # (F, F) — no_mit: thin shim over BackendEstimatorV2
                precision = 1.0 / math.sqrt(spec.shots)
                est = BackendEstimatorV2(
                    backend=self._backend,
                    options={"default_precision": precision},
                )
                raw_result = est.run([(qc, obs)]).result()
                pub_result = _MitigatedPubResult(
                    data=_MitigatedData(
                        evs=float(raw_result[0].data.evs),
                        stds=float(raw_result[0].data.stds),
                    ),
                    metadata={"spec": spec.name, "path": "no_mit"},
                )

            elif not spec.m3 and spec.zne:
                # (F, T) — zne only
                assert spec.zne_noise_factors is not None
                assert spec.zne_extrapolator is not None
                base_est = BackendEstimatorV2(backend=self._backend)
                extrapolated, raw_values = run_manual_zne(
                    base_est, qc, obs,
                    spec.zne_noise_factors,
                    spec.zne_extrapolator,
                    spec.shots,
                    backend=self._backend,
                )
                stds = float(np.std(raw_values) / math.sqrt(len(raw_values)))
                pub_result = _MitigatedPubResult(
                    data=_MitigatedData(evs=extrapolated, stds=stds),
                    metadata={
                        "spec": spec.name,
                        "path": "manual_zne",
                        "zne_raw_values": raw_values,
                        "zne_noise_factors": list(spec.zne_noise_factors),
                        "zne_extrapolator": spec.zne_extrapolator,
                    },
                )

            elif spec.m3 and not spec.zne:
                # (T, F) — m3 only
                evs = _eval_observable_with_m3(
                    qc, obs, self._backend, self._m3, spec.shots, physical_qubits
                )
                # Proxy std: shot-noise floor = sqrt(sum(coeff^2)) / sqrt(shots)
                coeff_norm = float(np.sqrt(np.sum(np.abs(obs.coeffs) ** 2)))
                stds = coeff_norm / math.sqrt(spec.shots)
                pub_result = _MitigatedPubResult(
                    data=_MitigatedData(evs=evs, stds=stds),
                    metadata={
                        "spec": spec.name,
                        "path": "manual_m3",
                        "m3_cal_shots": _M3_CAL_SHOTS,
                    },
                )

            else:
                # (T, T) — m3 + zne
                assert spec.zne_noise_factors is not None
                assert spec.zne_extrapolator is not None
                raw_values = []
                for c in spec.zne_noise_factors:
                    folded = _fold_circuit_global(qc, c)
                    folded = _retranslate_to_basis(folded, self._backend)
                    val = _eval_observable_with_m3(
                        folded, obs, self._backend, self._m3, spec.shots, physical_qubits
                    )
                    raw_values.append(val)
                extrapolated = _extrapolate_zne(
                    spec.zne_noise_factors, raw_values, spec.zne_extrapolator
                )
                stds = float(np.std(raw_values) / math.sqrt(len(raw_values)))
                pub_result = _MitigatedPubResult(
                    data=_MitigatedData(evs=extrapolated, stds=stds),
                    metadata={
                        "spec": spec.name,
                        "path": "manual_m3_plus_zne",
                        "zne_raw_values": raw_values,
                        "zne_noise_factors": list(spec.zne_noise_factors),
                        "zne_extrapolator": spec.zne_extrapolator,
                        "m3_cal_shots": _M3_CAL_SHOTS,
                    },
                )

            results.append(pub_result)

        return _MitigatedJob(_result=_MitigatedPrimitiveResult(_items=results))


def make_mitigated_estimator(
    backend_or_sim: BackendV2 | AerSimulator,
    spec: MitigationSpec,
) -> MitigatedEstimator:
    """Construct a MitigatedEstimator for the given backend and mitigation spec.

    Parameters
    ----------
    backend_or_sim:
        AerSimulator (local) or IBMBackend (hardware).
    spec:
        MitigationSpec from siam_vqe.mitigation; controls which mitigation
        branch is taken (none / ZNE / M3 / M3+ZNE).

    Returns
    -------
    MitigatedEstimator
        Exposes ``.run(pubs) -> _MitigatedJob``.  ``.result()`` returns a
        list-indexable result with ``.data.evs``, ``.data.stds``,
        ``.metadata`` per pub.

    Raises
    ------
    ValueError
        If ``spec.zne is True`` but ``spec.zne_extrapolator`` or
        ``spec.zne_noise_factors`` is ``None``.
    """
    if spec.zne and (spec.zne_extrapolator is None or spec.zne_noise_factors is None):
        raise ValueError(
            f"MitigationSpec '{spec.name}' has zne=True but zne_extrapolator="
            f"{spec.zne_extrapolator!r} and zne_noise_factors={spec.zne_noise_factors!r}. "
            "Both must be set when zne=True."
        )
    return MitigatedEstimator(backend=backend_or_sim, spec=spec)
