"""Noisy quantum-circuit execution and manual error mitigation (M3 readout correction + digital ZNE) for siam_vqe.

The hardware-reachable execution path routes counts through
qiskit_ibm_runtime.SamplerV2 (local Aer/fake backends and real IBM hardware
via the same API); MitigatedEstimator and the Hadamard-test helpers build on
it. The simulator-only helpers make_noisy_estimator and run_manual_zne retain
qiskit.primitives.BackendEstimatorV2 for the completed L2 noise study.

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
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.primitives import BackendEstimatorV2
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

from siam_vqe.mitigation import MitigationSpec

# Gate names ``transpile`` accepts through ``basis_gates`` without raising the
# Qiskit 1.3 "non-standard gates" DeprecationWarning (a hard error in 2.0):
# the standard gate set plus the structural ops the transpiler always allows.
# Mirrors qiskit's own allow-list in
# ``transpiler.preset_passmanagers.generate_preset_pass_manager._parse_basis_gates``.
# A bare AerSimulator's target also advertises save_*/set_*/mcx_gray names, which
# are not real gates and must be filtered out before they reach ``basis_gates=``.
_BASIS_GATES_ALLOWLIST: frozenset[str] = frozenset(
    set(get_standard_gate_name_mapping())
    | {"measure", "delay", "reset"}
    | set(CONTROL_FLOW_OP_NAMES)
)


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
_ISA_TRANSPILE_SEED: int = 1234  # deterministic routing/layout for reproducible ZNE folds


def _sample_counts(
    mode: Any,
    circuit: QuantumCircuit,
    shots: int,
    *,
    creg_name: str | None = None,
) -> dict[str, int]:
    """Run one measured circuit through qiskit_ibm_runtime.SamplerV2 and return counts.

    ``mode`` is anything SamplerV2 accepts: a BackendV2 (local Aer/fake) or an
    open Batch/Session (hardware; also valid locally). Counts are read from the
    result DataBin by classical-register name; if ``creg_name`` is None the
    circuit's sole classical register name is used.
    """
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    if creg_name is None:
        if len(circuit.cregs) != 1:
            raise ValueError(
                f"_sample_counts needs an explicit creg_name when the circuit has "
                f"{len(circuit.cregs)} classical registers (expected exactly 1)."
            )
        creg_name = circuit.cregs[0].name
    sampler = SamplerV2(mode=mode)
    result = sampler.run([(circuit,)], shots=shots).result()
    data_bin = result[0].data
    counts: dict[str, int] = getattr(data_bin, creg_name).get_counts()
    return counts


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


def _unwrap_backendv2(backend: Any) -> Any:
    """Return the underlying BackendV2 from a Batch/Session wrapper, or ``backend``.

    qiskit_ibm_runtime Batch/Session expose the device as ``._backend``; a bare
    BackendV2/AerSimulator has no such attribute and is returned unchanged.
    """
    return getattr(backend, "_backend", backend)


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
    backend: Any,
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
    folded/rotated circuit stays mapped to the same physical qubits. The
    introspected basis is filtered to ``_BASIS_GATES_ALLOWLIST`` first: a bare
    AerSimulator's target lists non-standard names (save_*, set_*, mcx_gray)
    that ``basis_gates=`` deprecates in Qiskit 1.3 and rejects in 2.0. Those
    names never appear in circuits, so dropping them from the allowed output
    basis is behaviour-preserving.

    ``transpile`` discards the input's ``TranspileLayout`` when only basis
    translation runs, so it is copied back onto the result: re-translation does
    not route, hence the original physical-qubit mapping still holds, and M3
    readout calibration downstream reads ``final_index_layout()`` off the pub
    circuit (this preserves it for any caller that inspects the result too).

    When ``backend`` is a Batch or Session (no ``.target``), the underlying
    backend is extracted via ``backend._backend`` for basis introspection.
    """
    # Unwrap Batch/Session to a real BackendV2 for target introspection.
    introspect = getattr(backend, "_backend", backend)
    target = getattr(introspect, "target", None)
    if target is not None and getattr(target, "operation_names", None):
        basis = list(target.operation_names)
    elif hasattr(introspect, "configuration"):
        basis = list(introspect.configuration().basis_gates)
    else:
        # No way to introspect — return unchanged; caller will surface any
        # downstream "unknown instruction" error with the original gate name.
        return circuit
    basis = [g for g in basis if g in _BASIS_GATES_ALLOWLIST]
    translated = transpile(circuit, basis_gates=basis, optimization_level=0)
    if circuit.layout is not None and translated.layout is None:
        translated._layout = circuit._layout
    return translated


def _assert_isa_connectivity(circuit: QuantumCircuit, isa_backend: Any) -> None:
    """Raise RuntimeError if any 2-qubit gate is off ``isa_backend``'s coupling map.

    No-op when the backend exposes no coupling map (e.g. bare AerSimulator) — there
    is nothing to validate against. Barriers (which span multiple qubits but are not
    gates) are skipped by name.
    """
    introspect = getattr(isa_backend, "_backend", isa_backend)
    cmap = getattr(introspect, "coupling_map", None)
    if cmap is None:
        target = getattr(introspect, "target", None)
        cmap = target.build_coupling_map() if target is not None else None
    if cmap is None:
        return
    edges = {tuple(e) for e in cmap.get_edges()}
    for instr in circuit.data:
        op = instr.operation
        if op.num_qubits == 2 and op.name != "barrier":
            q = tuple(circuit.find_bit(b).index for b in instr.qubits)
            if q not in edges and (q[1], q[0]) not in edges:
                name = getattr(introspect, "name", "backend")
                raise RuntimeError(
                    f"2-qubit gate {op.name!r} on physical qubits {q} is off the "
                    f"{name} coupling map (ISA-connectivity violation)."
                )


def _isa_basis_translate(circuit: QuantumCircuit, isa_backend: Any) -> QuantumCircuit:
    """Translate ``circuit`` into ``isa_backend``'s basis (1-qubit only; no routing).

    Sources the basis explicitly from ``isa_backend.target`` rather than the
    Batch-unwrapping heuristics in ``_retranslate_to_basis`` — ``isa_backend`` is
    always a concrete BackendV2 on the hardware path, so ``.target`` is reliably
    present. ``optimization_level=0`` runs BasisTranslator only (no layout/routing),
    so the physical-qubit mapping is preserved.
    """
    introspect = getattr(isa_backend, "_backend", isa_backend)
    target = getattr(introspect, "target", None)
    if target is not None and getattr(target, "operation_names", None):
        basis = list(target.operation_names)
    elif hasattr(introspect, "configuration"):
        basis = list(introspect.configuration().basis_gates)
    else:
        return circuit
    return transpile(circuit, basis_gates=basis, optimization_level=0)


def _fold_circuit_global_isa(
    qc_isa: QuantumCircuit, factor: float, isa_backend: Any
) -> QuantumCircuit:
    """ISA-level global ZNE fold ``U·(U†·U)^n`` for an odd-integer noise factor.

    ``qc_isa`` must already be transpiled to ``isa_backend``'s ISA (layout + basis +
    routing) and contain no measurements. The fold is built on the same physical
    qubits; the inverse of an ISA circuit stays connectivity-valid (cz self-inverse;
    sx→sxdg, rz(θ)→rz(−θ) are 1-qubit), so no re-routing is needed — only a 1-qubit
    basis re-translation of the introduced ``sxdg`` gates. Barriers separate each fold
    segment to defend against any downstream gate cancellation/fusion. Raises if
    ``factor`` is not an odd integer (local folding is out of ISA scope).
    """
    if factor != int(factor) or int(factor) % 2 == 0:
        raise ValueError(
            f"ISA ZNE requires odd-integer noise factors (global folding); got {factor}"
        )
    n = (int(factor) - 1) // 2
    folded = qc_isa.copy()
    if n > 0:
        inv = qc_isa.inverse()
        for _ in range(n):
            folded.barrier()
            folded = folded.compose(inv)
            folded.barrier()
            folded = folded.compose(qc_isa)
    folded = _isa_basis_translate(folded, isa_backend)
    _assert_isa_connectivity(folded, isa_backend)
    return folded


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


def _expval_from_counts(
    counts: dict[str, int],
    z_string: str,
    physical_qubits: list[int],
    n: int,
) -> float:
    """Raw (un-mitigated) parity expectation of a Pauli term from counts.

    ``counts`` keys are bitstrings of length ``len(physical_qubits)`` in
    Qiskit/M3 big-endian order: leftmost char = the qubit at
    ``physical_qubits[-1]``. ``z_string`` is the full n-qubit Pauli label
    (X/Y already rotated into Z by the basis circuit). A physical qubit
    contributes to the parity iff its Pauli is non-identity.
    """
    # Which measured positions (0 = leftmost = physical_qubits[-1]) are active.
    active_meas_positions = [
        i
        for i, q in enumerate(reversed(physical_qubits))
        if z_string[n - 1 - q] != "I"
    ]
    total = sum(counts.values())
    if total == 0:
        return 0.0
    acc = 0.0
    for bitstring, c in counts.items():
        parity = sum(int(bitstring[i]) for i in active_meas_positions) % 2
        acc += c * (1 if parity == 0 else -1)
    return acc / total


def _eval_observable_from_counts(
    qc: QuantumCircuit,
    obs: SparsePauliOp,
    mode: Any,
    m3: Any,  # mthree.M3Mitigation, or None for raw (un-mitigated) expectation
    shots: int,
    physical_qubits: list[int],
    isa_backend: Any = None,
) -> float:
    """Evaluate ⟨obs⟩ from sampled counts on ``physical_qubits``.

    Decomposes ``obs`` into per-basis measurement circuits (each measuring only
    ``physical_qubits``), runs them via ``_sample_counts`` (SamplerV2) after a
    basis re-translation step, and sums per-Pauli contributions. If ``m3`` is
    provided, applies M3 readout correction; if ``m3`` is None, uses the raw
    parity expectation.

    ``obs`` has ``qc.num_qubits`` qubits; identity-only positions outside
    ``physical_qubits`` are skipped implicitly (they contribute +1 to every
    bitstring's parity, so dropping them is exact).

    When ``mode`` has no introspectable basis (e.g. an open Batch on real
    hardware), the caller is responsible for passing circuits already
    transpiled to the backend ISA — ``_retranslate_to_basis`` returns such
    circuits unchanged.
    """
    if not np.allclose(obs.coeffs.imag, 0, atol=1e-12):
        raise ValueError(
            f"Observable coefficients must be Hermitian (real); "
            f"max imag = {np.max(np.abs(obs.coeffs.imag)):.3e}"
        )
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
        if isa_backend is not None:
            meas_circuit = _isa_basis_translate(meas_circuit, isa_backend)
        else:
            meas_circuit = _retranslate_to_basis(meas_circuit, mode)
        counts = _sample_counts(mode, meas_circuit, shots, creg_name="m3_meas")

        if m3 is not None:
            qd = m3.apply_correction(counts, qubits=physical_qubits)

        for z_string, coeff in terms:
            if m3 is not None:
                chars = [
                    z_string[n - 1 - q].replace("X", "Z").replace("Y", "Z")
                    for q in reversed(physical_qubits)
                ]
                ev = float(qd.expval("".join(chars)))
            else:
                ev = _expval_from_counts(counts, z_string, physical_qubits, n)
            total += coeff.real * ev

    return total


class MitigatedEstimator:
    """Manual M3 + ZNE wrapper over qiskit_ibm_runtime.SamplerV2.

    Quacks like a minimal EstimatorV2: exposes .run(pubs) -> _MitigatedJob
    whose .result() returns a list-indexable _MitigatedPrimitiveResult with
    .data.evs, .data.stds, and .metadata on each pub result.

    Construct via make_mitigated_estimator() — do not instantiate directly.
    """

    def __init__(
        self,
        mode: Any,  # BackendV2 | AerSimulator | Batch | Session
        spec: MitigationSpec,
        m3_backend: Any = None,  # underlying BackendV2 for mthree when mode is Batch/Session
        isa_backend: Any = None,  # concrete BackendV2 to ISA-transpile input circuits against
    ) -> None:
        self._mode = mode
        self._spec = spec
        # mthree.M3Mitigation requires a BackendV2, not a Batch/Session.  When
        # mode is an open Batch (hardware or local-fake), pass the underlying
        # backend explicitly via m3_backend.  Falls back to mode when None
        # (backwards-compatible for bare AerSimulator callers).
        self._m3_backend: Any = m3_backend if m3_backend is not None else mode
        # ISA transpile target.  When set, run() full-transpiles each logical input
        # circuit (routing + basis + layout) before capturing physical_qubits and
        # folding.  Fallback chain: isa_backend -> m3_backend -> mode.
        self._isa_backend: Any = isa_backend if isa_backend is not None else self._m3_backend
        self._m3: Any = None  # lazy; set on first run() if spec.m3
        self._m3_qubits: list[int] | None = None  # qubit list at calibration time
        # Physical layout pinned on the first ISA transpile and reused as
        # ``initial_layout`` for every subsequent input, so all per-Pauli-term
        # Hadamard circuits in one sweep map to the SAME physical qubits.  Their
        # 2-qubit entangling structure is identical across terms (only 1-qubit
        # gates differ), so a fixed layout keeps physical_qubits constant and the
        # M3 calibration (done once) valid across the whole operator.
        self._isa_layout: list[int] | None = None

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

        m3 = mthree.M3Mitigation(system=self._m3_backend)
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
            # Hardware path: ISA-transpile a logical (layout-None) input so routing,
            # basis, and layout are fixed before we capture physical_qubits and fold.
            # An already-ISA input (layout set) is left untouched.  Bare-Aer callers
            # (no real target) are unaffected.
            isa_dev = (
                _unwrap_backendv2(self._isa_backend)
                if self._isa_backend is not None
                else None
            )
            # ISA machinery engages only when the EXECUTION target (mode) enforces
            # ISA — a qiskit_ibm_runtime Batch/Session, which exposes the device as
            # ``._backend``.  A permissive AerSimulator (even ``.from_backend``,
            # which carries a coupling_map from the snapshot) accepts non-ISA
            # circuits, so it stays on the original sim path — preserving its
            # even/non-integer ZNE-factor support.  Keying off the transpile target's
            # coupling_map instead would wrongly route a from_backend sim into the
            # odd-only ISA fold (see test_hadamard_with_mitigated_estimator_fakemarrakesh).
            isa_active = getattr(self._mode, "_backend", None) is not None
            if qc.layout is None and isa_active:
                if self._isa_layout is None:
                    # First ISA transpile in this estimator: let the transpiler
                    # choose the layout, then pin it for all later circuits.
                    qc = transpile(
                        qc,
                        backend=isa_dev,
                        optimization_level=1,
                        seed_transpiler=_ISA_TRANSPILE_SEED,
                        translation_method="translator",
                    )
                    self._isa_layout = list(qc.layout.final_index_layout())
                else:
                    # Reuse the pinned layout so physical_qubits is identical to the
                    # first circuit's — keeps M3 calibration valid across all the
                    # per-Pauli-term Hadamard circuits of one operator.
                    qc = transpile(
                        qc,
                        backend=isa_dev,
                        initial_layout=self._isa_layout,
                        optimization_level=1,
                        seed_transpiler=_ISA_TRANSPILE_SEED,
                        translation_method="translator",
                    )
            # After ISA transpile the circuit spans the full device width; widen the
            # observable to match so per-Pauli indexing in _build_basis_circuit is valid.
            if qc.layout is not None and obs.num_qubits != qc.num_qubits:
                obs = obs.apply_layout(qc.layout)
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
                # (F, F) — no_mit: counts-based raw expectation via SamplerV2
                evs = _eval_observable_from_counts(
                    qc, obs, self._mode, None, spec.shots, physical_qubits,
                    isa_backend=isa_dev if isa_active else None,
                )
                coeff_norm = float(np.sqrt(np.sum(np.abs(obs.coeffs) ** 2)))
                stds = coeff_norm / math.sqrt(spec.shots)
                pub_result = _MitigatedPubResult(
                    data=_MitigatedData(evs=evs, stds=stds),
                    metadata={"spec": spec.name, "path": "no_mit"},
                )

            elif not spec.m3 and spec.zne:
                # (F, T) — zne only, counts-based (decoupled from standalone run_manual_zne)
                assert spec.zne_noise_factors is not None
                assert spec.zne_extrapolator is not None
                raw_values = []
                for c in spec.zne_noise_factors:
                    if isa_active:
                        folded = _fold_circuit_global_isa(qc, c, isa_dev)
                    else:
                        folded = _fold_circuit_global(qc, c)
                        folded = _retranslate_to_basis(folded, self._mode)
                    val = _eval_observable_from_counts(
                        folded, obs, self._mode, None, spec.shots, physical_qubits,
                        isa_backend=isa_dev if isa_active else None,
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
                        "path": "manual_zne",
                        "zne_raw_values": raw_values,
                        "zne_noise_factors": list(spec.zne_noise_factors),
                        "zne_extrapolator": spec.zne_extrapolator,
                    },
                )

            elif spec.m3 and not spec.zne:
                # (T, F) — m3 only
                evs = _eval_observable_from_counts(
                    qc, obs, self._mode, self._m3, spec.shots, physical_qubits,
                    isa_backend=isa_dev if isa_active else None,
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
                    if isa_active:
                        folded = _fold_circuit_global_isa(qc, c, isa_dev)
                    else:
                        folded = _fold_circuit_global(qc, c)
                        folded = _retranslate_to_basis(folded, self._mode)
                    val = _eval_observable_from_counts(
                        folded, obs, self._mode, self._m3, spec.shots, physical_qubits,
                        isa_backend=isa_dev if isa_active else None,
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
    mode: Any,
    spec: MitigationSpec,
    m3_backend: Any = None,
    isa_backend: Any = None,
) -> MitigatedEstimator:
    """Construct a MitigatedEstimator for the given backend and mitigation spec.

    Parameters
    ----------
    mode:
        AerSimulator (local), IBMBackend (hardware), or an open
        qiskit_ibm_runtime.Batch / Session. Passed directly to
        ``SamplerV2(mode=mode)`` for all shot-based evaluations.
    spec:
        MitigationSpec from siam_vqe.mitigation; controls which mitigation
        branch is taken (none / ZNE / M3 / M3+ZNE).
    m3_backend:
        Underlying BackendV2 to pass to mthree.M3Mitigation.  Required when
        ``mode`` is a Batch or Session (mthree requires a BackendV2, not a
        Batch).  When None, ``mode`` is used directly (backwards-compatible
        for bare AerSimulator callers).
    isa_backend:
        Concrete BackendV2 to ISA-transpile each input circuit against (routing +
        basis + layout) before folding/measurement.  Required for an opaque Batch on
        real hardware.  Falls back to ``m3_backend`` then ``mode`` when None.

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
    return MitigatedEstimator(
        mode=mode, spec=spec, m3_backend=m3_backend, isa_backend=isa_backend
    )


def _hadamard_circuit_for_pauli(
    qc_left: QuantumCircuit,
    qc_right: QuantumCircuit,
    pauli_label: str,
    imag: bool,
) -> QuantumCircuit:
    """Build a Hadamard-test circuit measuring ⟨0|U_L† · P · U_R|0⟩ on ancilla.

    Real part: H on anc → controlled-V → H on anc → measure anc.
    Imag part: H on anc → controlled-V → S† on anc → H on anc → measure anc.

    Where V = U_L† · P · U_R, applied to the system qubits.
    """
    n_sys = qc_left.num_qubits
    assert qc_right.num_qubits == n_sys, "left and right circuits must have same width"

    qc = QuantumCircuit(n_sys + 1, 1)  # qubit 0 = ancilla
    sys_qubits = list(range(1, n_sys + 1))

    qc.h(0)

    v_subcirc = QuantumCircuit(n_sys)
    v_subcirc.compose(qc_right, qubits=range(n_sys), inplace=True)
    for q_idx, p in enumerate(pauli_label[::-1]):
        if p == "X":
            v_subcirc.x(q_idx)
        elif p == "Y":
            v_subcirc.y(q_idx)
        elif p == "Z":
            v_subcirc.z(q_idx)
    v_subcirc.compose(qc_left.inverse(), qubits=range(n_sys), inplace=True)
    controlled_v = v_subcirc.to_gate().control(1)
    qc.append(controlled_v, [0, *sys_qubits])

    if imag:
        qc.sdg(0)

    qc.h(0)
    qc.measure(0, 0)
    return qc


def compute_hadamard_test(
    *,
    qc_left: QuantumCircuit,
    qc_right: QuantumCircuit,
    operator: SparsePauliOp,
    shots: int = 8192,
    mode: Any = None,
    isa_backend: Any = None,  # if set, full-transpile each Hadamard circuit to this backend's ISA
) -> tuple[float, float, float, float]:
    """Estimate ⟨ψ_left | O | ψ_right⟩ via Hadamard-test ancilla circuits.

    ψ_left = U_L|0⟩, ψ_right = U_R|0⟩ where U_L = qc_left, U_R = qc_right.

    Each Pauli term P_k of `operator` is Hadamard-tested individually
    for both Re and Im, then linearly combined with the (complex) coefficient.

    Returns (Re, Im, stderr_Re, stderr_Im).

    Parameters
    ----------
    mode:
        Anything accepted by ``SamplerV2(mode=...)``: a BackendV2 (local Aer /
        fake backend) or an open Batch/Session (hardware). Defaults to
        ``AerSimulator()`` when None.
    isa_backend:
        When provided, each Hadamard circuit is full-transpiled to this
        backend's ISA via ``qiskit.transpile`` (optimization_level=1) before
        submission.  Required for real IBM backends and Batch/Session contexts
        that enforce ISA (including local Batch(FakeMarrakesh)).  When None,
        falls back to ``_retranslate_to_basis`` (BasisTranslator-only, no
        routing) which works for bare AerSimulator but NOT for Batch/real hw.

    Note: this implementation issues a UserWarning if `operator` is detected
    to be anti-Hermitian (Observation #20 — anti-Hermitian operators in
    a commutator-metric pathway force ⟨[T†,T]⟩ = 0 identically; callers
    using this for qEOM matrices should use non-antisymmetric raising
    operators).
    """
    import warnings

    if mode is None:
        mode = AerSimulator()

    op_dense = operator.to_matrix()
    if np.allclose(op_dense, -op_dense.conj().T, atol=1e-10) and not np.allclose(op_dense, 0):
        warnings.warn(
            "Operator is anti-Hermitian; if used in a commutator-metric "
            "qEOM pathway, ⟨[T†,T]⟩ will be identically zero. See Observation #20.",
            UserWarning,
            stacklevel=2,
        )

    total_re = 0.0
    total_im = 0.0
    var_re = 0.0
    var_im = 0.0

    for pauli, coeff in zip(operator.paulis, operator.coeffs, strict=True):
        label = pauli.to_label()
        c = complex(coeff)

        qc_re = _hadamard_circuit_for_pauli(qc_left, qc_right, label, imag=False)
        # Hardware (Batch/real backend): full-transpile to ISA. Local Aer/fake:
        # BasisTranslator only — Aer accepts the unitary controlled-V gate.
        # translation_method="translator" pins the universal translator: real
        # IBM backends advertise the "ibm_dynamic_circuits" plugin as preferred,
        # but it requires the qiskit-ibm-transpiler extra (not installed). See
        # Task 5/6 review and hardware.transpile_for_backend (same pin).
        if isa_backend is not None:
            qc_re_t = transpile(
                qc_re, backend=isa_backend, optimization_level=1,
                seed_transpiler=1234, translation_method="translator",
            )
        else:
            qc_re_t = _retranslate_to_basis(qc_re, mode)
        counts_re = _sample_counts(mode, qc_re_t, shots, creg_name=qc_re_t.cregs[0].name)
        p0_re = counts_re.get("0", 0) / shots
        re_pauli = 2 * p0_re - 1
        var_re_pauli = (1 - re_pauli**2) / shots

        qc_im = _hadamard_circuit_for_pauli(qc_left, qc_right, label, imag=True)
        if isa_backend is not None:
            qc_im_t = transpile(
                qc_im, backend=isa_backend, optimization_level=1,
                seed_transpiler=1234, translation_method="translator",
            )
        else:
            qc_im_t = _retranslate_to_basis(qc_im, mode)
        counts_im = _sample_counts(mode, qc_im_t, shots, creg_name=qc_im_t.cregs[0].name)
        p0_im = counts_im.get("0", 0) / shots
        im_pauli = 2 * p0_im - 1
        var_im_pauli = (1 - im_pauli**2) / shots

        total_re += c.real * re_pauli - c.imag * im_pauli
        total_im += c.real * im_pauli + c.imag * re_pauli
        var_re += c.real**2 * var_re_pauli + c.imag**2 * var_im_pauli
        var_im += c.imag**2 * var_re_pauli + c.real**2 * var_im_pauli

    return total_re, total_im, math.sqrt(var_re), math.sqrt(var_im)


def _eval_via_mitigated_estimator(
    mit_est: MitigatedEstimator,
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
) -> tuple[float, float]:
    """Wrap MitigatedEstimator.run for a single (circuit, observable) tuple.

    Shot count is taken from ``mit_est``'s MitigationSpec — the underlying API
    does not accept a per-call shots argument (parameterised PUBs are not yet
    supported either; see MitigatedEstimator.run).

    Returns (expectation, variance).
    """
    job = mit_est.run([(circuit, observable)])
    result = job.result()
    pub = result[0]
    value = float(pub.data.evs)
    std = float(pub.data.stds)
    return value, std**2


def compute_hadamard_test_mitigated(
    *,
    qc_left: QuantumCircuit,
    qc_right: QuantumCircuit,
    operator: SparsePauliOp,
    shots: int = 4096,  # informational; spec.shots controls actual shots
    mitigated_estimator: MitigatedEstimator,
) -> tuple[float, float, float, float]:
    """Hadamard test ⟨ψ_left | O | ψ_right⟩ under M3+ZNE mitigation.

    For each Pauli term, builds the Hadamard circuit (Re or Im) and estimates
    ⟨Z_anc⟩ on the ancilla via ``mitigated_estimator.run``. ⟨Z_anc⟩ = 2·p(0) − 1
    equals the Re or Im component (matching the math from
    ``compute_hadamard_test``).

    The actual shot count is taken from
    ``mitigated_estimator._spec.shots`` — the wrapper's ``shots`` argument is
    accepted for symmetry with ``compute_hadamard_test`` but is informational
    only on the mitigated path. (MitigatedEstimator.run does not currently
    accept a per-PUB shots override.)

    Honors Observation #20 via the same anti-Hermitian warning as
    ``compute_hadamard_test``.
    """
    import warnings

    op_dense = operator.to_matrix()
    if np.allclose(op_dense, -op_dense.conj().T, atol=1e-10) and not np.allclose(op_dense, 0):
        warnings.warn(
            "Operator is anti-Hermitian; if used in a commutator-metric "
            "qEOM pathway, ⟨[T†,T]⟩ will be identically zero. See Observation #20.",
            UserWarning,
            stacklevel=2,
        )

    n_sys = qc_left.num_qubits
    # Hadamard-test circuit has n_sys + 1 qubits; ancilla is qubit 0.
    # Qiskit big-endian Pauli string: rightmost char = qubit 0. So Z on ancilla
    # with I on the n_sys system qubits = "I" * n_sys + "Z".
    z_anc_label = "I" * n_sys + "Z"
    z_anc_op = SparsePauliOp.from_list([(z_anc_label, 1.0)])

    total_re = 0.0
    total_im = 0.0
    var_re = 0.0
    var_im = 0.0

    for pauli, coeff in zip(operator.paulis, operator.coeffs, strict=True):
        label = pauli.to_label()
        c = complex(coeff)

        qc_re = _hadamard_circuit_for_pauli(qc_left, qc_right, label, imag=False)
        qc_re.remove_final_measurements()
        qc_im = _hadamard_circuit_for_pauli(qc_left, qc_right, label, imag=True)
        qc_im.remove_final_measurements()

        re_pauli, var_re_pauli = _eval_via_mitigated_estimator(
            mitigated_estimator, qc_re, z_anc_op
        )
        im_pauli, var_im_pauli = _eval_via_mitigated_estimator(
            mitigated_estimator, qc_im, z_anc_op
        )

        total_re += c.real * re_pauli - c.imag * im_pauli
        total_im += c.real * im_pauli + c.imag * re_pauli
        var_re += c.real**2 * var_re_pauli + c.imag**2 * var_im_pauli
        var_im += c.imag**2 * var_re_pauli + c.real**2 * var_im_pauli

    return total_re, total_im, math.sqrt(var_re), math.sqrt(var_im)
