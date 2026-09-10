#!/usr/bin/env python3
"""L1 XAS hardware demo driver.

Four backend / run modes:
  --backend statevector    AerSimulator statevector (seconds; deterministic)
  --backend FakeMarrakesh  AerSimulator from FakeMarrakesh fake backend
                           (minutes; noisy + M3 + ZNE mitigated)
  --backend ibm_marrakesh  Real IBM Quantum hardware (queue; mitigated)
  --smoke                  One minimal Hadamard Re term at low shots; confirms
                           end-to-end SamplerV2/Batch dispatch on the target
                           hardware or fake-backend without running the full
                           mitigated sweep.  --output is not required under
                           --smoke (returns before writing any JSON).

Outputs JSON with per-channel σ_XAS(ω), peak energy, and the cross-sector
spectral weight |⟨ψ'_GS | D | ψ_GS⟩|².

Physics
-------
L1 system: 2 spatial orbitals (impurity d + ligand p) × 2 spins = 4
spin-orbital modes. Mode ordering (per `siam_vqe.hamiltonian`):
    0 = d↑, 1 = p↑, 2 = d↓, 3 = p↓.

ψ_GS lives in the (n_↑, n_↓) = (1, 1) sector of H_L1.
ψ'_GS lives in the (2, 1) sector of H'_L1 = H_L1 + V_core (U_dc = 8.5 eV);
this sector is the global minimum of H'_L1 (degenerate with (1, 2) by SU(2)).

VQE construction (sector-locked via parity_tapered mapping, 2 qubits each):
  - ψ_GS  : UCCSD ansatz (2 spatial orbs, (1,1) particles)
  - ψ'_GS : EfficientSU2 ansatz (UCCSD is impossible for (2,1) on 2 spatial
            orbs because the up-spin shell is fully filled — qiskit-nature's
            non-generalized UCCSD pool is empty.) See `tests/test_l1_xas_pipeline.py`.

Dipole / channels
-----------------
L1 has no orbital substructure within the impurity d-shell (only ONE d-orbital
and ONE ligand p-orbital). The dipole operator is therefore the simplest
impurity-d creator with both spin variants:

    D = c†_{d↑} + c†_{d↓}   (mode 0 and mode 2; see `_L1_IMPURITY_MODES`)

The "lin_z" and "lin_xy" channel labels are aliases of the same physical
operator in L1 — a degenerate but methodologically-valid demo. The L2/L3
drivers exercise the truly multi-channel structure.

Cross-sector matrix element
---------------------------
ψ_GS and ψ'_GS are sector-locked to (1, 1) and (2, 1) respectively by their
parity-tapered ansatz constructions, so each lives in its own 2-qubit tapered
Hilbert space. The dipole D maps (1, 1) → (2, 1) ⊕ (1, 2): it is a
cross-sector operator and cannot be evaluated inside either tapered space.

We lift both states to the full 4-qubit Jordan-Wigner basis via
`lift_tapered_to_full_sector` + sector-to-full embedding (mirrors the L3
driver pattern in `scripts/run_l3_xas_sim.py:_tapered_to_full_jw`). The
matrix element is then evaluated either:
  - statevector: classically via ⟨ψ'_GS_full | D_JW | ψ_GS_full⟩
  - FakeMarrakesh / ibm_marrakesh: via Hadamard-test ancilla circuits using
    `QuantumCircuit.prepare_state` to build 4-qubit state-prep circuits.

Output JSON schema
------------------
{
  "backend": str,
  "shots_per_element": int,
  "channels": [str],
  "peak_energy_per_channel": {channel: float},  # eV (shifted so leading peak = 0)
  "delta_e_gs": float,                          # E'_GS - E_GS (eV, before shift)
  "energy_ground_state": float,                 # E_GS (eV)
  "energy_h_prime_ground_state": float,         # E'_GS (eV)
  "weight_squared_per_channel": {channel: float},  # |⟨ψ'_GS|D|ψ_GS⟩|^2 per channel
  "stderr_per_channel": {channel: float},       # 1-sigma error on |amp|^2
  "sigma_by_channel": {channel: [float]},       # σ(ω) on omega_grid
  "omega_grid": [float],
  "wall_time_s": float
}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

# Ensure the package root is on sys.path when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.ansatz import efficient_su2_ansatz, uccsd_ansatz
from siam_vqe.core_hole import CoreHoleParams, nio_l1_core_hole_hamiltonian
from siam_vqe.hamiltonian import _L1_IMPURITY_MODES, nio_l1_anderson
from siam_vqe.mappings import to_qubit_op
from siam_vqe.mitigation import MitigationSpec
from siam_vqe.noise import (
    compute_hadamard_test,
    compute_hadamard_test_mitigated,
    make_mitigated_estimator,
)
from siam_vqe.reference_ed import _build_sector_basis
from siam_vqe.tapering_l3 import lift_tapered_to_full_sector
from siam_vqe.vqe_runner import run_vqe
from siam_vqe.xas import lorentzian

# L1 parameters — half-filling reference (consistent with Task 3a/3b).
_L1_PARAMS = dict(U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5)
_U_DC = 8.5  # core-hole monopole strength (EDRIXS example_3 default)

_L1_NUM_SPIN_ORBITALS = 4
_INITIAL_SECTOR = (1, 1)
_FINAL_SECTOR = (2, 1)
_CHANNELS = ("lin_z", "lin_xy")
_OMEGA_GRID = np.linspace(-4.0, 4.0, 801)
_GAMMA_EV = 0.5  # Ni L₃ core-hole lifetime broadening


@contextmanager
def _tolerant_batch(cm):
    """Wrap a Batch/Session (or nullcontext) so a close-time network error is
    non-fatal. The Runtime job result is obtained inside the block; the session
    PATCH-close afterward can hit a transient ``ConnectionResetError`` that would
    otherwise crash the process with a non-zero exit AFTER the work succeeded.
    Body exceptions still propagate — only the close is made tolerant.
    """
    mode = cm.__enter__()
    try:
        yield mode
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception as exc:
            # Close-time network errors (e.g. ConnectionResetError on the session
            # PATCH) are non-fatal: the job result was already obtained.
            print(f"[run_l1_xas_hw] note: Batch session close raised (ignored): {exc!r}")


def _backend_check(name: str):
    """Pre-flight: confirm the backend is reachable. Halt on failure.

    Returns the backend object that the mitigated-estimator and Hadamard-test
    wrappers consume.
    """
    if name == "statevector":
        return AerSimulator(method="statevector")
    if name == "FakeMarrakesh":
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        return AerSimulator.from_backend(FakeMarrakesh())
    if name == "ibm_marrakesh":
        try:
            import qiskit_ibm_runtime
            print("qiskit_ibm_runtime version:", qiskit_ibm_runtime.__version__)
            svc = qiskit_ibm_runtime.QiskitRuntimeService()
            backends = [b.name for b in svc.backends() if b.status().operational]
            print("operational backends:", backends)
            if "ibm_marrakesh" not in backends:
                raise SystemExit(
                    f"ibm_marrakesh not in operational set: {backends}"
                )
            return svc.backend("ibm_marrakesh")
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"IBM Quantum auth failed: {e!r}") from e
    raise SystemExit(f"unknown backend: {name}")


def _build_l1_states(
    seed: int = 42,
) -> tuple[QuantumCircuit, QuantumCircuit, float, float]:
    """Build ψ_GS in (1,1) via UCCSD and ψ'_GS in (2,1) via EfficientSU2.

    Both circuits live on the parity_tapered (2-qubit) Hilbert space. Sector
    enforcement is inherent to the parity_tapered mapping for the given
    num_particles.

    Returns
    -------
    qc_gs : QuantumCircuit
        Tapered (2-qubit) state-prep circuit for ψ_GS (parameters bound).
    qc_pgs : QuantumCircuit
        Tapered (2-qubit) state-prep circuit for ψ'_GS (parameters bound).
    e_gs : float
        Optimized energy of ψ_GS (eV).
    e_pgs : float
        Optimized energy of ψ'_GS (eV).
    """
    # ψ_GS — H_L1, (1,1) sector, UCCSD.
    h_l1 = nio_l1_anderson(**_L1_PARAMS)
    h_l1_qop = to_qubit_op(h_l1, scheme="parity_tapered",
                           num_particles=_INITIAL_SECTOR)
    ansatz_h, x0_h = uccsd_ansatz(
        num_spatial_orbitals=2,
        num_particles=_INITIAL_SECTOR,
        mapper_scheme="parity_tapered",
        reps=2,
    )
    result_h = run_vqe(
        h_l1_qop, ansatz_h, x0_h,
        optimizer="SLSQP", maxiter=200, seed=seed,
    )
    qc_gs = ansatz_h.assign_parameters(result_h.params)

    # ψ'_GS — H'_L1, (2,1) sector, EfficientSU2 (UCCSD impossible here).
    ch = CoreHoleParams(U_dc=_U_DC)
    h_prime = nio_l1_core_hole_hamiltonian(ch, **_L1_PARAMS)
    hp_qop = to_qubit_op(h_prime, scheme="parity_tapered",
                         num_particles=_FINAL_SECTOR)
    ansatz_hp, x0_hp = efficient_su2_ansatz(
        num_qubits=hp_qop.num_qubits, reps=2, seed=seed,
    )
    result_hp = run_vqe(
        hp_qop, ansatz_hp, x0_hp,
        optimizer="COBYLA", maxiter=600, seed=seed,
    )
    qc_pgs = ansatz_hp.assign_parameters(result_hp.params)

    return qc_gs, qc_pgs, result_h.energy, result_hp.energy


def _embed_sector_to_full(
    psi_sector: np.ndarray,
    sector_basis: list[int],
    num_spin_orbitals: int,
) -> np.ndarray:
    """Scatter a sector-basis statevector into the full 2^N JW Hilbert space.

    Mirrors `scripts/run_l3_xas_sim.py:_embed_sector_to_full`.
    """
    full = np.zeros(2**num_spin_orbitals, dtype=complex)
    full[np.asarray(sector_basis, dtype=np.int64)] = psi_sector
    return full


def _tapered_to_full_jw(
    psi_tapered: np.ndarray, num_particles: tuple[int, int]
) -> np.ndarray:
    """Lift a 2-qubit tapered L1 statevector to the full 2^4 = 16-dim JW basis.

    Two-step: tapered → sector basis (length = sector_dim) → full 2^N JW.
    Re-normalizes to absorb sub-1e-6 sector-leak amplitude noise from VQE
    optimization, identically to `run_l3_xas_sim.py:_tapered_to_full_jw`.
    """
    psi_sector = lift_tapered_to_full_sector(
        psi_tapered,
        num_particles=num_particles,
        num_spin_orbitals=_L1_NUM_SPIN_ORBITALS,
    )
    norm = float(np.linalg.norm(psi_sector))
    if norm == 0.0:
        raise ValueError(
            f"Tapered statevector lifted to zero in sector {num_particles}; "
            "tapering / sector convention mismatch."
        )
    if abs(norm - 1.0) > 1e-4:
        # Renormalize against sector-leak amplitude noise.
        psi_sector = psi_sector / norm
    sector_basis = _build_sector_basis(
        num_d_spin_orbitals=_L1_NUM_SPIN_ORBITALS,
        n_up=num_particles[0],
        n_down=num_particles[1],
    )
    psi_full = _embed_sector_to_full(
        psi_sector, sector_basis, _L1_NUM_SPIN_ORBITALS
    )
    # Final renormalization (handles small sector-leak after embedding).
    full_norm = float(np.linalg.norm(psi_full))
    if full_norm > 0 and abs(full_norm - 1.0) > 1e-12:
        psi_full = psi_full / full_norm
    return psi_full


def _l1_dipole_jw_op() -> SparsePauliOp:
    """L1 dipole operator on the full 4-qubit JW basis.

    D = c†_{d↑} + c†_{d↓}  (mode 0 + mode 2; see `_L1_IMPURITY_MODES`).

    The L1 has no orbital substructure within the d-shell, so the same
    operator is used for every polarization channel (degenerate-but-valid
    L1 demo).
    """
    d_fop = FermionicOp(
        {f"+_{m}": 1.0 for m in _L1_IMPURITY_MODES},
        num_spin_orbitals=_L1_NUM_SPIN_ORBITALS,
    )
    pauli_op = JordanWignerMapper().map(d_fop)
    if not isinstance(pauli_op, SparsePauliOp):
        raise TypeError(
            f"JW map produced {type(pauli_op).__name__}, expected SparsePauliOp"
        )
    return pauli_op


def _full_jw_prep_circuit(psi_full: np.ndarray) -> QuantumCircuit:
    """Build a 4-qubit state-prep circuit on the full JW basis.

    Uses `QuantumCircuit.prepare_state` (modern Qiskit API) — emits no resets
    so the circuit composes cleanly into a Hadamard-test ancilla scheme.
    """
    n_qubits = int(np.log2(len(psi_full)))
    if 2**n_qubits != len(psi_full):
        raise ValueError(
            f"psi_full length {len(psi_full)} is not a power of 2."
        )
    qc = QuantumCircuit(n_qubits)
    qc.prepare_state(Statevector(psi_full), qubits=list(range(n_qubits)))
    return qc.decompose()


def _matrix_element_statevector(
    psi_pgs_full: np.ndarray,
    psi_gs_full: np.ndarray,
    d_qop: SparsePauliOp,
) -> tuple[float, float, float, float]:
    """Exact classical evaluation of ⟨ψ'_GS | D | ψ_GS⟩ via numpy.

    Returns (Re, Im, stderr_Re, stderr_Im) for schema parity with the
    Hadamard-test wrappers. Statevector evaluation is deterministic so both
    stderrs are zero.
    """
    D_mat = d_qop.to_matrix()
    amp = complex(np.vdot(psi_pgs_full, D_mat @ psi_gs_full))
    return float(amp.real), float(amp.imag), 0.0, 0.0


def _matrix_element_hadamard(
    qc_pgs_full: QuantumCircuit,
    qc_gs_full: QuantumCircuit,
    d_qop: SparsePauliOp,
    backend_obj,
    shots: int,
    mitigated_estimator=None,
) -> tuple[float, float, float, float]:
    """Hadamard-test evaluation of ⟨ψ'_GS | D | ψ_GS⟩.

    If `mitigated_estimator` is provided, dispatches through the M3+ZNE
    mitigated path; otherwise uses the raw `compute_hadamard_test` wrapper.
    """
    if mitigated_estimator is not None:
        return compute_hadamard_test_mitigated(
            qc_left=qc_pgs_full,
            qc_right=qc_gs_full,
            operator=d_qop,
            shots=shots,
            mitigated_estimator=mitigated_estimator,
        )
    return compute_hadamard_test(
        qc_left=qc_pgs_full,
        qc_right=qc_gs_full,
        operator=d_qop,
        shots=shots,
        mode=backend_obj,
    )


_DEFAULT_SWEEP_CONFIGS = ("no_mit", "m3_only", "zne_lin_135", "m3_zne_poly3_135")


def _make_mode_cm(backend_obj, real_backend):
    """Fresh per-config execution context yielding the SamplerV2 ``mode``.

    When ``real_backend`` is set (ibm_marrakesh, or FakeMarrakesh smoke rehearsal),
    a NEW ``Batch`` is opened on each call so every config runs in its own
    short-lived session — a session drop or network reset then isolates to one
    config instead of killing the whole sweep (the single-shared-Batch failure mode
    seen on the first real run, error 1217 "Session has been closed"). Otherwise
    (sim path) returns ``nullcontext(backend_obj)`` — a bare/from_backend
    AerSimulator that the estimator routes through its non-ISA sim branch.
    """
    if real_backend is not None:
        from qiskit_ibm_runtime import Batch
        return Batch(backend=real_backend)
    return nullcontext(backend_obj)


def _select_specs(names: list[str]) -> list[MitigationSpec]:
    """Resolve config names against mitigation.make_default_grid()."""
    from siam_vqe.mitigation import make_default_grid

    by_name = {s.name: s for s in make_default_grid()}
    out: list[MitigationSpec] = []
    for n in names:
        if n not in by_name:
            raise SystemExit(
                f"unknown config {n!r}; available: {sorted(by_name)}"
            )
        out.append(by_name[n])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L1 XAS hardware demo driver (statevector / FakeMarrakesh / ibm_marrakesh)"
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["statevector", "FakeMarrakesh", "ibm_marrakesh"],
    )
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=False, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Hardware smoke: one Hadamard Re term, low shots — proves end-to-end "
            "Runtime dispatch via Batch without the full sweep."
        ),
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(_DEFAULT_SWEEP_CONFIGS),
        help=(
            "Comma-separated mitigation config names from make_default_grid(). "
            "Default runs the 4-way comparison. Pass a single name for a "
            "production spectrum (then --output is written)."
        ),
    )
    args = parser.parse_args()

    requested_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if not args.smoke and args.output is None and len(requested_configs) == 1:
        raise SystemExit("--output is required for a single-config production run")

    if args.smoke and args.backend == "statevector":
        print(
            "[run_l1_xas_hw] --smoke ignored for statevector backend "
            "(no Batch/ISA path to exercise); running full statevector sweep."
        )
        args.smoke = False

    t0 = time.time()
    print(
        f"[run_l1_xas_hw] backend = {args.backend}, "
        f"shots = {args.shots}, output = {args.output}"
        + (" [SMOKE]" if args.smoke else "")
    )
    backend_obj = _backend_check(args.backend)

    # A real Runtime Batch is opened only where it's exercised:
    #   - ibm_marrakesh: always (real hardware).
    #   - FakeMarrakesh + --smoke: local rehearsal of the hardware Batch+ISA path.
    #   - everything else (statevector, FakeMarrakesh full sweep): no Batch.
    # `real_backend` set => a Batch is used; the Batch itself is opened lazily,
    # ONE PER CONFIG (see _make_mode_cm), so a session drop isolates to one config.
    real_backend = None
    if args.backend == "ibm_marrakesh":
        real_backend = backend_obj
    elif args.backend == "FakeMarrakesh" and args.smoke:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        real_backend = FakeMarrakesh()  # ISA transpile target for the local smoke rehearsal

    # 1) Build ψ_GS, ψ'_GS via run_vqe (tapered 2-qubit space).
    print("[run_l1_xas_hw] Building ψ_GS (UCCSD, (1,1)) and ψ'_GS "
          "(EfficientSU2, (2,1)) via run_vqe ...")
    qc_gs_tap, qc_pgs_tap, e_gs, e_pgs = _build_l1_states(seed=args.seed)
    delta_e = e_pgs - e_gs
    print(f"  E_GS    = {e_gs:.6f} eV")
    print(f"  E'_GS   = {e_pgs:.6f} eV")
    print(f"  ΔE      = {delta_e:+.6f} eV  (leading XAS peak position, pre-shift)")

    # 2) Lift to full JW (4 qubits) for the cross-sector dipole evaluation.
    psi_gs_tap = Statevector(qc_gs_tap).data
    psi_pgs_tap = Statevector(qc_pgs_tap).data
    psi_gs_full = _tapered_to_full_jw(psi_gs_tap, _INITIAL_SECTOR)
    psi_pgs_full = _tapered_to_full_jw(psi_pgs_tap, _FINAL_SECTOR)

    # 3) Build dipole operator on the full JW basis.
    d_qop = _l1_dipole_jw_op()
    print(
        f"  D_JW: {d_qop.num_qubits} qubits, {len(d_qop)} Pauli terms"
    )

    # Pre-declare per-channel accumulators (filled by the statevector path OR by the
    # single-config production branch inside the Batch block).
    weight_squared_per_channel: dict[str, float] = {}
    stderr_per_channel: dict[str, float] = {}
    sigma_by_channel: dict[str, list[float]] = {}
    peak_energy_per_channel: dict[str, float] = {}

    if args.backend == "statevector":
        re, im, err_re, err_im = _matrix_element_statevector(
            psi_pgs_full, psi_gs_full, d_qop
        )
        weight_squared = re * re + im * im
        for ch_name in _CHANNELS:
            weight_squared_per_channel[ch_name] = float(weight_squared)
            stderr_per_channel[ch_name] = 0.0
            sigma = weight_squared * lorentzian(_OMEGA_GRID, center=0.0, gamma=_GAMMA_EV)
            sigma_by_channel[ch_name] = sigma.tolist()
            peak_energy_per_channel[ch_name] = 0.0
    else:
        # State-prep circuits are classical — build them OUTSIDE any Batch.
        print("[run_l1_xas_hw] Building 4-qubit JW state-prep circuits ...")
        qc_gs_full = _full_jw_prep_circuit(psi_gs_full)
        qc_pgs_full = _full_jw_prep_circuit(psi_pgs_full)

        if args.smoke:
            with _tolerant_batch(_make_mode_cm(backend_obj, real_backend)) as mode:
                term_op = SparsePauliOp.from_list([
                    (d_qop.paulis[0].to_label(), complex(d_qop.coeffs[0]))
                ])
                re, im, _e_re, _e_im = compute_hadamard_test(
                    qc_left=qc_pgs_full,
                    qc_right=qc_gs_full,
                    operator=term_op,
                    shots=min(args.shots, 1024),
                    mode=mode,
                    isa_backend=real_backend,
                )
            print(f"[smoke] Re={re:.4f} Im={im:.4f} finite={np.isfinite(re) and np.isfinite(im)}")
            assert np.isfinite(re) and np.isfinite(im), "smoke produced non-finite value"
            print(f"[smoke] OK — end-to-end SamplerV2/Batch dispatch confirmed in {time.time()-t0:.1f}s")
            return

        # --- Full mitigated sweep over configs. The L1 hardware quantity is the
        # single cross-sector matrix element; the sweep is over mitigation configs,
        # not energy. Each config runs in its OWN Batch (via _make_mode_cm) and is
        # wrapped so a per-config failure (e.g. a dropped session, error 1217)
        # records status="error" and the sweep continues — partial results are still
        # reported. Exact reference computed for the effectiveness report. ---
        re_exact, im_exact, _, _ = _matrix_element_statevector(
            psi_pgs_full, psi_gs_full, d_qop
        )
        abs2_exact = re_exact * re_exact + im_exact * im_exact
        specs = _select_specs(requested_configs)
        if not specs:
            raise SystemExit("--configs resulted in no valid specs")
        print(f"[run_l1_xas_hw] mitigated sweep over configs: {requested_configs}")
        print(f"  exact |amp|^2 (statevector) = {abs2_exact:.6f}")

        sweep_results: dict[str, dict] = {}
        for base_spec in specs:
            spec = MitigationSpec(  # honor the CLI shot count
                name=base_spec.name, m3=base_spec.m3, zne=base_spec.zne,
                zne_extrapolator=base_spec.zne_extrapolator,
                zne_noise_factors=base_spec.zne_noise_factors, shots=args.shots,
            )
            c_t0 = time.time()
            try:
                with _tolerant_batch(_make_mode_cm(backend_obj, real_backend)) as mode:
                    est = make_mitigated_estimator(
                        mode=mode, spec=spec,
                        m3_backend=real_backend, isa_backend=real_backend,
                    )
                    re, im, err_re, err_im = compute_hadamard_test_mitigated(
                        qc_left=qc_pgs_full, qc_right=qc_gs_full,
                        operator=d_qop, mitigated_estimator=est,
                    )
                abs2 = re * re + im * im
                sweep_results[spec.name] = {
                    "re": float(re), "im": float(im),
                    "energy": float(abs2),  # reused by check_mitigation_effectiveness
                    "abs2": float(abs2),
                    "err_re": float(err_re), "err_im": float(err_im),
                    "status": "ok",
                }
                print(
                    f"  config={spec.name}: |amp|^2={abs2:.6f} "
                    f"(exact {abs2_exact:.6f}), wall={time.time()-c_t0:.1f}s"
                )
            except Exception as exc:  # isolate per-config hardware failures
                sweep_results[spec.name] = {"status": "error", "error": repr(exc)}
                print(
                    f"  config={spec.name}: ERROR — {exc!r} "
                    f"(wall={time.time()-c_t0:.1f}s); continuing sweep"
                )

        n_ok = sum(1 for r in sweep_results.values() if r.get("status") == "ok")
        print(f"[run_l1_xas_hw] configs succeeded: {n_ok}/{len(specs)}")

        # Effectiveness report (reused energy-ratio checker; |amp|^2 in 'energy').
        # Reported, NOT gated — dispatch-correctness is the hard gate.
        from siam_vqe.analysis import check_mitigation_effectiveness
        try:
            report = check_mitigation_effectiveness(
                sweep_results, ed_energy=abs2_exact, no_mit_key="no_mit"
            )
            print("[run_l1_xas_hw] mitigation effectiveness vs exact |amp|^2:")
            print(f"  best_config = {report.best_config}, "
                  f"best_ratio = {report.best_ratio:.3f}, "
                  f"passed(<1) = {report.passed}")
            for cname, ratio in report.per_config_ratio.items():
                print(f"    {cname}: ratio = {ratio:.3f}")
        except ValueError as exc:
            print(f"[run_l1_xas_hw] effectiveness report skipped: {exc}")

        # Production spectrum: only when a single config was selected AND it succeeded.
        if args.output is not None and len(specs) == 1:
            only = sweep_results[specs[0].name]
            if only.get("status") == "ok":
                weight_squared = only["abs2"]
                amp_abs = max(np.hypot(only["re"], only["im"]), 1e-12)
                err = float(2.0 * amp_abs * max(only["err_re"], only["err_im"]))
                for ch_name in _CHANNELS:
                    weight_squared_per_channel[ch_name] = float(weight_squared)
                    stderr_per_channel[ch_name] = err
                    sigma = weight_squared * lorentzian(
                        _OMEGA_GRID, center=0.0, gamma=_GAMMA_EV
                    )
                    sigma_by_channel[ch_name] = sigma.tolist()
                    peak_energy_per_channel[ch_name] = 0.0
            else:
                print("[run_l1_xas_hw] production JSON skipped: the single "
                      f"config errored ({only.get('error')}).")
        elif args.output is not None:
            print("[run_l1_xas_hw] --output ignored: production JSON is written "
                  "only when exactly one --configs entry is selected.")

    # 6) Write JSON only when a production spectrum was produced (single config,
    #    or statevector). Multi-config sweeps report to stdout only.
    if args.output is not None and weight_squared_per_channel:
        payload = {
            "backend": args.backend,
            "shots_per_element": int(args.shots),
            "seed": int(args.seed),
            "channels": list(_CHANNELS),
            "peak_energy_per_channel": peak_energy_per_channel,
            "delta_e_gs": float(delta_e),
            "energy_ground_state": float(e_gs),
            "energy_h_prime_ground_state": float(e_pgs),
            "weight_squared_per_channel": weight_squared_per_channel,
            "stderr_per_channel": stderr_per_channel,
            "sigma_by_channel": sigma_by_channel,
            "omega_grid": _OMEGA_GRID.tolist(),
            "wall_time_s": float(time.time() - t0),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"[run_l1_xas_hw] wrote {args.output} "
              f"(wall {time.time() - t0:.1f} s)")
    else:
        print(f"[run_l1_xas_hw] done (wall {time.time() - t0:.1f} s); "
              f"no JSON written (multi-config sweep or no --output).")


if __name__ == "__main__":
    main()
