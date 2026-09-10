#!/usr/bin/env python
"""End-to-end L3 XAS simulator driver (Phase 5 Task 18).

Replays Phase 4 ψ_GS and Phase 5 ψ'_GS, builds qEOM matrices on H' with the
(10, 9) UCCSD pool from ψ'_GS, solves the GHEP for excited-state amplitudes,
computes cross-sector spectral weights ⟨F_n|D_q|ψ_GS⟩ for q ∈ {lin_z, lin_xy},
assembles σ_q(ω) on the EDRIXS reference grid, runs validation layers 3, 4, 5,
and dumps JSON + per-channel figures.

Conventions
-----------
- qEOM is performed only in the (10, 9) final-state sector. The (9, 10) sector
  is its S_z mirror image and contributes the same spectral weight by symmetry,
  so weights are doubled in σ assembly to match EDRIXS reference scaling.
- Peak energies are shifted so that the leading (lowest-energy, nontrivial)
  peak across both polarization channels sits at ω₀ = 0, matching the
  EDRIXS NPZ convention.

Example
-------
    python -u scripts/run_l3_xas_sim.py \\
        --output notebooks/05_L3_xas_result.json \\
        --figures-dir figures
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the package root is on sys.path when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.adapt_vqe import (
    apply_exp_iT,
    build_l3_multistart_seeds,
    build_l3_xas_seeds,
    build_uccsd_pool,
    hf_state_to_tapered_statevector,
    pool_to_tapered_paulis,
)
from siam_vqe.analysis import (
    check_layer_xas_peak_energies,
    check_layer_xas_spectral_weight,
    check_layer_xas_sum_rule,
    plot_xas_spectrum,
)
from siam_vqe.core_hole import CoreHoleParams, tapered_l3_h_prime_pauli
from siam_vqe.dipole import dipole_channel, dipole_operator
from siam_vqe.hamiltonian_l3 import L3Params, nio_l3_hamiltonian
from siam_vqe.qeom import (
    build_eom_matrices,
    compute_spectral_weights,
    solve_qeom,
)
from siam_vqe.reference_ed import _build_sector_basis
from siam_vqe.tapering_l3 import lift_tapered_to_full_sector, tapered_l3_pauli
from siam_vqe.xas import XASSpectrum, assemble_xas

_NUM_SPIN_ORBITALS = 20
_INITIAL_SECTOR = (9, 9)
_FINAL_SECTOR = (10, 9)
_CHANNELS = ("lin_z", "lin_xy")
_WEIGHT_TOL = 1e-12


def _dedup_peaks(
    energies: np.ndarray, weights: np.ndarray, tol_eV: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Merge peaks within `tol_eV` by summing their weights.

    The EDRIXS NPZ concatenates peaks from (10, 9) and (9, 10) sectors, so
    the same physical transition appears twice at numerically-identical
    energies. Layer 3's nearest-peak check counts identical-energy peaks as
    distinct mismatches, so we collapse them here before comparison.
    """
    if len(energies) == 0:
        return energies.copy(), weights.copy()
    order = np.argsort(energies)
    E_sorted = np.asarray(energies)[order]
    W_sorted = np.asarray(weights)[order]
    merged_E: list[float] = [float(E_sorted[0])]
    merged_W: list[float] = [float(W_sorted[0])]
    for E, W in zip(E_sorted[1:], W_sorted[1:], strict=True):
        if abs(E - merged_E[-1]) <= tol_eV:
            merged_W[-1] += float(W)
        else:
            merged_E.append(float(E))
            merged_W.append(float(W))
    return np.array(merged_E), np.array(merged_W)


def _build_qeom_excitation_pool(
    *,
    occupied: list[int],
    virtual: list[int],
    num_spin_orbitals: int = _NUM_SPIN_ORBITALS,
) -> list[FermionicOp]:
    """Build a Bauer-style qEOM excitation pool of c†_a c_i operators.

    These are NOT the antisymmetrized UCCSD generators (c†_a c_i − h.c.) used
    for ADAPT-VQE. The antisymmetric pool has S ≡ 0 on a closed-shell-like
    reference (Observation #20 in `test_qeom.py`); the bare excitation pool
    gives a non-singular metric and the standard chemistry qEOM eigenproblem.

    Each generator is S_z-conserving (no spin-flips).
    """
    half = num_spin_orbitals // 2
    pool: list[FermionicOp] = []
    for i in occupied:
        spin_i = i < half
        for a in virtual:
            spin_a = a < half
            if spin_i != spin_a:
                continue
            pool.append(
                FermionicOp(
                    {f"+_{a} -_{i}": 1.0},
                    num_spin_orbitals=num_spin_orbitals,
                )
            )
    return pool


def _qeom_pool_to_tapered_paulis(
    pool: list[FermionicOp],
    num_particles: tuple[int, int],
) -> list:
    """Tapered Pauli form of the c†_a c_i excitation pool (no i × factor).

    Unlike `pool_to_tapered_paulis`, we do NOT multiply by i: these operators
    are not anti-Hermitian, and `build_eom_matrices` consumes them directly
    via their JW + tapering map.
    """
    return [tapered_l3_pauli(T, num_particles=num_particles).simplify() for T in pool]


def _embed_sector_to_full(
    psi_sector: np.ndarray,
    sector_basis: list[int],
    num_spin_orbitals: int,
) -> np.ndarray:
    """Scatter a sector-basis statevector into the full 2^N JW Hilbert space.

    Mirrors `scripts/build_edrixs_xas_reference.py:_embed_sector_to_full`.
    """
    full = np.zeros(2**num_spin_orbitals, dtype=complex)
    full[np.asarray(sector_basis, dtype=np.int64)] = psi_sector
    return full


def _tapered_to_full_jw(
    psi_tapered: np.ndarray,
    num_particles: tuple[int, int],
) -> np.ndarray:
    """Lift an 18-qubit tapered statevector into the full 2^20 JW basis.

    Two-step: tapered → sector basis (length = sector_dim) → full 2^N JW.
    """
    psi_sector = lift_tapered_to_full_sector(
        psi_tapered,
        num_particles=num_particles,
        num_spin_orbitals=_NUM_SPIN_ORBITALS,
    )
    norm = np.linalg.norm(psi_sector)
    if norm == 0.0:
        raise ValueError(
            f"Tapered statevector lifted to zero in sector {num_particles}; "
            "tapering/sector convention mismatch."
        )
    if abs(norm - 1.0) > 1e-6:
        # Renormalize to suppress sector-leak amplitude noise.
        psi_sector = psi_sector / norm
    sector_basis = _build_sector_basis(
        num_d_spin_orbitals=_NUM_SPIN_ORBITALS,
        n_up=num_particles[0],
        n_down=num_particles[1],
    )
    return _embed_sector_to_full(psi_sector, sector_basis, _NUM_SPIN_ORBITALS)


def _replay_phase4_psi_gs(
    *,
    params: L3Params,
    phase4: dict,
) -> tuple[np.ndarray, float]:
    """Reconstruct Phase 4 ψ_GS in (9, 9) tapered form by replaying ADAPT ops."""
    H = nio_l3_hamiltonian(params)
    H_pauli = tapered_l3_pauli(H, num_particles=_INITIAL_SECTOR)

    # Phase 4 used best_seed_index = 0; rebuild that seed and the matching pool.
    best_seed = int(phase4["best_seed_index"])
    if best_seed != 0:
        raise NotImplementedError(
            "This driver currently assumes best_seed_index = 0 for ψ_GS replay; "
            f"got {best_seed}. Pool partition is derived from seed 0 per Obs #19."
        )
    hf_seeds = build_l3_multistart_seeds(num_seeds=1)
    hf_gs = hf_seeds[0]
    psi_gs_tapered = hf_state_to_tapered_statevector(hf_gs)

    occupied = list(hf_gs.occupied)
    virtual = sorted(set(range(hf_gs.num_spin_orbitals)) - set(occupied))
    fermi_pool = build_uccsd_pool(
        num_spin_orbitals=_NUM_SPIN_ORBITALS,
        occupied=occupied,
        virtual=virtual,
    )
    pool_pauli = pool_to_tapered_paulis(fermi_pool, num_particles=_INITIAL_SECTOR)

    ops_picked = list(phase4["operators_picked"])
    theta = list(phase4["theta"])
    ops = [pool_pauli[k] for k in ops_picked]
    psi_gs_tapered = apply_exp_iT(psi_gs_tapered, ops, theta)

    H_mat = H_pauli.to_matrix(sparse=True)
    energy_check = float(np.real(np.vdot(psi_gs_tapered, H_mat @ psi_gs_tapered)))
    return psi_gs_tapered, energy_check


def _replay_phase5_psi_prime_gs(
    *,
    params: L3Params,
    ch: CoreHoleParams,
    phase5: dict,
) -> tuple[
    np.ndarray,
    float,
    list[int],
    list[int],
]:
    """Reconstruct Phase 5 ψ'_GS in (10, 9) tapered form by replaying ADAPT ops.

    Returns (psi_prime_gs_tapered, energy_check, occupied, virtual). The
    (occupied, virtual) partition of the d⁹ ²E_g reference is returned so the
    caller can build the qEOM excitation pool against the same partition.
    """
    H_prime_pauli = tapered_l3_h_prime_pauli(
        params, ch, num_particles=_FINAL_SECTOR
    )

    best_seed = int(phase5["best_seed_index"])
    if best_seed != 0:
        raise NotImplementedError(
            "This driver currently assumes best_seed_index = 0 for ψ'_GS "
            f"replay; got {best_seed}."
        )
    hf_seeds = build_l3_xas_seeds(num_seeds=1, num_particles=_FINAL_SECTOR)
    hf_p_gs = hf_seeds[0]
    psi_p_gs_tapered = hf_state_to_tapered_statevector(hf_p_gs)

    occupied = list(hf_p_gs.occupied)
    virtual = sorted(set(range(hf_p_gs.num_spin_orbitals)) - set(occupied))
    # ADAPT used the antisymmetrized UCCSD pool — must replay with the SAME
    # pool that selected the operators in the Phase 5 result JSON.
    adapt_fermi_pool = build_uccsd_pool(
        num_spin_orbitals=_NUM_SPIN_ORBITALS,
        occupied=occupied,
        virtual=virtual,
    )
    adapt_pool_pauli = pool_to_tapered_paulis(
        adapt_fermi_pool, num_particles=_FINAL_SECTOR
    )

    ops_picked = list(phase5["operators_picked"])
    theta = list(phase5["theta"])
    ops = [adapt_pool_pauli[k] for k in ops_picked]
    psi_p_gs_tapered = apply_exp_iT(psi_p_gs_tapered, ops, theta)

    H_prime_mat = H_prime_pauli.to_matrix(sparse=True)
    energy_check = float(
        np.real(np.vdot(psi_p_gs_tapered, H_prime_mat @ psi_p_gs_tapered))
    )
    return psi_p_gs_tapered, energy_check, occupied, virtual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L3 XAS simulator driver (Phase 5 Task 18)."
    )
    parser.add_argument(
        "--phase4-result",
        type=Path,
        default=Path("notebooks/04_L3_vqe_result.json"),
    )
    parser.add_argument(
        "--phase5-h-prime",
        type=Path,
        default=Path("notebooks/05_L3_h_prime_vqe_result.json"),
    )
    parser.add_argument(
        "--edrixs-reference",
        type=Path,
        default=Path("data/edrixs_xas_l3_reference.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/05_L3_xas_result.json"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
    )
    args = parser.parse_args()

    params = L3Params()
    ch = CoreHoleParams()

    print(f"[run_l3_xas_sim] U_dc = {ch.U_dc} eV")

    # --- 1. Replay Phase 4 ψ_GS in (9, 9). ----------------------------------
    print("[run_l3_xas_sim] Replaying Phase 4 ψ_GS ...")
    with args.phase4_result.open() as f:
        phase4 = json.load(f)
    t0 = time.time()
    psi_gs_tapered, e_gs_check = _replay_phase4_psi_gs(
        params=params, phase4=phase4
    )
    e_gs = float(phase4["final_energy"])
    print(
        f"  ψ_GS reconstructed: ⟨H⟩ = {e_gs_check:.6f} eV "
        f"(JSON: {e_gs:.6f} eV) in {time.time()-t0:.1f}s"
    )

    # --- 2. Replay Phase 5 ψ'_GS in (10, 9). --------------------------------
    print("[run_l3_xas_sim] Replaying Phase 5 ψ'_GS ...")
    with args.phase5_h_prime.open() as f:
        phase5_h_prime = json.load(f)
    t0 = time.time()
    (
        psi_p_gs_tapered,
        e_p_gs_check,
        occupied_p,
        virtual_p,
    ) = _replay_phase5_psi_prime_gs(
        params=params, ch=ch, phase5=phase5_h_prime
    )
    e_p_gs = float(phase5_h_prime["best_energy"])
    print(
        f"  ψ'_GS reconstructed: ⟨H'⟩ = {e_p_gs_check:.6f} eV "
        f"(JSON: {e_p_gs:.6f} eV) in {time.time()-t0:.1f}s"
    )

    # --- 3. Build qEOM excitation pool + matrices, solve GHEP. --------------
    print("[run_l3_xas_sim] Building qEOM pool + matrices on H' ...")
    t0 = time.time()
    qeom_fermi_pool = _build_qeom_excitation_pool(
        occupied=occupied_p, virtual=virtual_p
    )
    qeom_pool_pauli = _qeom_pool_to_tapered_paulis(
        qeom_fermi_pool, num_particles=_FINAL_SECTOR
    )
    print(f"  qEOM pool size = {len(qeom_fermi_pool)} (c†_a c_i form)")
    H_prime_pauli = tapered_l3_h_prime_pauli(
        params, ch, num_particles=_FINAL_SECTOR
    )
    M, S = build_eom_matrices(H_prime_pauli, qeom_pool_pauli, psi_p_gs_tapered)
    energies_qeom, amplitudes = solve_qeom(M, S, s_tol=1e-8)
    print(
        f"  qEOM: {len(energies_qeom)} surviving excitations "
        f"(S dropped {len(qeom_pool_pauli) - len(energies_qeom)}); "
        f"{time.time()-t0:.1f}s"
    )

    # --- 4. Lift both ψ_GS and ψ'_GS to the full 2^20 JW basis. -------------
    print("[run_l3_xas_sim] Lifting to full JW basis ...")
    t0 = time.time()
    psi_gs_full = _tapered_to_full_jw(psi_gs_tapered, _INITIAL_SECTOR)
    psi_p_gs_full = _tapered_to_full_jw(psi_p_gs_tapered, _FINAL_SECTOR)
    print(f"  Lifts done in {time.time()-t0:.1f}s")

    # --- 5. Per-channel spectral weights + σ assembly. ----------------------
    edrixs = np.load(args.edrixs_reference, allow_pickle=True)
    omega_grid = edrixs["omega_grid_eV"]
    gamma = float(edrixs["Gamma_eV"])

    # Raw per-channel peaks (E_F − E_GS) and weights, BEFORE shift.
    # Two contributions:
    #   (a) Ground-to-ground: |⟨ψ'_GS | D | ψ_GS⟩|² at peak E_p_GS − E_GS.
    #       Bauer qEOM excludes the reference itself, so this term must be
    #       added explicitly to recover the full Lehmann sum.
    #   (b) qEOM excited states: |⟨F_n | D | ψ_GS⟩|² at (E_p_GS + ε_n) − E_GS.
    jw = JordanWignerMapper()
    raw_per_channel: dict[str, dict] = {}
    for ch_name in _CHANNELS:
        D = dipole_operator(dipole_channel(ch_name), params=params)
        D_mat = jw.map(D).to_matrix(sparse=True)

        # (a) Ground-to-ground amplitude.
        D_psi_gs = D_mat @ psi_gs_full
        amp_gs2gs = complex(np.vdot(psi_p_gs_full, D_psi_gs))
        w_gs2gs = float(abs(amp_gs2gs) ** 2)
        E_gs2gs = float(e_p_gs - e_gs)

        # (b) qEOM excited-state weights.
        weights_qeom = compute_spectral_weights(
            psi_gs_full=psi_gs_full,
            psi_prime_gs_full=psi_p_gs_full,
            pool_fermi=qeom_fermi_pool,
            amplitudes=amplitudes,
            dipole=D,
            num_spin_orbitals=_NUM_SPIN_ORBITALS,
        )
        E_qeom = (e_p_gs + energies_qeom) - e_gs

        peaks_raw = np.concatenate([[E_gs2gs], E_qeom])
        weights_all = np.concatenate([[w_gs2gs], weights_qeom])

        raw_per_channel[ch_name] = {
            "peaks_raw": peaks_raw,
            "weights": weights_all,
            "D": D,
            "gs2gs_weight": w_gs2gs,
            "gs2gs_energy": E_gs2gs,
        }

    # Leading-peak shift: lowest peak across both channels with nontrivial weight.
    leading_candidates = []
    for ch_name in _CHANNELS:
        E = raw_per_channel[ch_name]["peaks_raw"]
        W = raw_per_channel[ch_name]["weights"]
        mask = W > _WEIGHT_TOL
        if mask.any():
            leading_candidates.append(float(E[mask].min()))
    if not leading_candidates:
        raise RuntimeError(
            "All channels produced zero spectral weight. "
            "Check dipole convention and qEOM pool."
        )
    leading_E = min(leading_candidates)
    print(f"[run_l3_xas_sim] Leading peak at E_F − E_GS = {leading_E:.6f} eV "
          f"(shifted to ω₀ = 0)")

    # --- 6. Assemble σ_q(ω) and run validation layers. ----------------------
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    for ch_name in _CHANNELS:
        D = raw_per_channel[ch_name]["D"]
        peak_E_raw = raw_per_channel[ch_name]["peaks_raw"]
        weights = raw_per_channel[ch_name]["weights"]

        # Drop numerically-zero weights to declutter peak comparison.
        keep_mask = weights > _WEIGHT_TOL
        peaks_shifted = peak_E_raw[keep_mask] - leading_E
        weights_kept = weights[keep_mask]

        # Double weights to account for the (9, 10) S_z-mirror sector
        # (not explicitly computed; symmetry-equivalent to (10, 9)).
        weights_for_sigma = 2.0 * weights_kept

        spectrum = assemble_xas(
            peak_energies=peaks_shifted,
            peak_weights=weights_for_sigma,
            omega_grid_eV=omega_grid,
            Gamma_eV=gamma,
            channel=ch_name,
        )

        edrixs_sigma = edrixs[f"sigma_{ch_name}"]
        edrixs_peaks = edrixs[f"peak_energies_{ch_name}"]
        edrixs_weights = edrixs[f"peak_weights_{ch_name}"]
        edrixs_spec = XASSpectrum(
            omega_eV=omega_grid,
            sigma=edrixs_sigma,
            channel=ch_name,
            Gamma_eV=gamma,
            peak_energies=edrixs_peaks,
            peak_weights=edrixs_weights,
        )

        # Layer 3: peak energy agreement. The EDRIXS NPZ concatenates peaks
        # from (10, 9) and (9, 10) sectors at identical energies — deduplicate
        # on both sides before nearest-peak comparison so the count test is
        # over unique physical transitions.
        p5_E_dedup, p5_W_dedup = _dedup_peaks(peaks_shifted, weights_for_sigma)
        ed_E_dedup, ed_W_dedup = _dedup_peaks(edrixs_peaks, edrixs_weights)
        layer3 = check_layer_xas_peak_energies(
            phase5_peaks=p5_E_dedup,
            edrixs_peaks=ed_E_dedup,
            phase5_weights=p5_W_dedup,
            edrixs_weights=ed_W_dedup,
            tolerance_eV=0.05,
        )

        # Layer 4: spectral-weight L2 distance.
        layer4 = check_layer_xas_spectral_weight(
            sigma_phase5=spectrum.sigma,
            sigma_edrixs=edrixs_sigma,
            tolerance_frac=0.05,
        )

        # Layer 5: sum-rule against full ⟨GS|D†D|GS⟩.
        # The (10, 9) qEOM captures one spin channel; we double its sum-rule
        # contribution to match the full ⟨D†D⟩.
        jw = JordanWignerMapper()
        DdD_pauli = jw.map((D.adjoint() @ D).simplify())
        DdD_mat = DdD_pauli.to_matrix(sparse=True)
        sum_rule_expected = float(
            np.real(np.vdot(psi_gs_full, DdD_mat @ psi_gs_full))
        )
        sum_weights = float(weights_for_sigma.sum())
        layer5 = check_layer_xas_sum_rule(
            sum_weights=sum_weights,
            expected=sum_rule_expected,
            tolerance_frac=0.01,
        )

        # Per-channel figure.
        fig_path = args.figures_dir / f"05_xas_{ch_name}.pdf"
        plot_xas_spectrum(
            spectrum,
            edrixs_spectrum=edrixs_spec,
            output_path=str(fig_path),
        )

        results[ch_name] = {
            "peak_energies": peaks_shifted.tolist(),
            "peak_weights": weights_for_sigma.tolist(),
            "peak_weights_raw_sector": weights_kept.tolist(),
            "sigma": spectrum.sigma.tolist(),
            "edrixs_sigma": edrixs_sigma.tolist(),
            "edrixs_peak_energies": edrixs_peaks.tolist(),
            "edrixs_peak_weights": edrixs_weights.tolist(),
            "sum_weights": sum_weights,
            "sum_rule_expected": sum_rule_expected,
            "layer3_peak_energies": layer3,
            "layer4_spectral_weight": layer4,
            "layer5_sum_rule": layer5,
            "figure_path": str(fig_path),
        }
        print(
            f"  channel {ch_name}: "
            f"L3 pass={layer3['pass']}, "
            f"L4 pass={layer4['pass']} (L2={layer4['l2_distance_frac']:.3f}), "
            f"L5 pass={layer5['pass']} (Σw={sum_weights:.4f} vs "
            f"⟨D†D⟩={sum_rule_expected:.4f})"
        )

    # --- 7. Aggregate phase5_pass. ------------------------------------------
    phase5_pass = all(
        results[ch_name]["layer3_peak_energies"]["pass"]
        and results[ch_name]["layer4_spectral_weight"]["pass"]
        and results[ch_name]["layer5_sum_rule"]["pass"]
        for ch_name in _CHANNELS
    )

    out_payload: dict[str, object] = dict(results)
    out_payload["phase5_pass"] = bool(phase5_pass)
    out_payload["omega_grid_eV"] = omega_grid.tolist()
    out_payload["Gamma_eV"] = gamma
    out_payload["E_GS_initial_eV"] = e_gs
    out_payload["E_prime_GS_eV"] = e_p_gs
    out_payload["leading_peak_energy_eV"] = leading_E
    out_payload["energies_qeom_eV"] = energies_qeom.tolist()
    out_payload["pool_size"] = len(qeom_pool_pauli)
    out_payload["s_dropped"] = len(qeom_pool_pauli) - len(energies_qeom)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            out_payload,
            f,
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o),
        )
    print(
        f"[run_l3_xas_sim] Wrote {args.output}, phase5_pass = {phase5_pass}"
    )


if __name__ == "__main__":
    main()
