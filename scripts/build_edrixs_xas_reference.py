#!/usr/bin/env python
"""Build the EDRIXS-style XAS reference NPZ at data/edrixs_xas_l3_reference.npz.

Self-consistent reference via scipy ED of H' — used when edrixs is unavailable.

We diagonalise H' = H + V_core in both XAS final-state sectors (10, 9) and
(9, 10), evaluate cross-sector dipole matrix elements
    ⟨F_n | D_q | ψ_GS⟩
against Phase 4's exact (9, 9) ground state, and assemble σ_q(ω) on a
fixed grid. The two spin-channel sectors are orthogonal (D_q creates either
an up or a down d-electron); peak weights from each sector add coherently
in σ_q(ω).

Grid: ω ∈ [−4, +4] eV in 401 points, with the leading peak shifted to 0.
Lorentzian core-hole broadening Γ = 0.5 eV. Channels: lin_z, lin_xy.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Project root = parent of this scripts/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from siam_vqe.core_hole import CoreHoleParams, nio_l3_core_hole_hamiltonian
from siam_vqe.dipole import dipole_channel, dipole_operator
from siam_vqe.hamiltonian_l3 import L3Params, nio_l3_hamiltonian
from siam_vqe.reference_ed import (
    _build_sector_basis,
    _fermionic_op_to_sparse_matrix,
)
from siam_vqe.xas import assemble_xas

OUT_PATH = _PROJECT_ROOT / "data" / "edrixs_xas_l3_reference.npz"
GAMMA_EV = 0.5
OMEGA_GRID = np.linspace(-4.0, 4.0, 401)
CHANNELS = ("lin_z", "lin_xy")
WEIGHT_TOL = 1e-12  # drop numerically-zero peaks


def _embed_sector_to_full(
    psi_sector: np.ndarray,
    sector_basis: list[int],
    num_spin_orbitals: int,
) -> np.ndarray:
    """Place a sector-basis statevector into the full 2^N JW Hilbert space."""
    full = np.zeros(2**num_spin_orbitals, dtype=complex)
    full[np.asarray(sector_basis, dtype=np.int64)] = psi_sector
    return full


def _diagonalise_sector(H_fermi, basis: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Dense eigh of a fermionic operator restricted to a sector basis."""
    H_mat = _fermionic_op_to_sparse_matrix(H_fermi, basis).toarray()
    H_mat = 0.5 * (H_mat + H_mat.conj().T)
    return np.linalg.eigh(H_mat)


def _peak_set_for_sector(
    *,
    psi_gs_full: np.ndarray,
    H_prime_fermi,
    sector: tuple[int, int],
    D_mat,
    num_spin_orbitals: int,
    e_gs_initial: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise H' in `sector`, compute |⟨F_n|D|ψ_GS⟩|² and (E_F − E_GS)."""
    basis = _build_sector_basis(
        num_d_spin_orbitals=num_spin_orbitals,
        n_up=sector[0],
        n_down=sector[1],
    )
    eigvals, eigvecs = _diagonalise_sector(H_prime_fermi, basis)

    D_psi_gs = D_mat @ psi_gs_full

    peak_E = np.zeros(len(eigvals))
    peak_W = np.zeros(len(eigvals))
    for n in range(len(eigvals)):
        psi_F_full = _embed_sector_to_full(
            eigvecs[:, n], basis, num_spin_orbitals
        )
        amp = np.vdot(psi_F_full, D_psi_gs)
        peak_E[n] = eigvals[n] - e_gs_initial
        peak_W[n] = abs(amp) ** 2

    return peak_E, peak_W


def main() -> None:
    params = L3Params()
    ch = CoreHoleParams()  # U_dc = 8.5, no multipoles, ζ_d = 0 by absence
    print(f"[build_edrixs_xas_reference] Output: {OUT_PATH}")
    print(
        f"[build_edrixs_xas_reference] L3 params: F2={params.F2_dd}, "
        f"F4={params.F4_dd}, U_dd={params.U_dd}, Δ={params.Delta}, "
        f"V_eg={params.V_eg}, V_t2g={params.V_t2g}, 10Dq={params.ten_dq}, "
        f"cf_bath_split={params.cf_bath_split}, U_dc={ch.U_dc}"
    )

    # Phase 4 H ED in (9, 9): produce ψ_GS in the sector basis and embed.
    H_fermi = nio_l3_hamiltonian(params)
    init_basis = _build_sector_basis(
        num_d_spin_orbitals=params.num_spin_orbitals, n_up=9, n_down=9
    )
    e_init_all, vec_init_all = _diagonalise_sector(H_fermi, init_basis)
    e_gs = float(e_init_all[0])
    psi_gs_sector = vec_init_all[:, 0]
    psi_gs_full = _embed_sector_to_full(
        psi_gs_sector, init_basis, params.num_spin_orbitals
    )
    print(
        f"[build_edrixs_xas_reference] ψ_GS in (9, 9): E = {e_gs:.6f} eV "
        f"(sector dim = {len(init_basis)})"
    )

    # H' = H + V_core; we'll ED in (10, 9) and (9, 10).
    H_prime_fermi = nio_l3_core_hole_hamiltonian(params, ch)

    # Verify both sector dimensions = C(10, n_up) * C(10, n_dn).
    expected_dim = sum(1 for _ in combinations(range(10), 10)) * sum(
        1 for _ in combinations(range(10), 9)
    )
    print(
        f"[build_edrixs_xas_reference] Expected (10, 9) / (9, 10) sector "
        f"dimension = {expected_dim}"
    )

    jw = JordanWignerMapper()

    per_channel: dict[str, dict[str, np.ndarray]] = {}
    for channel_name in CHANNELS:
        channel = dipole_channel(channel_name)
        D_fermi = dipole_operator(channel, params=params)
        D_mat = jw.map(D_fermi).to_matrix(sparse=True)

        # Final states in (10, 9) — d-electron added with spin up.
        E_a, W_a = _peak_set_for_sector(
            psi_gs_full=psi_gs_full,
            H_prime_fermi=H_prime_fermi,
            sector=(10, 9),
            D_mat=D_mat,
            num_spin_orbitals=params.num_spin_orbitals,
            e_gs_initial=e_gs,
        )
        # Final states in (9, 10) — d-electron added with spin down.
        E_b, W_b = _peak_set_for_sector(
            psi_gs_full=psi_gs_full,
            H_prime_fermi=H_prime_fermi,
            sector=(9, 10),
            D_mat=D_mat,
            num_spin_orbitals=params.num_spin_orbitals,
            e_gs_initial=e_gs,
        )

        peak_E = np.concatenate([E_a, E_b])
        peak_W = np.concatenate([W_a, W_b])
        per_channel[channel_name] = {"E_raw": peak_E, "W_raw": peak_W}

        nonzero = peak_W > WEIGHT_TOL
        print(
            f"  channel {channel_name}: {int(nonzero.sum())} peaks with "
            f"w > {WEIGHT_TOL:g} out of {len(peak_W)}; "
            f"Σw = {peak_W.sum():.4f}"
        )

    # Shift convention: leading peak (lowest-energy peak with nontrivial
    # weight, max across channels) goes to ω₀ = 0.
    leading_E_candidates = []
    for name in CHANNELS:
        E = per_channel[name]["E_raw"]
        W = per_channel[name]["W_raw"]
        mask = W > WEIGHT_TOL
        if mask.any():
            leading_E_candidates.append(float(E[mask].min()))
    if not leading_E_candidates:
        raise RuntimeError(
            "All channels produced zero spectral weight — physics check failed."
        )
    leading_E = min(leading_E_candidates)
    print(
        f"[build_edrixs_xas_reference] Leading peak at E_F - E_GS = "
        f"{leading_E:.6f} eV (shift this to ω₀ = 0)"
    )

    sigma: dict[str, np.ndarray] = {}
    shifted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in CHANNELS:
        E_raw = per_channel[name]["E_raw"]
        W_raw = per_channel[name]["W_raw"]
        keep = W_raw > WEIGHT_TOL
        E_shift = E_raw[keep] - leading_E
        W_keep = W_raw[keep]
        spec = assemble_xas(
            peak_energies=E_shift,
            peak_weights=W_keep,
            omega_grid_eV=OMEGA_GRID,
            Gamma_eV=GAMMA_EV,
            channel=name,
        )
        sigma[name] = spec.sigma
        shifted[name] = (E_shift, W_keep)

    notes = (
        "Self-consistent XAS reference via scipy ED of H' in (10, 9) and "
        "(9, 10) — used when edrixs is unavailable. ψ_GS from H ED in "
        f"(9, 9); E_GS = {e_gs:.6f} eV. Leading peak at "
        f"(E_F − E_GS) = {leading_E:.6f} eV shifted to ω₀ = 0. "
        f"U_dc = {ch.U_dc} eV, Γ = {GAMMA_EV} eV, ζ_d = 0, "
        "multipoles G^1 = G^3 = 0. EDRIXS example_3 / Haverkort PRB 85, "
        "165113 (2012) conventions."
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        omega_grid_eV=OMEGA_GRID,
        sigma_lin_z=sigma["lin_z"],
        sigma_lin_xy=sigma["lin_xy"],
        Gamma_eV=GAMMA_EV,
        notes=notes,
        peak_energies_lin_z=shifted["lin_z"][0],
        peak_weights_lin_z=shifted["lin_z"][1],
        peak_energies_lin_xy=shifted["lin_xy"][0],
        peak_weights_lin_xy=shifted["lin_xy"][1],
        leading_peak_energy_eV=leading_E,
        E_GS_initial_eV=e_gs,
    )
    print(f"[build_edrixs_xas_reference] Saved -> {OUT_PATH}")
    print(
        f"[build_edrixs_xas_reference] max σ_lin_z = {sigma['lin_z'].max():.4f}, "
        f"max σ_lin_xy = {sigma['lin_xy'].max():.4f}"
    )


if __name__ == "__main__":
    main()
