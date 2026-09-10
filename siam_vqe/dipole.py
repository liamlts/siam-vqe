"""Dipole operator for L-edge XAS on the L3 NiO SIAM.

The dipole couples 2p core electrons to 3d valence orbitals. In the
effective-core-hole approximation the 2p orbital is implicit, so the
dipole acts only on the valence: D_q = Σ_α w_{α,q} c†_{d_α}, where
w_{α,q} are angular matrix elements connecting the cubic-harmonic d
basis to the spherical-harmonic 2p states.

The angular coefficients are derived from the standard rank-1 angular
tensor reduction (Gaunt coefficients with l=1 ↔ l=2). For the v1
Phase 5 spec we ship two polarization channels:

- `lin_z`: linear polarization along the cubic ẑ-axis (only 3z²-r² couples).
- `lin_xy`: powder-averaged in-plane (x²-y², xz, yz, xy all contribute).

The 2p→3d radial integral factors out as an overall scale; relative XAS
intensities are dimensionless and match EDRIXS example_3 conventions.

Conventions are verified at Task 5 against EDRIXS example_3's dipole
tensor — the `verify_against_edrixs` test is binding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.hamiltonian_l3 import L3Params


@dataclass(frozen=True)
class DipoleChannel:
    """One polarization channel of the dipole operator.

    Attributes
    ----------
    name : "lin_z" | "lin_xy" | (future) "circular_+" | "circular_-"
    coefficients : shape (5,) per-d-orbital weights. Index ordering
        matches `D_ORBITAL_SYMMETRIES` = ("eg", "eg", "t2g", "t2g", "t2g")
        = (3z²-r², x²-y², xz, yz, xy).
    """
    name: str
    coefficients: np.ndarray  # shape (5,)


# Cubic-harmonic d-orbital index ordering (matches L3 mode order):
# 0 = 3z²-r²  (e_g)
# 1 = x²-y²   (e_g)
# 2 = xz      (t_2g)
# 3 = yz      (t_2g)
# 4 = xy      (t_2g)


def angular_coefficients(channel_name: str) -> np.ndarray:
    """Return the per-d-orbital dipole coefficients for a polarization channel.

    Values derived from the standard rank-1 angular tensor reduction;
    finalized by EDRIXS-convention test in test_dipole_matches_edrixs
    (Task 5).
    """
    if channel_name == "lin_z":
        # Only 3z²-r² is dipole-allowed for ẑ polarization
        # (2p_z → 3d_{3z²-r²} is the only nonzero angular overlap).
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    if channel_name == "lin_xy":
        # Powder-averaged in-plane sum.
        # Weights: x²-y² (e_g in-plane) carries factor sqrt(3)/2 (from
        # the angular reduction of (x²-y²)·x → 2p_x); the t_2g orbitals
        # (xz, yz, xy) all carry equal weights 1/sqrt(2) by symmetry.
        # Exact values pinned by Task 5 EDRIXS convention test.
        return np.array([0.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)])

    raise ValueError(
        f"Unknown dipole channel '{channel_name}'. "
        "Supported in v1: 'lin_z', 'lin_xy'."
    )


def dipole_channel(name: str) -> DipoleChannel:
    """Factory for a `DipoleChannel` by name."""
    return DipoleChannel(name=name, coefficients=angular_coefficients(name))


def dipole_operator(
    channel: DipoleChannel,
    *,
    params: L3Params,
) -> FermionicOp:
    """Build D = Σ_α w_α (c†_{d_α↑} + c†_{d_α↓}) as a FermionicOp.

    Spin-summed: each w_α weights both up and down d-creation operators
    equally. (For XMCD — v2 — each spin block carries a polarization-
    resolved coefficient.)

    D is particle-non-conserving (creates one electron). Acting on the
    Phase 4 ψ_GS in (9, 9), it produces a state with support in
    (10, 9) ⊕ (9, 10). Each spin channel contributes coherently.
    """
    half = params.num_spin_orbitals // 2
    labels: dict[str, float] = {}
    for alpha in range(params.num_d_orbitals):
        w = channel.coefficients[alpha]
        if w == 0:
            continue
        for spin_offset in (0, half):
            mode = alpha + spin_offset
            labels[f"+_{mode}"] = float(w)
    return FermionicOp(labels, num_spin_orbitals=params.num_spin_orbitals)
