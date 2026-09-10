"""XAS cross-section assembly for L-edge spectroscopy on the L3 NiO SIAM.

σ_XAS,q(ω) = Σ_F |⟨F|D_q|ψ_GS⟩|² · L_Γ(ω − (E_F − E_GS))

with L_Γ a normalized Lorentzian and Γ the core-hole lifetime broadening
(Ni L₃ default = 0.5 eV).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XASSpectrum:
    omega_eV: np.ndarray
    sigma: np.ndarray
    channel: str
    Gamma_eV: float
    peak_energies: np.ndarray
    peak_weights: np.ndarray


def lorentzian(omega: np.ndarray, center: float, gamma: float) -> np.ndarray:
    """Normalized Lorentzian (integral = 1)."""
    return (gamma / np.pi) / ((omega - center) ** 2 + gamma ** 2)


def assemble_xas(
    *,
    peak_energies: np.ndarray,
    peak_weights: np.ndarray,
    omega_grid_eV: np.ndarray,
    Gamma_eV: float = 0.5,
    channel: str = "lin_z",
) -> XASSpectrum:
    """Assemble σ_XAS(ω) by summing weighted Lorentzians."""
    if len(peak_energies) != len(peak_weights):
        raise ValueError("peak_energies and peak_weights length mismatch")

    sigma = np.zeros_like(omega_grid_eV, dtype=float)
    for E, w in zip(peak_energies, peak_weights, strict=True):
        sigma += w * lorentzian(omega_grid_eV, E, Gamma_eV)

    return XASSpectrum(
        omega_eV=omega_grid_eV,
        sigma=sigma,
        channel=channel,
        Gamma_eV=Gamma_eV,
        peak_energies=peak_energies.copy(),
        peak_weights=peak_weights.copy(),
    )
