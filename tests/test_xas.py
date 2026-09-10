"""Tests for siam_vqe.xas — Lorentzian broadening and spectrum assembly."""
from __future__ import annotations

import numpy as np
import pytest

from siam_vqe.xas import XASSpectrum, assemble_xas, lorentzian


def test_lorentzian_integrates_to_one_on_fine_grid():
    # Grid must be wide enough that analytical truncation
    # (2/pi)*arctan(R/gamma) is within tolerance of 1.0; for gamma=0.5
    # and tol=1e-3, need R/gamma >~ 1273 -> R >~ 640.
    omega = np.linspace(-1000, 1000, 200001)
    L = lorentzian(omega, center=0.0, gamma=0.5)
    integral = np.trapezoid(L, omega)
    assert abs(integral - 1.0) < 1e-3


def test_assemble_xas_empty_peaks_gives_zero_spectrum():
    omega = np.linspace(0, 10, 101)
    spec = assemble_xas(
        peak_energies=np.array([]),
        peak_weights=np.array([]),
        omega_grid_eV=omega,
        Gamma_eV=0.5,
        channel="lin_z",
    )
    assert np.allclose(spec.sigma, 0)
    assert spec.channel == "lin_z"


def test_assemble_xas_two_peaks_matches_analytical_sum():
    omega = np.linspace(0, 10, 1001)
    peaks = np.array([3.0, 7.0])
    weights = np.array([1.0, 0.5])
    Gamma = 0.5
    spec = assemble_xas(
        peak_energies=peaks,
        peak_weights=weights,
        omega_grid_eV=omega,
        Gamma_eV=Gamma,
        channel="lin_z",
    )
    expected = (lorentzian(omega, 3.0, Gamma) * 1.0
                + lorentzian(omega, 7.0, Gamma) * 0.5)
    np.testing.assert_allclose(spec.sigma, expected, atol=1e-12)


def test_xas_spectrum_immutability():
    spec = XASSpectrum(
        omega_eV=np.zeros(5),
        sigma=np.zeros(5),
        channel="lin_z",
        Gamma_eV=0.5,
        peak_energies=np.zeros(0),
        peak_weights=np.zeros(0),
    )
    with pytest.raises((AttributeError, Exception)):
        spec.channel = "lin_xy"
