"""Tests for siam_vqe.reference_edrixs — closed-form (E_d, E_L) from Haverkort/EDRIXS."""

from __future__ import annotations

import pytest

from siam_vqe.reference_edrixs import compute_l1_levels


def test_compute_l1_levels_returns_two_floats() -> None:
    eps_d, eps_p = compute_l1_levels()
    assert isinstance(eps_d, float)
    assert isinstance(eps_p, float)


def test_compute_l1_levels_default_values_match_haverkort() -> None:
    """Defaults: U_dd=7.3, Delta=4.7, nd=8, ten_dq=0.56, ten_dq_bath=1.44.

    Closed-form from CT_imp_bath (edrixs/utils.py):
        E_d = (10*Delta - n*(19+n)*U_dd/2) / (10+n)
            = (10*4.7 - 8*27*3.65)/18
            = (47 - 788.4)/18 = -41.188888...
        E_L = n*((1+n)*U_dd/2 - Delta)/(10+n)
            = 8*(32.85 - 4.7)/18 = 12.511111...

    Plus eg CF shifts (+0.6*10Dq):
        eps_d = E_d + 0.336 = -40.852888...
        eps_p = E_L + 0.864 = 13.375111...
    """
    eps_d, eps_p = compute_l1_levels()
    assert eps_d == pytest.approx(-40.85288888888889, abs=1e-9)
    assert eps_p == pytest.approx(13.375111111111111, abs=1e-9)


def test_compute_l1_levels_no_cf_when_ten_dq_zero() -> None:
    """With ten_dq=0 and ten_dq_bath=0, returns the bare (E_d, E_L) from CT_imp_bath."""
    eps_d, eps_p = compute_l1_levels(ten_dq=0.0, ten_dq_bath=0.0)
    assert eps_d == pytest.approx(-41.18888888888889, abs=1e-9)
    assert eps_p == pytest.approx(12.511111111111111, abs=1e-9)


def test_compute_l1_levels_atomic_limit_zero_delta() -> None:
    """At Delta=0 (full p-shell already aligned), CT_imp_bath gives:
        E_d = -n*(19+n)*U_dd/2 / (10+n)
        E_L = n*(1+n)*U_dd/2 / (10+n)
    For n=8, U_dd=7.3: E_d = -788.4/18 = -43.8; E_L = 32.85*8/18 = 14.6.
    """
    eps_d, eps_p = compute_l1_levels(Delta=0.0, ten_dq=0.0, ten_dq_bath=0.0)
    assert eps_d == pytest.approx(-43.8, abs=1e-9)
    assert eps_p == pytest.approx(14.6, abs=1e-9)


def test_compute_l1_levels_zero_U_collapses_to_simple_form() -> None:
    """At U_dd=0, CT_imp_bath simplifies:
        E_d = 10*Delta/(10+n);  E_L = -n*Delta/(10+n)
    For n=8, Delta=4.7: E_d = 47/18 = 2.6111; E_L = -37.6/18 = -2.0889.
    """
    eps_d, eps_p = compute_l1_levels(U_dd=0.0, ten_dq=0.0, ten_dq_bath=0.0)
    assert eps_d == pytest.approx(2.6111111111111111, abs=1e-9)
    assert eps_p == pytest.approx(-2.0888888888888888, abs=1e-9)
