"""Shared pytest fixtures for siam_vqe."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded numpy RNG for reproducible random ansatz initialization."""
    return np.random.default_rng(seed=20260524)


@pytest.fixture
def dimer_params() -> dict[str, float]:
    """Canonical Hubbard-dimer parameters used across tests.

    Half-filling (n=2), Sz=0, with analytic ground-state energy
        E_0 = eps - sqrt(U^2/4 + 4 t^2) + U/2.
    """
    return {"U": 4.0, "t": 1.0, "eps": 0.0}
