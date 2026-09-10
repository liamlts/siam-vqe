"""L3 observables for the d⁸ + ligand-bath SIAM.

Mode ordering (must match hamiltonian_l3):
    0..4   d orbitals (3z²−r², x²−y², xz, yz, xy), spin↑
    5..9   bath orbitals, spin↑
    10..14 d orbitals, spin↓
    15..19 bath orbitals, spin↓
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.reference_ed import _fermionic_op_to_sparse_matrix

if TYPE_CHECKING:
    from siam_vqe.reference_l3 import L3Reference

_N_SPATIAL = 10  # 5 d + 5 bath
_N_D = 5
_N_BATH = 5
_D_NAMES = ("3z2", "x2y2", "xz", "yz", "xy")


def _make_number_op(mode: int, num_spin_orbitals: int) -> FermionicOp:
    return FermionicOp({f"+_{mode} -_{mode}": 1.0}, num_spin_orbitals=num_spin_orbitals)


def _make_d_orbital_number_op(alpha: int, num_spin_orbitals: int) -> FermionicOp:
    """n_{d_α} = n_{d_α↑} + n_{d_α↓}."""
    up = _make_number_op(alpha, num_spin_orbitals)
    dn = _make_number_op(alpha + _N_SPATIAL, num_spin_orbitals)
    return (up + dn).simplify()


def _make_S2(num_spin_orbitals: int) -> FermionicOp:
    """S² = S_+ S_- + S_z² - S_z, on the impurity d shell only."""
    n = num_spin_orbitals
    # S_z = ½ Σ_α (n_{α↑} - n_{α↓})  (only over impurity d modes)
    sz_labels: dict[str, float] = {}
    for alpha in range(_N_D):
        sz_labels[f"+_{alpha} -_{alpha}"] = sz_labels.get(f"+_{alpha} -_{alpha}", 0.0) + 0.5
        a_dn = alpha + _N_SPATIAL
        sz_labels[f"+_{a_dn} -_{a_dn}"] = sz_labels.get(f"+_{a_dn} -_{a_dn}", 0.0) - 0.5
    S_z = FermionicOp(sz_labels, num_spin_orbitals=n)

    # S_+ = Σ_α c†_{α↑} c_{α↓}
    sp_labels: dict[str, float] = {}
    for alpha in range(_N_D):
        a_up = alpha
        a_dn = alpha + _N_SPATIAL
        sp_labels[f"+_{a_up} -_{a_dn}"] = 1.0
    S_plus = FermionicOp(sp_labels, num_spin_orbitals=n)
    S_minus = S_plus.adjoint()

    # S² = S_+ S_- + S_z² - S_z
    S_sq = (S_plus @ S_minus + S_z @ S_z - S_z).simplify()
    return S_sq


def make_l3_observables(num_spin_orbitals: int = 20) -> dict[str, FermionicOp]:
    """Return the L3 observable dictionary.

    Keys: n_d, n_p, S2, n_d_3z2, n_d_x2y2, n_d_xz, n_d_yz, n_d_xy.
    """
    obs: dict[str, FermionicOp] = {}

    # Total d-shell occupation.
    n_d_labels: dict[str, float] = {}
    for spin in (0, 1):
        for alpha in range(_N_D):
            mode = alpha + spin * _N_SPATIAL
            n_d_labels[f"+_{mode} -_{mode}"] = 1.0
    obs["n_d"] = FermionicOp(n_d_labels, num_spin_orbitals=num_spin_orbitals).simplify()

    # Total bath occupation.
    n_p_labels: dict[str, float] = {}
    for spin in (0, 1):
        for beta in range(_N_BATH):
            mode = _N_D + beta + spin * _N_SPATIAL
            n_p_labels[f"+_{mode} -_{mode}"] = 1.0
    obs["n_p"] = FermionicOp(n_p_labels, num_spin_orbitals=num_spin_orbitals).simplify()

    # Per-d-orbital occupations.
    for alpha, name in enumerate(_D_NAMES):
        obs[f"n_d_{name}"] = _make_d_orbital_number_op(alpha, num_spin_orbitals)

    # Total spin-squared on the d shell.
    obs["S2"] = _make_S2(num_spin_orbitals)

    return obs


def evaluate_observable_on_sector(
    operator: FermionicOp,
    sector_vector: np.ndarray,
    basis: Iterable[int],
    num_spin_orbitals: int,
) -> float:
    """⟨ψ|O|ψ⟩ for a sector-basis statevector."""
    basis_list = list(basis)
    O_mat = _fermionic_op_to_sparse_matrix(operator, basis_list)
    return float(np.real(np.vdot(sector_vector, O_mat @ sector_vector)))


def compute_reference_observables(ref: L3Reference) -> dict[str, float]:
    """Evaluate every observable on the L3 reference ground vector."""
    obs_ops = make_l3_observables(num_spin_orbitals=20)
    return {
        name: evaluate_observable_on_sector(
            op, ref.ground_vector, basis=ref.basis, num_spin_orbitals=20,
        )
        for name, op in obs_ops.items()
    }
