"""L3 reference: scipy.sparse exact diagonalization in the (9, 9) sector.

The (N_↑, N_↓) = (9, 9) sector has dimension D = C(10, 9)² = 100.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse.linalg as spla
from qiskit_nature.second_q.operators import FermionicOp

from siam_vqe.hamiltonian_l3 import L3Params, nio_l3_hamiltonian
from siam_vqe.reference_ed import _build_sector_basis, _fermionic_op_to_sparse_matrix

if TYPE_CHECKING:
    from siam_vqe.core_hole import CoreHoleParams


@dataclass(frozen=True)
class L3Reference:
    """Cached L3 reference result in the (9, 9) sector basis.

    Attributes
    ----------
    ground_energy : float
    ground_vector : np.ndarray, shape (sector_dim,)
    excited_energies : np.ndarray, sorted ascending
    sector_dim : int
    sector : tuple[int, int]
    basis : tuple[int, ...]
        Occupation-number integers (bits index occupied modes).
    observables : dict[str, float]
        Reference expectation values. Empty here; populated by
        observables_l3.compute_reference_observables in a later task.
    """
    ground_energy: float
    ground_vector: np.ndarray
    excited_energies: np.ndarray
    sector_dim: int
    sector: tuple[int, int]
    basis: tuple[int, ...]
    observables: dict[str, float]

    @property
    def num_particles(self) -> tuple[int, int]:
        """Alias for ``sector`` (Qiskit Nature naming)."""
        return self.sector

    @property
    def energies(self) -> np.ndarray:
        """All cached eigenvalues, ascending: [ground, *excited]."""
        return np.concatenate(([self.ground_energy], self.excited_energies))


def compute_l3_reference(
    params: L3Params,
    *,
    sector: tuple[int, int] | None = None,
    num_particles: tuple[int, int] | None = None,
    k_states: int = 6,
    hamiltonian_override: FermionicOp | None = None,
) -> L3Reference:
    """Exact-diagonalize the L3 Hamiltonian in the (n_up, n_down) sector.

    Parameters
    ----------
    params
        L3 Hamiltonian parameters.
    sector, num_particles
        (n_up, n_down) particle counts. ``num_particles`` is an alias kept
        for Qiskit Nature compatibility. Exactly one may be specified; if
        neither is given, defaults to (9, 9). Specifying both raises
        ``ValueError``.
    k_states
        Number of lowest eigenpairs to compute via ``scipy.sparse.linalg.eigsh``.
    hamiltonian_override
        If provided, exact-diagonalize this operator instead of
        ``nio_l3_hamiltonian(params)``. Used by ``compute_l3_xas_reference``
        to share the sector-basis/observables machinery for H' = H + V_core.
    """
    if sector is not None and num_particles is not None:
        raise ValueError(
            "Pass at most one of `sector` or `num_particles` (they are aliases)."
        )
    if num_particles is not None:
        sector = num_particles
    if sector is None:
        sector = (9, 9)

    H = nio_l3_hamiltonian(params) if hamiltonian_override is None else hamiltonian_override
    basis = _build_sector_basis(
        num_d_spin_orbitals=params.num_spin_orbitals,
        n_up=sector[0],
        n_down=sector[1],
    )
    H_mat = _fermionic_op_to_sparse_matrix(H, basis)
    # Hermitize numerically to suppress sparse-build rounding.
    H_mat = 0.5 * (H_mat + H_mat.conj().T)

    energies, vectors = spla.eigsh(H_mat, k=k_states, which="SA")
    order = np.argsort(energies)
    energies = energies[order]
    vectors = vectors[:, order]

    # Populate reference observables.
    from siam_vqe.observables_l3 import compute_reference_observables
    partial = L3Reference(
        ground_energy=float(energies[0]),
        ground_vector=vectors[:, 0],
        excited_energies=energies[1:],
        sector_dim=len(basis),
        sector=sector,
        basis=tuple(basis),
        observables={},
    )
    observables = compute_reference_observables(partial)
    return L3Reference(
        ground_energy=partial.ground_energy,
        ground_vector=partial.ground_vector,
        excited_energies=partial.excited_energies,
        sector_dim=partial.sector_dim,
        sector=partial.sector,
        basis=partial.basis,
        observables=observables,
    )


def compute_l3_xas_reference(
    params: L3Params,
    ch: CoreHoleParams,
    *,
    num_particles: tuple[int, int] = (10, 9),
    k_states: int = 10,
) -> L3Reference:
    """scipy.sparse-ED reference for H' = H + V_core in the (n_up, n_dn)
    sector. Mirrors ``compute_l3_reference`` but on H' instead of H."""
    from siam_vqe.core_hole import nio_l3_core_hole_hamiltonian
    H_prime = nio_l3_core_hole_hamiltonian(params, ch)
    return compute_l3_reference(
        params,
        num_particles=num_particles,
        k_states=k_states,
        hamiltonian_override=H_prime,
    )


def save_l3_reference(ref: L3Reference, path: Path | str) -> None:
    """Save L3Reference to an NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ground_energy=ref.ground_energy,
        ground_vector=ref.ground_vector,
        excited_energies=ref.excited_energies,
        sector_dim=ref.sector_dim,
        sector=np.array(ref.sector, dtype=np.int64),
        basis=np.array(ref.basis, dtype=np.int64),
        observables_keys=np.array(list(ref.observables.keys()), dtype=object),
        observables_vals=np.array(list(ref.observables.values()), dtype=float),
    )


def load_l3_reference(path: Path | str) -> L3Reference:
    """Load L3Reference from an NPZ file."""
    data = np.load(path, allow_pickle=True)
    obs_keys = list(data["observables_keys"])
    obs_vals = list(data["observables_vals"])
    observables = {str(k): float(v) for k, v in zip(obs_keys, obs_vals, strict=False)}
    return L3Reference(
        ground_energy=float(data["ground_energy"]),
        ground_vector=data["ground_vector"],
        excited_energies=data["excited_energies"],
        sector_dim=int(data["sector_dim"]),
        sector=tuple(int(x) for x in data["sector"]),  # type: ignore[arg-type]
        basis=tuple(int(x) for x in data["basis"]),
        observables=observables,
    )


def inspect_multiplets(ref: L3Reference) -> list[dict[str, Any]]:
    """Return a list of dict rows summarising the low-lying spectrum.

    Each row: {'index', 'energy_eV', 'gap_eV'}. Used by Notebook 04 to
    print the L3 spectrum before ADAPT runs.
    """
    energies = np.concatenate(([ref.ground_energy], ref.excited_energies))
    return [
        {
            "index": i,
            "energy_eV": float(E),
            "gap_eV": float(E - ref.ground_energy),
        }
        for i, E in enumerate(energies)
    ]
