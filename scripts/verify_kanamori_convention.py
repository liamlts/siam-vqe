"""Compare e_g² impurity multiplet energies under Pavarini and STK conventions.

Pure script: builds the impurity-only block of the Kanamori Hamiltonian (V=0,
bath isolated), diagonalizes the (2 electrons in 2 e_g orbitals) sector, and
checks which J convention reproduces the Tanabe-Sugano gaps for ³A₂g, ¹E_g, ¹A₁g.

Run once during Phase 3 task 1; the result is committed as a markdown decision
record and the chosen J is hardcoded in reference_edrixs.compute_l2_levels().
"""

from __future__ import annotations

import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp


F2 = 9.787  # eV
F4 = 6.078  # eV

# Slater integrals → Pavarini averaged params
U_avg = 7.3  # = F0 (from EDRIXS example_3)
U_eg_pavarini = U_avg + 4 * F2 / 49 + 36 * F4 / 441
J_pavarini = 3 * F2 / 49 + 20 * F4 / 441

# Slater integrals → STK Racah-derived params
B = (9 * F2 - 5 * F4) / 441
C = 5 * F4 / 63
A = U_avg - 49 * F4 / 441
U_eg_stk = A + 4 * B + 3 * C       # ³A₂g + 8B = singlet intra; intra-orbital U
J_stk = 4 * B + C                  # e_g-specific Hund

print(f"Pavarini:    U_eg = {U_eg_pavarini:.4f} eV, J_H = {J_pavarini:.4f} eV")
print(f"STK Racah:   U_eg = {U_eg_stk:.4f} eV, J_eg = {J_stk:.4f} eV")
print(f"Racah B,C:   B = {B:.4f}, C = {C:.4f}")

# Tanabe-Sugano gaps (analytic, independent of which J convention we choose)
ts_gap_3E_to_3A2 = 2 * C            # E(¹E_g) - E(³A₂g)
ts_gap_1A1_to_3A2 = 16 * B + 4 * C  # E(¹A₁g) - E(³A₂g)
print(f"\nTanabe-Sugano analytic gaps:")
print(f"  E(¹E_g) - E(³A₂g) = 2C = {ts_gap_3E_to_3A2:.4f} eV")
print(f"  E(¹A₁g) - E(³A₂g) = 16B + 4C = {ts_gap_1A1_to_3A2:.4f} eV")


def build_eg_impurity_hamiltonian(U: float, U_prime: float, J: float) -> FermionicOp:
    """Build the e_g² impurity Kanamori block on 4 spin-orbitals.

    Mode ordering: 0=eg_a↑, 1=eg_b↑, 2=eg_a↓, 3=eg_b↓.
    """
    labels: dict[str, float] = {
        # Intra-orbital U (same orbital, opposite spin)
        "+_0 -_0 +_2 -_2": U,
        "+_1 -_1 +_3 -_3": U,
        # Inter-orbital, opposite spin: U'
        "+_0 -_0 +_3 -_3": U_prime,
        "+_1 -_1 +_2 -_2": U_prime,
        # Inter-orbital, same spin: U' - J
        "+_0 -_0 +_1 -_1": U_prime - J,
        "+_2 -_2 +_3 -_3": U_prime - J,
        # Spin-flip: -J Σ_{α≠β} d†_{α↑} d_{α↓} d†_{β↓} d_{β↑}
        "+_0 -_2 +_3 -_1": -J,
        "+_1 -_3 +_2 -_0": -J,
        # Pair-hop: +J Σ_{α≠β} d†_{α↑} d†_{α↓} d_{β↓} d_{β↑}
        # α=a (0,2), β=b (1,3): +_0 +_2 -_3 -_1
        # α=b (1,3), β=a (0,2): +_1 +_3 -_2 -_0
        "+_0 +_2 -_3 -_1": J,
        "+_1 +_3 -_2 -_0": J,
    }
    return FermionicOp(labels, num_spin_orbitals=4)


def diag_e_g_2electron_sector(U: float, U_prime: float, J: float) -> np.ndarray:
    """Return the 6 lowest eigenvalues in the 2-electron sector of the e_g² impurity."""
    fop = build_eg_impurity_hamiltonian(U, U_prime, J)
    h = JordanWignerMapper().map(fop).to_matrix()
    eigvals = np.linalg.eigvalsh(h)
    # Filter to 2-electron sector by total particle number = 2.
    # Build N operator: Σ n_i
    n_op = FermionicOp(
        {f"+_{i} -_{i}": 1.0 for i in range(4)}, num_spin_orbitals=4
    )
    n_mat = JordanWignerMapper().map(n_op).to_matrix()
    # Diagonalize H and compute N expectation in each eigenstate.
    _, eigvecs = np.linalg.eigh(h)
    n_vals = np.real(np.einsum("ij,jk,ki->i", eigvecs.conj().T, n_mat, eigvecs))
    two_electron_mask = np.isclose(n_vals, 2.0, atol=1e-6)
    return eigvals[two_electron_mask]


for label, J in [("Pavarini", J_pavarini), ("STK Racah", J_stk)]:
    U = U_eg_pavarini if label == "Pavarini" else U_eg_stk
    U_prime = U - 2 * J
    eigvals = diag_e_g_2electron_sector(U, U_prime, J)
    print(f"\n{label} (U={U:.3f}, U'={U_prime:.3f}, J={J:.3f}):")
    print(f"  e_g² 2-electron spectrum: {np.round(eigvals - eigvals[0], 4)}")
    if len(eigvals) >= 6:
        gap_excited_singlet = eigvals[3] - eigvals[0]  # ³A₂g → ¹E_g (heuristic ordering)
        gap_high_singlet = eigvals[5] - eigvals[0]      # ³A₂g → ¹A₁g
        print(f"  Implied 3A2g → 1E_g gap:  {gap_excited_singlet:.4f} eV (TS analytic: {ts_gap_3E_to_3A2:.4f})")
        print(f"  Implied 3A2g → 1A1g gap: {gap_high_singlet:.4f} eV (TS analytic: {ts_gap_1A1_to_3A2:.4f})")
