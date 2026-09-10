"""L1 XAS pipeline — ψ_GS and ψ'_GS via run_vqe, statevector mode.

Pipeline-correctness smokes before Task 3c builds the L1 hardware driver:
    1. run_vqe on H_L1 reaches the (1,1)-sector ground state.
    2. run_vqe on H'_L1 = H_L1 + V_core (U_dc=8.5) reaches the (2,1)-sector
       ground state, which coincides with the global minimum of H'_L1
       (verified offline: see Task 3b implementation notes).

Both tests use the parity-tapered mapping so the UCCSD ansatz lives on 2
qubits and stays sector-locked to its Hartree-Fock initial state.
"""
from __future__ import annotations

import numpy as np


def test_l1_ground_state_via_run_vqe() -> None:
    """run_vqe on H_L1 (statevector, parity-tapered) matches scipy-ED ground
    energy of the (1,1) sector to <1e-6 eV."""
    from siam_vqe.ansatz import uccsd_ansatz
    from siam_vqe.hamiltonian import nio_l1_anderson
    from siam_vqe.mappings import to_qubit_op
    from siam_vqe.vqe_runner import run_vqe

    h_l1 = nio_l1_anderson(U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5)
    h_qop = to_qubit_op(h_l1, scheme="parity_tapered", num_particles=(1, 1))
    circuit, x0 = uccsd_ansatz(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        mapper_scheme="parity_tapered",
        reps=2,
    )

    result = run_vqe(
        h_qop, circuit, x0, optimizer="SLSQP", maxiter=200, seed=42
    )

    # Reference: lowest eigenvalue of the tapered Hamiltonian. Tapering
    # already projects onto the (1,1) sector, so this is the sector GS.
    e_ref = float(np.linalg.eigvalsh(h_qop.to_matrix()).min())
    assert abs(result.energy - e_ref) < 1e-6, (
        f"L1 VQE E={result.energy:.6f} vs ED E={e_ref:.6f}"
    )


def test_l1_h_prime_ground_state_via_run_vqe() -> None:
    """run_vqe on H'_L1 (statevector, parity-tapered) matches scipy-ED
    ground energy of the (2,1) sector to <50 meV.

    Sector choice: the (2,1) sector hosts the global minimum of H'_L1
    at U_dc=8.5 (verified by sector-resolved diagonalization of the
    JW-mapped H'; degenerate with (1,2) by SU(2) symmetry).

    Ansatz choice: UCCSD cannot be constructed for the (2,1) sector on
    2 spatial orbitals — the up-spin shell is fully filled, so the
    non-generalized UCCSD excitation pool is empty (qiskit-nature raises
    ValueError). We use EfficientSU2 on the 2-qubit tapered space
    instead. The tapering enforces the (2,1) sector, so the optimizer
    cannot escape it.
    """
    from siam_vqe.ansatz import efficient_su2_ansatz
    from siam_vqe.core_hole import CoreHoleParams, nio_l1_core_hole_hamiltonian
    from siam_vqe.mappings import to_qubit_op
    from siam_vqe.vqe_runner import run_vqe

    ch = CoreHoleParams(U_dc=8.5)
    h_prime = nio_l1_core_hole_hamiltonian(ch)
    hp_qop = to_qubit_op(h_prime, scheme="parity_tapered", num_particles=(2, 1))
    circuit, x0 = efficient_su2_ansatz(
        num_qubits=hp_qop.num_qubits, reps=2, seed=42
    )

    result = run_vqe(
        hp_qop, circuit, x0, optimizer="COBYLA", maxiter=600, seed=42
    )

    # Reference: lowest eigenvalue of the tapered H' restricted to (2,1).
    e_ref = float(np.linalg.eigvalsh(hp_qop.to_matrix()).min())
    assert abs(result.energy - e_ref) < 0.05, (
        f"L1 H' VQE E={result.energy:.6f} vs ED E={e_ref:.6f}"
    )
