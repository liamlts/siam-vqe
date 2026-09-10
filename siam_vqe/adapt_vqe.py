"""Operator-ADAPT-VQE: pool construction + gradient screening + outer loop.

Reference: Grimsley, Economou, Barnes, Mayhall, "An adaptive variational
algorithm for exact molecular simulations on a quantum computer," Nat.
Commun. 10, 3007 (2019).

This is an educational implementation per the parent spec — it is NOT a
research-grade ADAPT-VQE. Pool is UCCSD-flavored fermionic (occupied →
virtual), particle- and S_z-conserving.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

from siam_vqe.tapering_l3 import project_full_to_tapered, tapered_l3_pauli

_N_SPATIAL_L3 = 10


def _is_up_spin(mode: int, num_spin_orbitals: int) -> bool:
    return mode < num_spin_orbitals // 2


def _make_single_excitation(p: int, q: int, num_spin_orbitals: int) -> FermionicOp:
    """T = c†_p c_q − c†_q c_p."""
    return FermionicOp(
        {f"+_{p} -_{q}": 1.0, f"+_{q} -_{p}": -1.0},
        num_spin_orbitals=num_spin_orbitals,
    ).simplify()


def _make_double_excitation(p: int, q: int, r: int, s: int,
                             num_spin_orbitals: int) -> FermionicOp:
    """T = c†_p c†_q c_s c_r − c†_r c†_s c_q c_p (anti-Hermitian)."""
    labels = {
        f"+_{p} +_{q} -_{s} -_{r}": 1.0,
        f"+_{r} +_{s} -_{q} -_{p}": -1.0,
    }
    return FermionicOp(labels, num_spin_orbitals=num_spin_orbitals).simplify()


def build_uccsd_pool(
    *,
    num_spin_orbitals: int,
    occupied: Sequence[int],
    virtual: Sequence[int],
    include_doubles_same_spin: bool = True,
    include_doubles_mixed_spin: bool = True,
    include_singles: bool = True,
) -> list[FermionicOp]:
    """Build a UCCSD-like fermionic pool.

    Particle-conserving (each generator moves one or two particles from
    occupied to virtual) and S_z-conserving (no spin-changing generators).

    Parameters
    ----------
    num_spin_orbitals : int
    occupied : sequence of mode indices in the reference Slater determinant.
    virtual : sequence of mode indices outside the reference.
    """
    pool: list[FermionicOp] = []
    occ = list(occupied)
    virt = list(virtual)

    # Singles: c†_a c_i − h.c., spin-conserving.
    if include_singles:
        for i in occ:
            for a in virt:
                if _is_up_spin(i, num_spin_orbitals) != _is_up_spin(a, num_spin_orbitals):
                    continue
                pool.append(_make_single_excitation(a, i, num_spin_orbitals))

    # Doubles: c†_a c†_b c_j c_i − h.c., S_z-conserving.
    if include_doubles_mixed_spin or include_doubles_same_spin:
        for idx_i, i in enumerate(occ):
            for j in occ[idx_i + 1:]:
                spin_i = _is_up_spin(i, num_spin_orbitals)
                spin_j = _is_up_spin(j, num_spin_orbitals)
                for idx_a, a in enumerate(virt):
                    for b in virt[idx_a + 1:]:
                        spin_a = _is_up_spin(a, num_spin_orbitals)
                        spin_b = _is_up_spin(b, num_spin_orbitals)
                        if sorted([spin_i, spin_j]) != sorted([spin_a, spin_b]):
                            continue
                        same_spin = (spin_i == spin_j)
                        if same_spin and not include_doubles_same_spin:
                            continue
                        if not same_spin and not include_doubles_mixed_spin:
                            continue
                        pool.append(_make_double_excitation(a, b, i, j,
                                                              num_spin_orbitals))

    return pool


def pool_to_tapered_paulis(
    pool: Iterable[FermionicOp],
    num_particles: tuple[int, int] = (9, 9),
) -> list[SparsePauliOp]:
    """Convert each pool generator to its tapered Pauli representation.

    Each fermionic generator T = c†c − h.c. is anti-Hermitian; under JW +
    parity tapering it remains anti-Hermitian. We multiply by `i` so that iT
    is Hermitian (downstream commutator math `[H, iT]` returns real-valued
    expectation values).
    """
    tapered: list[SparsePauliOp] = []
    for T in pool:
        P = tapered_l3_pauli(T, num_particles=num_particles)
        P = P * 1j
        tapered.append(P.simplify())
    return tapered


@dataclass(frozen=True)
class HFState:
    """A Slater-determinant Hartree-Fock initial state."""
    occupied: tuple[int, ...]   # sorted mode indices
    num_spin_orbitals: int
    num_particles: tuple[int, int]


def hartree_fock_initial_state(
    *,
    num_spin_orbitals: int = 20,
    num_particles: tuple[int, int] = (9, 9),
    swap_pairs: Sequence[tuple[int, int]] | None = None,
) -> HFState:
    """Build a Slater-determinant HF state in the (n_up, n_down) sector.

    Default: lowest-`n_up` up-spin modes + lowest-`n_down` down-spin modes
    occupied (closed-shell-like). With `swap_pairs=[(i, j), ...]`, each
    listed swap replaces mode i (currently occupied) with mode j (currently
    virtual) in the occupation; both must have the same spin block.

    F5 mitigation: an explicit sector check is performed before return.
    """
    half = num_spin_orbitals // 2
    n_up, n_dn = num_particles
    occ_up = set(range(n_up))
    occ_dn = set(range(half, half + n_dn))
    occupied = occ_up | occ_dn

    if swap_pairs:
        for (i, j) in swap_pairs:
            if i not in occupied:
                raise ValueError(f"Mode {i} not occupied; cannot swap.")
            if j in occupied:
                raise ValueError(f"Mode {j} already occupied; cannot swap.")
            # Same spin block: both in [0, half) or both in [half, 2*half)
            i_up = i < half
            j_up = j < half
            if i_up != j_up:
                raise ValueError(
                    f"Swap ({i}, {j}) crosses spin boundary; would change S_z."
                )
            occupied.remove(i)
            occupied.add(j)

    # F5 mitigation: explicit sector check.
    actual_up = sum(1 for m in occupied if m < half)
    actual_dn = sum(1 for m in occupied if m >= half)
    if (actual_up, actual_dn) != (n_up, n_dn):
        raise ValueError(
            f"HF state landed in sector ({actual_up}, {actual_dn}), "
            f"expected {num_particles}. F5 mitigation fired."
        )

    return HFState(
        occupied=tuple(sorted(occupied)),
        num_spin_orbitals=num_spin_orbitals,
        num_particles=num_particles,
    )


def hf_state_to_tapered_statevector(hf: HFState) -> np.ndarray:
    """Convert a Slater-determinant HF state to its 18-qubit tapered statevector."""
    # Bare JW: occupation-number basis index = sum of 2^mode for each occupied mode.
    occ_int = sum(1 << m for m in hf.occupied)
    psi_full = np.zeros(2**hf.num_spin_orbitals, dtype=complex)
    psi_full[occ_int] = 1.0
    psi_tapered = project_full_to_tapered(psi_full, num_particles=hf.num_particles)
    # Tapering is unitary on the sector subspace; if norm differs from 1, that's a bug.
    norm = np.linalg.norm(psi_tapered)
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(
            f"Tapered HF statevector has norm {norm}, expected 1.0. "
            f"Likely the chosen HF determinant is not in the requested sector "
            f"or the parity-tapering convention is misaligned."
        )
    return psi_tapered / norm


def screen_gradients(
    H_pauli: SparsePauliOp,
    pool_paulis: Sequence[SparsePauliOp],
    psi: np.ndarray,
) -> np.ndarray:
    """Compute |⟨ψ|[H, iT_k]|ψ⟩| for every pool operator T_k.

    All inputs are in the tapered space. Each iT_k is Hermitian (per
    `pool_to_tapered_paulis`), so the commutator with H is also Hermitian
    and the expectation value is real.

    Returns shape (len(pool_paulis),), dtype float.
    """
    H_mat = H_pauli.to_matrix(sparse=True)
    H_psi = H_mat @ psi  # cached: only one matvec with H

    grads = np.zeros(len(pool_paulis), dtype=float)
    for k, T_pauli in enumerate(pool_paulis):
        T_mat = T_pauli.to_matrix(sparse=True)
        T_psi = T_mat @ psi
        # ⟨ψ| H T |ψ⟩
        a = np.vdot(psi, H_mat @ T_psi)
        # ⟨ψ| T H |ψ⟩
        b = np.vdot(psi, T_mat @ H_psi)
        # ⟨[H, T]⟩ = a - b. T_pauli is Hermitian (pool_to_tapered_paulis
        # multiplies by i so iT is Hermitian), so [H, T] is anti-Hermitian
        # and ⟨[H, T]⟩ is pure imaginary. The energy gradient at θ=0 for
        # U(θ) = exp(iθ T) is i⟨[H, T]⟩, real-valued. Magnitude = |⟨[H, T]⟩|.
        grads[k] = float(np.abs(a - b))
    return grads


def apply_exp_iT(
    psi: np.ndarray,
    operators: Sequence[SparsePauliOp],
    parameters: Sequence[float],
) -> np.ndarray:
    """Apply U(θ) = exp(θ_n · iT_n) ... exp(θ_1 · iT_1) to |ψ⟩.

    Each T_k is the i-times-anti-Hermitian Pauli built by
    `pool_to_tapered_paulis` (so T_k is Hermitian and exp(θ · iT_k) is unitary).
    """
    if len(operators) != len(parameters):
        raise ValueError("operators and parameters must have equal length")
    psi_out = psi.copy()
    for T_pauli, theta in zip(operators, parameters, strict=True):
        if theta == 0.0:
            continue
        # exp(θ · i T_pauli) where T_pauli is i × (anti-Hermitian fermionic).
        # T_pauli itself was multiplied by i in `pool_to_tapered_paulis` to be
        # Hermitian. So we apply exp(θ · 1j · T_pauli.to_matrix()).
        T_mat = T_pauli.to_matrix(sparse=True)
        psi_out = expm_multiply(1j * theta * T_mat, psi_out)
    return psi_out


@dataclass(frozen=True)
class AdaptConfig:
    """Configuration for the ADAPT outer loop.

    Defaults match spec §5.5.
    """
    gradient_threshold: float = 1e-4   # eV
    max_operators: int = 30
    inner_optimizer: str = "cobyla"
    inner_max_iter: int = 500
    inner_rhobeg: float = 0.1
    inner_rhoend: float = 1e-6


@dataclass(frozen=True)
class AdaptResult:
    """Result of one ADAPT-VQE run."""
    final_energy: float
    theta: np.ndarray
    operators_picked: tuple[int, ...]
    trace: tuple[dict, ...]
    converged_reason: str  # 'gradient', 'max_operators', or 'max_outer'


def _build_energy_callable(
    H_mat,
    pool_paulis: Sequence[SparsePauliOp],
    operators_picked: Sequence[int],
    psi_0: np.ndarray,
):
    """Return a callable θ → ⟨ψ_0| U†(θ) H U(θ) |ψ_0⟩."""
    selected = [pool_paulis[k] for k in operators_picked]

    def energy(theta_vec):
        psi = apply_exp_iT(psi_0, selected, theta_vec)
        return float(np.real(np.vdot(psi, H_mat @ psi)))

    return energy


def run_adapt_vqe(
    H_pauli: SparsePauliOp,
    pool_paulis: Sequence[SparsePauliOp],
    psi_0: np.ndarray,
    config: AdaptConfig,
) -> AdaptResult:
    """Outer-loop ADAPT-VQE.

    Pseudocode (spec §5.2):
        repeat:
            grads = [|⟨ψ|[H, iT_k]|ψ⟩| for T_k in pool]
            if max(grads) < gradient_threshold: break (gradient convergence)
            if len(operators_picked) >= max_operators: break (depth budget)
            k* = argmax(grads); append k*; θ.append(0.0)
            θ ← scipy.minimize(energy(θ), x0=θ)
            ψ ← U(θ)|ψ_0⟩
        return AdaptResult
    """
    H_mat = H_pauli.to_matrix(sparse=True)
    psi = psi_0.copy()
    theta: list[float] = []
    operators_picked: list[int] = []
    trace: list[dict] = []

    converged_reason = "max_outer"
    for outer in range(config.max_operators + 1):
        grads = screen_gradients(H_pauli, pool_paulis, psi)
        g_max = float(np.max(grads))
        k_star = int(np.argmax(grads))

        # Stopping checks (BEFORE appending).
        if g_max < config.gradient_threshold:
            converged_reason = "gradient"
            trace.append({
                "iteration": outer,
                "energy": float(np.real(np.vdot(psi, H_mat @ psi))),
                "g_max": g_max,
                "k_star": -1,
                "n_operators": len(operators_picked),
                "event": "converged",
            })
            break
        if len(operators_picked) >= config.max_operators:
            converged_reason = "max_operators"
            trace.append({
                "iteration": outer,
                "energy": float(np.real(np.vdot(psi, H_mat @ psi))),
                "g_max": g_max,
                "k_star": -1,
                "n_operators": len(operators_picked),
                "event": "budget_exhausted",
            })
            break

        # Append + re-optimize.
        operators_picked.append(k_star)
        theta.append(0.0)
        energy = _build_energy_callable(H_mat, pool_paulis, operators_picked, psi_0)
        scipy_opts = {
            "maxiter": config.inner_max_iter,
            "rhobeg": config.inner_rhobeg,
            "tol": config.inner_rhoend,
        } if config.inner_optimizer.lower() == "cobyla" else {
            "maxiter": config.inner_max_iter,
        }
        result = minimize(energy, x0=np.array(theta), method=config.inner_optimizer.upper(),
                          options=scipy_opts)
        theta = list(result.x)
        psi = apply_exp_iT(psi_0,
                           [pool_paulis[k] for k in operators_picked],
                           theta)

        trace.append({
            "iteration": outer,
            "energy": float(result.fun),
            "g_max": g_max,
            "k_star": k_star,
            "n_operators": len(operators_picked),
            "event": "operator_added",
        })

    final_energy = float(np.real(np.vdot(psi, H_mat @ psi)))
    return AdaptResult(
        final_energy=final_energy,
        theta=np.array(theta),
        operators_picked=tuple(operators_picked),
        trace=tuple(trace),
        converged_reason=converged_reason,
    )


def build_l3_multistart_seeds(
    *,
    num_seeds: int = 4,
    num_spin_orbitals: int = 20,
    num_particles: tuple[int, int] = (9, 9),
) -> list[HFState]:
    """Build perturbed HF Slater determinants for the L3 ADAPT-VQE multistart.

    L3 mode ordering (hamiltonian_l3.py):
        0-4   d orbitals ↑ (eg, eg, t2g, t2g, t2g)
        5-9   bath orbitals ↑
        10-14 d orbitals ↓
        15-19 bath orbitals ↓

    The physical ground state is d⁸ + bath¹⁰ (octahedral ³A_2g). With the
    EDRIXS example_3 charge-transfer parameters (Δ = 4.7 eV, U_dd = 7.3 eV),
    the impurity d levels lie ~10 eV ABOVE the bath. So
    `hartree_fock_initial_state(...)` — which fills the lowest-INDEX modes —
    lands in d¹⁰ + bath⁸, ~20 eV above the true ground state. With only one
    virtual per spin in the (9, 9) sector, ADAPT-VQE cannot tunnel out.
    Seeds must start in d⁸ directly.

    Seed 0 (³A_2g M_S=0 sublevel):
        up: e_g #1 (mode 0) + full t_2g (2,3,4) + full bath (5-9)
        dn: e_g #2 (mode 11) + full t_2g (12,13,14) + full bath (15-19)

    Seeds 1-3 perturb the e_g singly-occupied labels (the only virtuals are
    mode 1 ↑ and mode 10 ↓). Seeds 4+ promote a t_2g electron to the e_g
    virtual.
    """
    half = num_spin_orbitals // 2  # 10

    base_occupied = {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19}
    eg_virtual_up = 1
    eg_virtual_dn = 10
    t2g_up = (2, 3, 4)
    t2g_dn = (12, 13, 14)

    seed_swaps: list[list[tuple[int, int]]] = [
        [],
        [(0, eg_virtual_up)],
        [(11, eg_virtual_dn)],
        [(0, eg_virtual_up), (11, eg_virtual_dn)],
    ]
    while len(seed_swaps) < num_seeds:
        k = len(seed_swaps) - 4
        # Alternate t_2g↑→eg↑ and t_2g↓→eg↓ promotions.
        if k % 2 == 0:
            seed_swaps.append([(t2g_up[(k // 2) % 3], eg_virtual_up)])
        else:
            seed_swaps.append([(t2g_dn[(k // 2) % 3], eg_virtual_dn)])

    seeds: list[HFState] = []
    for swaps in seed_swaps[:num_seeds]:
        occupied = set(base_occupied)
        for (i, j) in swaps:
            if i not in occupied:
                raise ValueError(f"Mode {i} not occupied in d⁸ base seed.")
            if j in occupied:
                raise ValueError(f"Mode {j} already occupied in d⁸ base seed.")
            if (i < half) != (j < half):
                raise ValueError(f"Swap ({i}, {j}) crosses spin boundary.")
            occupied.remove(i)
            occupied.add(j)
        actual_up = sum(1 for m in occupied if m < half)
        actual_dn = sum(1 for m in occupied if m >= half)
        if (actual_up, actual_dn) != num_particles:
            raise ValueError(
                f"Seed landed in sector ({actual_up}, {actual_dn}), "
                f"expected {num_particles}."
            )
        seeds.append(
            HFState(
                occupied=tuple(sorted(occupied)),
                num_spin_orbitals=num_spin_orbitals,
                num_particles=num_particles,
            )
        )
    return seeds
