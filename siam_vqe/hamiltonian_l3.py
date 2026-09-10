"""L3 Hamiltonian: full Ni 3d + 5-orbital symmetry-adapted ligand bath.

Mode ordering (20 spin-orbitals):
    0..4   d orbitals (3z²−r², x²−y², xz, yz, xy), spin up
    5..9   bath orbitals (e_g a, e_g b, t_2g a, t_2g b, t_2g c), spin up
    10..14 d orbitals, spin down
    15..19 bath orbitals, spin down

Parameters from EDRIXS example_3 / Haverkort PRB 85, 165113 (2012).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from sympy.physics.wigner import gaunt


@dataclass(frozen=True)
class L3Params:
    """L3 Hamiltonian parameters.

    All energies in eV. Defaults are EDRIXS example_3 verbatim.
    """
    F2_dd: float = 9.787
    F4_dd: float = 6.078
    U_dd: float = 7.3
    ten_dq: float = 0.56          # impurity cubic CF splitting (e_g at +0.6, t_2g at -0.4)
    cf_bath_split: float = 1.44   # bath cubic CF splitting, same e_g/t_2g pattern
    Delta: float = 4.7            # charge-transfer energy
    V_eg: float = 2.06            # impurity-bath hybridization, e_g channel
    V_t2g: float = 1.21           # impurity-bath hybridization, t_2g channel
    num_d_orbitals: int = 5
    num_bath_orbitals: int = 5

    @property
    def num_spatial(self) -> int:
        return self.num_d_orbitals + self.num_bath_orbitals

    @property
    def num_spin_orbitals(self) -> int:
        return 2 * self.num_spatial


# Orbital symmetry labels (e_g vs t_2g). Cubic-harmonic ordering.
D_ORBITAL_SYMMETRIES: tuple[str, ...] = ("eg", "eg", "t2g", "t2g", "t2g")
D_ORBITAL_NAMES: tuple[str, ...] = ("3z2-r2", "x2-y2", "xz", "yz", "xy")


def racah_BC(F2: float, F4: float) -> tuple[float, float]:
    """Convert Slater-Condon F², F⁴ to Racah parameters B, C.

    Standard d-shell definitions (Cowan eq. 7.59 / STK eq. 4.78):
        B = (9 F² - 5 F⁴) / 441
        C = 35 F⁴ / 441
    """
    B = (9.0 * F2 - 5.0 * F4) / 441.0
    C = 35.0 * F4 / 441.0
    return B, C


def atomic_multiplet_energies(F2: float, F4: float) -> dict[str, float]:
    """Closed-form d⁸ atomic-LS-term energies (relative spacings).

    Returns energies for ³F, ¹D, ³P, ¹G, ¹S in eV, computed from the
    Racah parameterization (Sugano-Tanabe-Kamimura *Multiplets of
    Transition-Metal Ions in Crystals* (1970), Appendix Table I, d⁸
    column; equivalently Griffith *Theory of Transition Metal Ions*
    (1961) Table A21).

    The absolute zero is set by the ³F-relative table coefficients,
    i.e., E values are reported in Racah B, C with F⁰ (U_dd) excluded.
    """
    B, C = racah_BC(F2, F4)
    return {
        "3F": -8.0 * B + 3.0 * C,
        "1D": -3.0 * B + 5.0 * C,
        "3P": 7.0 * B + 3.0 * C,
        "1G": 4.0 * B + 5.0 * C,
        "1S": 14.0 * B + 10.0 * C,
    }


# Transformation matrix from real cubic harmonics (3z²−r², x²−y², xz, yz, xy)
# to complex spherical harmonics (Y_{2,m} for m = -2,-1,0,+1,+2).
#
# Standard convention (Condon-Shortley phase):
#   Y_{2,0} = 3z²−r² (up to radial)
#   Y_{2,±1} ∝ ∓(xz ± i yz)/√2
#   Y_{2,±2} ∝ (x²−y² ± 2i xy)/√2 (i.e., real cubic = combinations of m=±|m|)
#
# Resulting matrix M[α, m] s.t. |α_real⟩ = Σ_m M[α, m] |m_spherical⟩.
_REAL_TO_SPHERICAL = np.array([
    #     m=-2,           m=-1,           m=0,            m=+1,           m=+2
    [   0.0+0.0j,        0.0+0.0j,       1.0+0.0j,        0.0+0.0j,        0.0+0.0j],         # 3z²−r²
    [   1.0/np.sqrt(2),  0.0+0.0j,       0.0+0.0j,        0.0+0.0j,        1.0/np.sqrt(2)],   # x²−y²
    [   0.0+0.0j,       -1.0/np.sqrt(2), 0.0+0.0j,        1.0/np.sqrt(2),  0.0+0.0j],          # xz
    [   0.0+0.0j,        1.0j/np.sqrt(2),0.0+0.0j,        1.0j/np.sqrt(2), 0.0+0.0j],          # yz
    [  -1.0j/np.sqrt(2), 0.0+0.0j,       0.0+0.0j,        0.0+0.0j,        1.0j/np.sqrt(2)],   # xy
], dtype=complex)


def _gaunt_radial_factor(k: int) -> float:
    """4π / (2k+1) prefactor on the radial Slater integral in U expansion."""
    return 4.0 * np.pi / (2.0 * k + 1.0)


def slater_to_u_tensor(F2: float, F4: float, F0: float = 0.0) -> np.ndarray:
    """Build the 5×5×5×5 Coulomb tensor U[α,β,γ,δ] = ⟨αβ|γδ⟩ in the real
    cubic-harmonic d basis (physicists' notation).

    The physicists' two-electron integral is
        ⟨αβ|γδ⟩ = ∫ φ_α*(1) φ_β*(2) (1/r₁₂) φ_γ(1) φ_δ(2) d¹d²
    so that the Hamiltonian's two-body term is
        H₂ = ½ Σ U[α,β,γ,δ] c†_α c†_β c_δ c_γ.

    Internally the Gaunt sum is naturally written in chemists' notation
    (αβ|γδ) — same physics, different index order. After the Gaunt sum we
    store in physicists' order by swapping the middle two indices:
        ⟨αβ|γδ⟩ = (αγ|βδ)  ⇒  U_phys[α,β,γ,δ] = U_chem[α,γ,β,δ].
    The storage line below does the swap by writing into [i_a, i_c, i_b, i_d].
    """
    # Spherical-basis 4-index tensor by direct Gaunt-coefficient sum.
    # Indices below (i_a, i_b, i_c, i_d) correspond to chemists' (αβ|γδ);
    # storage swaps i_b ↔ i_c to land in physicists' ⟨αβ|γδ⟩.
    U_sph = np.zeros((5, 5, 5, 5), dtype=complex)
    F_k = {0: F0, 2: F2, 4: F4}

    m_values = [-2, -1, 0, 1, 2]

    for k, Fk in F_k.items():
        if Fk == 0.0 and k > 0:
            continue
        prefactor = _gaunt_radial_factor(k)
        for i_a, m_a in enumerate(m_values):
            for i_b, m_b in enumerate(m_values):
                for i_c, m_c in enumerate(m_values):
                    for i_d, m_d in enumerate(m_values):
                        # Chemists' angular conservation:
                        #   particle 1 transfers q = m_a - m_b; particle 2 absorbs -q
                        if (m_a - m_b) != (m_d - m_c):
                            continue
                        q = m_a - m_b
                        g1 = float(gaunt(2, k, 2, -m_a, q, m_b))
                        g2 = float(gaunt(2, k, 2, -m_c, m_c - m_d, m_d))
                        # Phase is (-1)^{m_α + m_δ} in BOTH chemists' (αβ|γδ)
                        # and physicists' ⟨αβ|γδ⟩, because the two Gaunt
                        # contributions correspond to c^k(α,·) c^k(δ,·) and the
                        # (-1)^{m'} factor in the relation
                        #   c^k(l',m'; l,m) = √(4π/(2k+1)) (-1)^{m'} gaunt(...)
                        # picks up m_α from the first Gaunt and m_δ from the
                        # second Gaunt. In the chemists'-indexed loop, this is
                        # m_{i_a} + m_{i_d}.
                        phase = (-1.0) ** (m_a + m_d)
                        # Storage in PHYSICISTS' order: U[α,β,γ,δ] = U_chem[α,γ,β,δ]
                        # → loop indices (i_a chem-α, i_b chem-β, i_c chem-γ, i_d chem-δ)
                        # → physicists' storage [i_a, i_c, i_b, i_d].
                        U_sph[i_a, i_c, i_b, i_d] += (
                            Fk * prefactor * phase * g1 * g2
                        )

    # Rotate from spherical to real cubic-harmonic basis.
    # All four contractions use M[real, spherical]; conjugates are placed on
    # bra indices (α and β in physicists' ⟨αβ|γδ⟩) and the kets get M without conj.
    # Equivalent einsum factorization yields a real array within numerical noise.
    M = _REAL_TO_SPHERICAL
    U_real = np.einsum("ai,bj,ck,dl,ijkl->abcd",
                       np.conj(M), np.conj(M), M, M, U_sph)

    if np.max(np.abs(U_real.imag)) > 1e-9:
        raise ValueError(
            f"U tensor has nonzero imaginary part: max|imag| = "
            f"{np.max(np.abs(U_real.imag))}"
        )
    u_out: np.ndarray = U_real.real
    return u_out


def nio_l3_impurity_only(params: L3Params, F0: float | None = None) -> FermionicOp:
    """Build the impurity-only L3 Hamiltonian on 10 d-spin-orbitals (5 orbital × 2 spin).

    H = ½ Σ_{αβγδ, σσ′} U[α,β,γ,δ] c†_{α σ} c†_{β σ′} c_{δ σ′} c_{γ σ}

    U is the 4-index Coulomb tensor in physicists' notation: U[α,β,γ,δ] = ⟨αβ|γδ⟩.
    The operator structure c†_α c†_β c_δ c_γ matches the physicists' convention.

    Used by the atomic-limit eigenvalue test (F2 mitigation gate) and as the
    impurity block of the full L3 Hamiltonian (Task 5).

    Mode ordering: 0..4 = d orbitals spin↑, 5..9 = d orbitals spin↓.
    """
    F0_val = params.U_dd if F0 is None else F0
    U = slater_to_u_tensor(params.F2_dd, params.F4_dd, F0_val)
    n_d = params.num_d_orbitals  # 5

    labels: dict[str, float] = {}
    # Two-body term: ½ Σ U[α,β,γ,δ] c†_{ασ} c†_{βσ'} c_{δσ'} c_{γσ}
    for alpha in range(n_d):
        for beta in range(n_d):
            for gamma in range(n_d):
                for delta in range(n_d):
                    u = U[alpha, beta, gamma, delta]
                    if abs(u) < 1e-12:
                        continue
                    for sigma in (0, 1):
                        for sigma_p in (0, 1):
                            a_mode = alpha + sigma * n_d
                            b_mode = beta + sigma_p * n_d
                            d_mode = delta + sigma_p * n_d
                            g_mode = gamma + sigma * n_d
                            if a_mode == b_mode:
                                continue  # c†_X c†_X = 0
                            if g_mode == d_mode:
                                continue  # c_X c_X = 0
                            key = f"+_{a_mode} +_{b_mode} -_{d_mode} -_{g_mode}"
                            labels[key] = labels.get(key, 0.0) + 0.5 * u

    return FermionicOp(labels, num_spin_orbitals=2 * n_d).simplify()


def _impurity_on_site_energies(params: L3Params) -> dict[int, float]:
    """ε_d(α) for each d orbital from charge-transfer + cubic CF.

    Cubic CF convention (10Dq positive): e_g at +0.6·10Dq, t_2g at -0.4·10Dq.
    The absolute zero is set so the orbital-averaged impurity energy sits the
    charge-transfer offset below the bath after Coulomb-double-counting.
    """
    base = -params.Delta + (params.num_d_orbitals - 1) * params.U_dd / 2.0
    eps = {}
    for alpha, sym in enumerate(D_ORBITAL_SYMMETRIES):
        if sym == "eg":
            eps[alpha] = base + 0.6 * params.ten_dq
        else:  # t2g
            eps[alpha] = base - 0.4 * params.ten_dq
    return eps


def _bath_on_site_energies(params: L3Params) -> dict[int, float]:
    """ε_p(β) for each bath orbital.

    Bath cubic CF mirrors the impurity (e_g vs t_2g). Indices: 0,1 = e_g (a,b),
    2,3,4 = t_2g (a,b,c). Reference energy = 0 (bath is absolute zero).
    """
    bath_symmetries = ("eg", "eg", "t2g", "t2g", "t2g")
    eps = {}
    for beta, sym in enumerate(bath_symmetries):
        if sym == "eg":
            eps[beta] = +0.6 * params.cf_bath_split
        else:
            eps[beta] = -0.4 * params.cf_bath_split
    return eps


def nio_l3_hamiltonian(params: L3Params) -> FermionicOp:
    """Build the full L3 Hamiltonian on 20 spin-orbitals.

    H = Σ_{α∈d, σ} ε_d(α) n_{dασ}
      + Σ_{β∈bath, σ} ε_p(β) n_{pβσ}
      + Σ_{α↔β, σ} V_(sym(α)) (d†_{ασ} p_{βσ} + h.c.)
      + ½ Σ_{αβγδ, σσ′} U[α,β,γ,δ] d†_{ασ} d†_{βσ′} d_{δσ′} d_{γσ}

    Mode indices (must match observables_l3 and tapering_l3):
        0..4   d orbitals spin↑
        5..9   bath orbitals spin↑
        10..14 d orbitals spin↓
        15..19 bath orbitals spin↓
    """
    n_d = params.num_d_orbitals
    n_b = params.num_bath_orbitals
    n_s = params.num_spatial  # 10

    labels: dict[str, float] = {}

    # 1. Impurity on-site + cubic CF.
    eps_d = _impurity_on_site_energies(params)
    for sigma in (0, 1):
        for alpha in range(n_d):
            mode = alpha + sigma * n_s
            labels[f"+_{mode} -_{mode}"] = labels.get(f"+_{mode} -_{mode}", 0.0) + eps_d[alpha]

    # 2. Bath on-site.
    eps_p = _bath_on_site_energies(params)
    for sigma in (0, 1):
        for beta in range(n_b):
            mode = n_d + beta + sigma * n_s
            labels[f"+_{mode} -_{mode}"] = labels.get(f"+_{mode} -_{mode}", 0.0) + eps_p[beta]

    # 3. Hybridization: d_α ↔ p_β paired by symmetry channel.
    #    d_eg(a) ↔ p_eg(a)  (α=0, β=0+n_d=5)
    #    d_eg(b) ↔ p_eg(b)  (α=1, β=1+n_d=6)
    #    d_t2g(a/b/c) ↔ p_t2g(a/b/c)  (α=2..4, β=7..9)
    for alpha in range(n_d):
        beta = alpha + n_d  # paired bath orbital
        V = params.V_eg if D_ORBITAL_SYMMETRIES[alpha] == "eg" else params.V_t2g
        for sigma in (0, 1):
            d_mode = alpha + sigma * n_s
            p_mode = beta + sigma * n_s
            labels[f"+_{d_mode} -_{p_mode}"] = labels.get(f"+_{d_mode} -_{p_mode}", 0.0) + V
            labels[f"+_{p_mode} -_{d_mode}"] = labels.get(f"+_{p_mode} -_{d_mode}", 0.0) + V

    # 4. Two-body d-d Coulomb (only acts on the 5 d orbitals).
    #    Use the Task-4-verified operator string:
    #        c†_{ασ} c†_{βσ'} c_{δσ'} c_{γσ}
    #    with α, γ sharing spin σ and β, δ sharing spin σ' (physicists' convention).
    U = slater_to_u_tensor(params.F2_dd, params.F4_dd, params.U_dd)
    for alpha in range(n_d):
        for beta in range(n_d):
            for gamma in range(n_d):
                for delta in range(n_d):
                    u = U[alpha, beta, gamma, delta]
                    if abs(u) < 1e-12:
                        continue
                    for sigma in (0, 1):
                        for sigma_p in (0, 1):
                            a_mode = alpha + sigma * n_s
                            b_mode = beta + sigma_p * n_s
                            d_mode = delta + sigma_p * n_s
                            g_mode = gamma + sigma * n_s
                            if a_mode == b_mode or g_mode == d_mode:
                                continue
                            key = f"+_{a_mode} +_{b_mode} -_{d_mode} -_{g_mode}"
                            labels[key] = labels.get(key, 0.0) + 0.5 * u

    return FermionicOp(labels, num_spin_orbitals=2 * n_s).simplify()
