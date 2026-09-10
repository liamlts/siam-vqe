"""Tests for siam_vqe.dipole."""
from __future__ import annotations

import numpy as np
import pytest

from siam_vqe.dipole import (
    DipoleChannel,
    angular_coefficients,
    dipole_operator,
)


def test_angular_coefficients_lin_z_only_3z2():
    """Linear-z polarization couples 2p_z → 3d_{3z²-r²} only."""
    coeffs = angular_coefficients("lin_z")
    # d-orbital index ordering: 0=3z2, 1=x2-y2, 2=xz, 3=yz, 4=xy
    assert coeffs[0] != 0
    for i in (1, 2, 3, 4):
        assert coeffs[i] == 0


def test_angular_coefficients_lin_xy_includes_t2g_pair():
    """Linear-xy powder-averaged channel couples to (x²-y², xz, yz, xy)
    — anything with nonzero in-plane character."""
    coeffs = angular_coefficients("lin_xy")
    # 3z² has no in-plane character.
    assert coeffs[0] == 0
    # Other d-orbitals are nonzero (xz, yz, xy by symmetry; x²-y² has
    # x²+y² which contributes in-plane).
    for i in (1, 2, 3, 4):
        assert coeffs[i] != 0


def test_angular_coefficients_unknown_channel_raises():
    with pytest.raises(ValueError, match="Unknown dipole channel"):
        angular_coefficients("circular_+")


def test_dipole_channel_dataclass():
    """DipoleChannel is frozen and carries name + coeffs."""
    ch = DipoleChannel(name="lin_z", coefficients=np.array([1.0, 0, 0, 0, 0]))
    with pytest.raises((AttributeError, Exception)):
        ch.name = "lin_xy"


def test_dipole_anticommutator_is_sum_w_squared():
    """{D, D†} = D D† + D† D = (Σ_α |w_α|²) × I on the spin-doubled basis.

    For our XAS convention D = Σ_α w_α (c†_{d_α↑} + c†_{d_α↓}) the canonical
    anticommutator {c_i, c†_j} = δ_{ij} gives

        D D† + D† D = Σ_α (|w_α|² + |w_α|²) × I = 2 Σ_α |w_α|² × I,

    where the factor 2 accounts for the two spin modes per orbital. This
    locks Σ_α |w_α|² exactly. For lin_z that sum is 1 (only 3z²-r² with
    w=1 → total 2 with spin). For lin_xy it is (3/4 + 3·1/2) = 9/4 → 9/2
    with spin.

    Operator-level equality is hard to check in qiskit_nature without
    normal-ordering, so we verify the constant by evaluating ⟨D D† + D† D⟩
    on the Phase 4 (9, 9) ground state and on a random sector vector —
    both must give the scalar. Two independent vectors is sufficient to
    rule out ψ-dependent off-diagonal terms.
    """
    from pathlib import Path

    from siam_vqe.dipole import dipole_channel
    from siam_vqe.hamiltonian_l3 import L3Params
    from siam_vqe.observables_l3 import evaluate_observable_on_sector
    from siam_vqe.reference_l3 import load_l3_reference

    params = L3Params()
    ref_path = Path(__file__).resolve().parent.parent / "data" / "nio_l3_reference.npz"
    if not ref_path.exists():
        pytest.skip("Phase 4 reference NPZ not committed in this worktree.")
    ref = load_l3_reference(ref_path)

    for channel_name in ("lin_z", "lin_xy"):
        ch = dipole_channel(channel_name)
        expected = 2.0 * float(np.sum(ch.coefficients**2))  # x2 for spin doubling

        D = dipole_operator(ch, params=params)
        anti = (D @ D.adjoint() + D.adjoint() @ D).simplify()

        # ⟨ψ|anti|ψ⟩ on the GS
        val_gs = evaluate_observable_on_sector(
            anti, ref.ground_vector, basis=ref.basis,
            num_spin_orbitals=params.num_spin_orbitals,
        )
        # ⟨ψ|anti|ψ⟩ on a random unit vector in the same sector
        rng = np.random.default_rng(0)
        psi = rng.normal(size=ref.sector_dim) + 1j * rng.normal(size=ref.sector_dim)
        psi /= np.linalg.norm(psi)
        val_rand = evaluate_observable_on_sector(
            anti, psi, basis=ref.basis,
            num_spin_orbitals=params.num_spin_orbitals,
        )

        assert abs(val_gs - expected) < 1e-9, (
            f"{channel_name}: ⟨GS|{{D,D†}}|GS⟩ = {val_gs}, expected {expected}"
        )
        assert abs(val_rand - expected) < 1e-9, (
            f"{channel_name}: ⟨rand|{{D,D†}}|rand⟩ = {val_rand}, expected {expected}"
        )


def test_dipole_lin_z_gs_sum_rule():
    """Lehmann sum-rule gate for XAS absorption from the Phase 4 d⁸ GS.

    With D = Σ_α w_α c†_{d_α} (electron-creation, modelling the 2p→3d
    transition with the 2p spectator factored out), the absorption sum
    rule is

        Σ_f |⟨f|D|GS⟩|² = ⟨GS|D†D|GS⟩
                       = Σ_α |w_α|² (2 - ⟨n_{d_α}⟩) + spin-flip cross terms,

    where ⟨n_{d_α}⟩ = ⟨n_↑ + n_↓⟩ is spin-summed. On the (9, 9) GS the
    spin-flip cross terms ⟨c†_{α↑} c_{α↓}⟩ vanish (S_z eigenstate), so the
    sum rule reduces to Σ_α |w_α|² (2 - ⟨n_{d_α}⟩).

    For lin_z (w_{3z²} = 1, others zero) and Phase 4 ⟨n_{d,3z²}⟩ ≈ 1.001:

        ⟨D†D⟩ ≈ 1 × (2 - 1.001) ≈ 0.999.

    This is the XAS "empty-DOS" contribution from the 3z²-r² channel,
    which is correct physics: a half-occupied e_g orbital contributes
    half a unit of absorption per spin. If this gate fires, suspect the
    dipole angular coefficients or the Phase 4 ⟨n_d⟩ rather than the test
    bounds.

    NOTE: A previous draft of this test expected ⟨D†D⟩ ≈ 2.002 by
    treating D as an annihilation operator (so D†D → number operator).
    With the XAS convention D = c† used in dipole.py, D†D reduces to the
    *hole* density (2 − n), not the electron density n; and Phase 4's
    n_{3z²} = 1.001 is the spin-summed total, not per spin block. The
    correct sum-rule value is ~0.999 (XAS sees empty states).
    """
    from pathlib import Path

    from siam_vqe.dipole import dipole_channel
    from siam_vqe.hamiltonian_l3 import L3Params
    from siam_vqe.observables_l3 import evaluate_observable_on_sector
    from siam_vqe.reference_l3 import load_l3_reference

    params = L3Params()
    ref_path = Path(__file__).resolve().parent.parent / "data" / "nio_l3_reference.npz"
    if not ref_path.exists():
        pytest.skip("Phase 4 reference NPZ not committed in this worktree.")
    ref = load_l3_reference(ref_path)

    D = dipole_operator(dipole_channel("lin_z"), params=params)
    DdD = (D.adjoint() @ D).simplify()

    sum_rule = evaluate_observable_on_sector(
        DdD, ref.ground_vector, basis=ref.basis,
        num_spin_orbitals=params.num_spin_orbitals,
    )

    # Closed-form expected value from Phase 4 ⟨n_{d,3z²}⟩:
    n_3z2 = ref.observables["n_d_3z2"]
    expected = 2.0 - n_3z2  # Σ|w|²(2 - n_α), w only on 3z² with w=1

    assert abs(sum_rule - expected) < 1e-6, (
        f"sum-rule = {sum_rule}, closed-form expected = {expected}"
    )
    # Physical window: d⁸ with nearly half-filled e_g gives ~1.0.
    assert 0.95 < sum_rule < 1.05, f"sum-rule = {sum_rule} outside [0.95, 1.05]"
