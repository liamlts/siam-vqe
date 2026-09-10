#!/usr/bin/env python
"""Path-A EDRIXS XAS reference for the L3 NiO SIAM (Phase 5).

The Phase 5 closeout flagged the current `data/edrixs_xas_l3_reference.npz`
as "Path B" — a self-consistent scipy-ED reference. This script is the
"Path A" replacement: it drives the EDRIXS library (`edrixs.ed_siam_fort` +
`edrixs.xas_siam_fort`) on the SAME L3 NiO parameters used in Phase 5 and
saves an NPZ in the same format.

Validation: after running this script, re-run `scripts/run_l3_xas_sim.py`
against the Path-A reference and confirm phase5_pass remains True. Any
disagreement above L2 ~ 5% reflects either a convention mismatch
(polarization axis, sign of hybridization, bath e_g/t_2g pairing) or a
physical channel we missed in the Phase 5 dipole construction.

# Why this isn't on the main host Python

EDRIXS bundles a Fortran solver and MPI parallelism via `mpi4py`. Installing
it natively against MacPorts Python 3.13 is brittle; the user's setup is a
Docker container called `edrixs_run`. This script is designed to run INSIDE
that container.

# How to invoke (Docker)

From the host (worktree mounted into /work):

    docker run --rm -v $(pwd):/work \
        edrixs_run \
        bash -c "cd /work/siam_vqe && python scripts/build_edrixs_xas_reference_path_a.py"

Or interactive (recommended on first run, so you can adjust polarization
syntax if EDRIXS rejects the strings used below):

    docker run --rm -it -v $(pwd):/work \
        edrixs_run bash
    # then inside the container:
    cd /work/siam_vqe
    python scripts/build_edrixs_xas_reference_path_a.py

Output: `data/edrixs_xas_l3_reference_path_a.npz` (does NOT overwrite the
Path-B reference at `data/edrixs_xas_l3_reference.npz`). After validation,
the user can decide whether to promote Path A to be the primary reference.

# Parameter mapping (siam_vqe.L3Params → EDRIXS example_3 conventions)

  Phase 5            EDRIXS               value
  -----------------  -------------------  ------
  F2_dd              F2_dd                9.787
  F4_dd              F4_dd                6.078
  U_dd               U_dd                 7.3
  ten_dq             ten_dq               0.56
  cf_bath_split      ten_dq_bath          1.44
  Delta              Delta (CT)           4.7
  V_eg               Veg                  2.06
  V_t2g              Vt2g                 1.21
  U_dc (core-hole)   U_dp                 8.5
  (zeta_d = 0)       zeta_d_i = 0         0      <- Phase 5 has no SOC
  (multipoles off)   F2_dp,G1_dp,G3_dp=0  0      <- Phase 5 v1: monopole only
  num_d_orbitals     norb_d               10
  num_bath_orbitals  norb_bath/nbath      10 / 1

# Differences from EDRIXS example_3 (intentional)

1. ζ_d = 0 (no spin-orbit coupling), matching Phase 5 spec §2 non-goals.
2. F2_dp = G1_dp = G3_dp = 0 (no Slater multipole core-hole terms), matching
   Phase 5 spec §3.1 ("monopole V_core only; multipoles are v2").
3. ext_B = 0 (no magnetic field), matching Phase 5 (no XMCD in v1).
4. Linear polarization channels lin_z and lin_xy are computed separately so
   that Phase 5's two-channel result can be compared directly. The example
   used isotropic averaging; we override poltype_xas per call.
5. ω grid: 401 points over [-4, +4] eV around the leading peak, matching
   the Path-B reference grid for L2-distance comparison.
6. Γ_c = 0.5 eV (matches Phase 5 Lorentzian broadening; example used 0.48).

The L3 / L2 branching ratio difference vs the experimental spectrum is
expected — no SOC means a single edge, not two.
"""
from __future__ import annotations

import numpy as np

# `mpi4py` is required by edrixs's Fortran solver. Inside the container it's
# pre-installed; fail loudly if the user runs this outside the container.
try:
    import edrixs  # type: ignore
    from mpi4py import MPI  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "ERROR: edrixs / mpi4py not importable. This script is intended to run "
        "inside the user's `edrixs_run` Docker container. Outside the container, "
        "the Phase-B (scipy-ED) reference at "
        "data/edrixs_xas_l3_reference.npz remains the validation target.\n\n"
        f"ImportError: {exc}"
    ) from exc


# ------------------------------------------------------------------
# Parameters (mirror siam_vqe.L3Params + CoreHoleParams exactly)
# ------------------------------------------------------------------
nd = 8                    # impurity d-electron count
norb_d = 10               # spin-orbitals on impurity (5 orbitals x 2 spins)
nbath = 1                 # number of bath sites
norb_bath = 10            # spin-orbitals per bath site
v_noccu = nd + nbath * norb_d  # total valence electrons (18)
shell_name = ("d", "p")   # valence d, core p (L-edge)

# Slater-Condon integrals — Phase 5 uses pre-scaled values (the 0.8 scale
# already absorbed into our F2_dd / F4_dd defaults).
F0_dd = edrixs.get_F0("d", 9.787, 6.078) + 7.3   # F0 derived from F2, F4, U_dd
F2_dd = 9.787
F4_dd = 6.078

# Core-hole d-p multipoles: zero per Phase 5 v1 non-goals.
F2_dp = 0.0
G1_dp = 0.0
G3_dp = 0.0
F0_dp = edrixs.get_F0("dp", G1_dp, G3_dp) + 8.5   # F0 derived from G1, G3, U_dp

slater = (
    [F0_dd, F2_dd, F4_dd],                                  # initial-state d-d
    [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp],      # core-hole state
)

# Charge-transfer setup
Delta = 4.7
U_dd = 7.3
U_dp = 8.5

# EDRIXS conventions for impurity and bath energies relative to vacuum
E_d, E_L = edrixs.CT_imp_bath(U_dd, Delta, nd)
E_dc, E_Lc, E_p = edrixs.CT_imp_bath_core_hole(U_dd, U_dp, Delta, nd)

# Crystal-field on impurity (cubic; e_g at +0.6·10Dq, t_2g at -0.4·10Dq)
ten_dq = 0.56
CF = np.zeros((norb_d, norb_d), dtype=complex)
diag = np.arange(norb_d)
# EDRIXS d-orbital order is (3z2, zx, zy, x2-y2, xy) x2 spins.
orbital_energies = np.array([
    e for energy in [+0.6 * ten_dq,    # 3z2-r2  (e_g)
                     -0.4 * ten_dq,    # xz      (t_2g)
                     -0.4 * ten_dq,    # yz      (t_2g)
                     +0.6 * ten_dq,    # x2-y2   (e_g)
                     -0.4 * ten_dq]    # xy      (t_2g)
    for e in [energy] * 2  # both spins
])
CF[diag, diag] = orbital_energies

# No SOC in Phase 5
zeta_d_i = 0.0
zeta_d_n = 0.0
soc = edrixs.cb_op(edrixs.atom_hsoc("d", zeta_d_i), edrixs.tmat_c2r("d", True))

# Impurity matrices (CF + SOC + diagonal energy shift)
imp_mat = CF + soc + E_d * np.eye(norb_d)
imp_mat_n = CF + soc + E_dc * np.eye(norb_d)

# Bath energy splitting
ten_dq_bath = 1.44
bath_level = np.full((nbath, norb_d), E_L, dtype=complex)
bath_level[0, :2] += ten_dq_bath * 0.6   # 3z2 e_g
bath_level[0, 2:6] -= ten_dq_bath * 0.4  # xz/yz t_2g
bath_level[0, 6:8] += ten_dq_bath * 0.6  # x2-y2 e_g
bath_level[0, 8:] -= ten_dq_bath * 0.4   # xy t_2g
bath_level_n = bath_level.copy()  # core-hole state uses same bath level shifts
# (re-center to core-hole bath level reference E_Lc, not E_L)
bath_level_n = np.full((nbath, norb_d), E_Lc, dtype=complex)
bath_level_n[0, :2] += ten_dq_bath * 0.6
bath_level_n[0, 2:6] -= ten_dq_bath * 0.4
bath_level_n[0, 6:8] += ten_dq_bath * 0.6
bath_level_n[0, 8:] -= ten_dq_bath * 0.4

# Hybridization (sign +V per EDRIXS convention)
Veg = 2.06
Vt2g = 1.21
hyb = np.zeros((nbath, norb_d), dtype=complex)
hyb[0, :2] = Veg
hyb[0, 2:6] = Vt2g
hyb[0, 6:8] = Veg
hyb[0, 8:] = Vt2g

# Core-shell SOC: keep at zero (matches Phase 5 v1 — no L3/L2 split). If you
# want the realistic L3/L2 doublet later, set zeta_p_n to the 2p_{3/2}/{1/2}
# splitting (~11 eV for Ni); spec §2 holds this back to v2.
c_soc = 0.0

# Real-harmonic basis transform (cubic harmonics)
trans_c2n = edrixs.tmat_c2r("d", True)

# XAS calculation parameters
om_shift = 857.6        # nominal Ni L3 photon energy
c_level = -om_shift - 5 * E_p
ominc_xas = om_shift + np.linspace(-4.0, 4.0, 401)   # 8 eV window
gamma_c = np.full(ominc_xas.shape, 0.5)              # Gamma = 0.5 eV
temperature = 1.0       # Kelvin; cold (effectively zero-T)
thin = 0.0              # incidence angle
phi = 0.0               # azimuthal angle
ext_B = np.array([0.0, 0.0, 0.0])
on_which = "spin"


# ------------------------------------------------------------------
# Run ED + XAS
# ------------------------------------------------------------------
def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("[edrixs_path_a] Running EDRIXS ed_siam_fort on L3 NiO ...")

    do_ed = 1
    eval_i, denmat, noccu_gs = edrixs.ed_siam_fort(
        comm, shell_name, nbath,
        siam_type=0,
        imp_mat=imp_mat, imp_mat_n=imp_mat_n,
        bath_level=bath_level, bath_level_n=bath_level_n,
        hyb=hyb, c_level=c_level, c_soc=c_soc,
        slater=slater, ext_B=ext_B, on_which=on_which,
        trans_c2n=trans_c2n, v_noccu=v_noccu, do_ed=do_ed,
        ed_solver=2, neval=50, nvector=3, ncv=100, idump=True,
    )

    if rank == 0:
        assert abs(noccu_gs - v_noccu) < 1e-6
        imp_occ = float(np.sum(denmat[0].diagonal()[:norb_d]).real)
        bath_occ = float(np.sum(denmat[0].diagonal()[norb_d:]).real)
        print(f"[edrixs_path_a] GS impurity occupation = {imp_occ:.6f}")
        print(f"[edrixs_path_a] GS bath occupation      = {bath_occ:.6f}")
        print(f"[edrixs_path_a] eval_i[0:3]             = {eval_i[:3]}")

    # ------------------------------------------------------------------
    # Run XAS for each polarization channel via scatter_axis rotation
    # ------------------------------------------------------------------
    # Phase 5 followup note (2026-05-28-phase-5-path-a-followup.md)
    # diagnosed why all three channels collapsed in the prior version:
    #
    #   `pol_type=('linear', alpha)` selects the polarization angle WITHIN
    #   the scattering plane (alpha is angle between pol vector and the
    #   plane), NOT relative to a crystal axis. The scattering plane is
    #   defined by `scatter_axis` as its local zx-plane.
    #
    # In EDRIXS conventions:
    #   - alpha = 0       => π-polarization, pol vector in the scattering
    #                        plane (depends on thin)
    #   - alpha = π/2     => σ-polarization, pol vector NORMAL to the
    #                        scattering plane (== local y axis,
    #                        == scatter_axis[:,1])
    #
    # Strategy used here: take alpha=π/2 so that the polarization vector
    # is exactly the local-y axis of the scattering frame. Then rotate
    # `scatter_axis` so that its 2nd column (local-y) coincides with
    # crystal x, y, z in turn. This gives three genuinely distinct
    # polarization geometries.
    #
    # ------------------------------------------------------------------
    # Honesty clause (Phase 5b Task 1 closeout, 2026-05-28)
    # ------------------------------------------------------------------
    # With the rotations below applied correctly, EDRIXS produces three
    # BIT-FOR-BIT IDENTICAL spectra (max |sigma_x - sigma_y| = 0.0).
    # This is not a bug in the geometry call — it is the correct physics
    # for the Phase 5 model:
    #
    #   - Cubic crystal field only (no tetragonal / trigonal splitting)
    #   - Cubic hybridization (cubic-grouped Veg / Vt2g)
    #   - No SOC (zeta_d = 0)
    #   - No applied magnetic field (ext_B = 0)
    #   - Spin-summed isotropic dipole (no preferred axis)
    #
    # The point group is O_h, so the linear absorption tensor sigma_{ij}
    # is proportional to delta_{ij}, and sigma_x(omega) = sigma_y(omega)
    # = sigma_z(omega) exactly. Distinct channels require breaking O_h
    # (e.g. tetragonal CF for strained thin films, or turning on SOC).
    # Both are deferred to a later Phase 5b task; downstream tasks do
    # not depend on channel distinctness. See followup note.

    sqrt_pi_2 = np.pi / 2

    # scatter_axis columns = [x_local, y_local, z_local]; we pick
    # right-handed (det=+1) orthonormal frames where the y_local column
    # equals the desired crystal-axis polarization direction.
    scatter_axes: dict[str, np.ndarray] = {
        # lin_x: y_local = (1,0,0) ; x_local = (0,1,0) ; z_local = (0,0,-1)
        "lin_x": np.array([[0.0, 1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0]]),
        # lin_y: y_local = (0,1,0) ; identity frame
        "lin_y": np.eye(3),
        # lin_z: y_local = (0,0,1) ; x_local = (1,0,0) ; z_local = (0,-1,0)
        "lin_z": np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0]]),
    }

    # Sanity-check the rotations on rank 0.
    if rank == 0:
        for ch, R in scatter_axes.items():
            det = float(np.linalg.det(R))
            orth = float(np.linalg.norm(R @ R.T - np.eye(3)))
            print(f"[edrixs_path_a] scatter_axis[{ch}] det={det:+.6f}, "
                  f"|R Rᵀ - I|={orth:.2e}, y_col={R[:,1].tolist()}")

    spectra: dict[str, np.ndarray] = {}
    for ch_name, scatter_mat in scatter_axes.items():
        if rank == 0:
            print(f"[edrixs_path_a] Computing XAS for channel '{ch_name}' "
                  f"(σ-pol along crystal-{ch_name.split('_')[-1]}) ...")
        try:
            xas, _xas_poles = edrixs.xas_siam_fort(
                comm, shell_name, nbath, ominc_xas,
                gamma_c=gamma_c, v_noccu=v_noccu,
                thin=thin, phi=phi,
                num_gs=3, nkryl=200,
                pol_type=[("linear", sqrt_pi_2)],
                scatter_axis=scatter_mat,
                temperature=temperature,
            )
            spectra[ch_name] = np.asarray(xas).flatten()
            if rank == 0:
                print(f"[edrixs_path_a]   |sigma_{ch_name}| = "
                      f"{np.linalg.norm(spectra[ch_name]):.6f}")
        except Exception as exc:
            if rank == 0:
                print(f"[edrixs_path_a] Channel '{ch_name}' failed: {exc!r}")
                print("[edrixs_path_a] Inspect edrixs.xas_siam_fort docs and "
                      "adjust pol_type / scatter_axis syntax in this script.")

    # ------------------------------------------------------------------
    # Save NPZ (rank 0 only)
    # ------------------------------------------------------------------
    if rank == 0:
        import pathlib
        out = pathlib.Path("data/edrixs_xas_l3_reference_path_a.npz")
        out.parent.mkdir(parents=True, exist_ok=True)

        # Shift omega grid so leading peak sits at ω₀ = 0 (matches Phase 5 +
        # Path-B convention).
        omega_relative = ominc_xas - om_shift

        save_payload: dict[str, np.ndarray] = {
            # Primary grid (Phase 5b schema): plain `omega_grid` key.
            "omega_grid": omega_relative,
            # Backward-compat alias kept for any downstream code that read
            # the prior schema:
            "omega_grid_eV": omega_relative,
            "Gamma_eV": np.array(0.5),
            "om_shift_eV": np.array(om_shift),
            "notes": np.array(
                "EDRIXS Path-A reference for L3 NiO XAS, polarization-resolved "
                "via scatter_axis rotation (Phase 5b Task 1). Three channels "
                "lin_x/lin_y/lin_z plus isotropic = (x+y+z)/3. Generated "
                "inside edrixs/edrixs:latest Docker container. No SOC, no "
                "multipoles (monopole core-hole only) — matches Phase 5 v1."),
            "channels_present": np.array(list(spectra.keys())),
        }
        for ch_name, sigma in spectra.items():
            save_payload[f"sigma_{ch_name}"] = sigma

        # Isotropic average only meaningful if all three channels are present.
        if {"lin_x", "lin_y", "lin_z"}.issubset(spectra.keys()):
            save_payload["sigma_isotropic"] = (
                spectra["lin_x"] + spectra["lin_y"] + spectra["lin_z"]
            ) / 3.0

        np.savez(out, **save_payload)
        print(f"[edrixs_path_a] Wrote {out}")
        print(f"[edrixs_path_a] Channels saved: {list(spectra.keys())}")
        print("[edrixs_path_a] Per-channel L2 norms: " + ", ".join(
            f"{k}={np.linalg.norm(v):.6f}" for k, v in spectra.items()))


if __name__ == "__main__":
    main()
