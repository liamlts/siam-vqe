"""Path-A EDRIXS reference: assert polarization-resolved channels are distinct.

Phase 5b Task 1 split the prior single-channel EDRIXS call into three calls
with rotated ``scatter_axis`` matrices, one per crystal-axis polarization
direction, and writes ``sigma_lin_x / sigma_lin_y / sigma_lin_z`` plus
``sigma_isotropic`` into ``data/edrixs_xas_l3_reference_path_a.npz``.

Two checks:

1. Schema test (must pass): the NPZ has the new keys, channels have
   nonzero L2, and ``sigma_isotropic == (sx + sy + sz) / 3``.
2. Distinctness test (xfail expected on Phase 5b v1): the three channels
   should be pairwise distinct at ~5% L2 level. The Phase 5 L3 NiO SIAM
   has *exact* cubic O_h symmetry (cubic CF only, no SOC, no B-field,
   spin-summed isotropic dipole), so the linear XAS tensor is necessarily
   isotropic and the three channels collapse to bit-for-bit identical
   spectra. This is correct physics, not a geometry bug. Distinct
   channels require breaking the cubic symmetry (tetragonal CF or SOC),
   which is deferred to a later Phase 5b task per the followup note
   ``docs/superpowers/notes/2026-05-28-phase-5-path-a-followup.md``.
"""
from pathlib import Path
import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "data" / "edrixs_xas_l3_reference_path_a.npz"


@pytest.mark.skipif(not DATA.exists(), reason="Path-A NPZ not built yet")
def test_path_a_npz_schema():
    """NPZ has the polarization-resolved schema and isotropic average works."""
    d = np.load(DATA)
    for key in ("sigma_lin_x", "sigma_lin_y", "sigma_lin_z",
                "sigma_isotropic", "omega_grid"):
        assert key in d.files, f"missing NPZ key: {key}"
    sx, sy, sz = d["sigma_lin_x"], d["sigma_lin_y"], d["sigma_lin_z"]
    iso = d["sigma_isotropic"]
    assert np.linalg.norm(sx) > 0
    assert np.linalg.norm(sy) > 0
    assert np.linalg.norm(sz) > 0
    np.testing.assert_allclose(iso, (sx + sy + sz) / 3.0, rtol=1e-6)


@pytest.mark.skipif(not DATA.exists(), reason="Path-A NPZ not built yet")
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Cubic O_h symmetry of the Phase 5 NiO SIAM (no SOC, no symmetry-"
        "breaking CF) forces sigma_x = sigma_y = sigma_z. Channels are "
        "rotated correctly via scatter_axis; collapse is physical, not a "
        "script bug. See docs/superpowers/notes/"
        "2026-05-28-phase-5-path-a-followup.md."
    ),
)
def test_path_a_npz_has_three_distinct_polarization_channels():
    d = np.load(DATA)
    sx, sy, sz = d["sigma_lin_x"], d["sigma_lin_y"], d["sigma_lin_z"]
    assert np.linalg.norm(sx - sy) > 0.05 * np.linalg.norm(sx)
    assert np.linalg.norm(sy - sz) > 0.05 * np.linalg.norm(sy)
    assert np.linalg.norm(sx - sz) > 0.05 * np.linalg.norm(sx)
