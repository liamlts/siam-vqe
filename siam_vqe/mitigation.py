"""MitigationSpec dataclass + default grid for the Phase 3 noise study.

Pure data. No Qiskit imports — this module is consumed by both the CLI sweep
driver (which builds estimators) and the analysis notebook (which only reads
spec.name and spec.zne_* fields).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MitigationSpec:
    """One row of the Phase 3 mitigation grid.

    Attributes
    ----------
    name : short identifier used in JSON filenames and figure legends.
    m3 : enable M3 readout-error mitigation.
    zne : enable Zero-Noise Extrapolation.
    zne_extrapolator : one of "linear", "exponential", "polynomial_degree_3".
        Required when zne=True; None when zne=False.
    zne_noise_factors : tuple of noise scale factors to fold the circuit to.
        Required when zne=True; None when zne=False.
    shots : per-circuit shot count. Default 8192 matches Phase 2.
    """

    name: str
    m3: bool
    zne: bool
    zne_extrapolator: str | None
    zne_noise_factors: tuple[float, ...] | None
    shots: int = 8192


def make_default_grid() -> list[MitigationSpec]:
    """Return the 8 default Phase 3 mitigation configurations (spec §4.1)."""
    return [
        MitigationSpec(
            name="no_mit",
            m3=False, zne=False,
            zne_extrapolator=None, zne_noise_factors=None,
        ),
        MitigationSpec(
            name="m3_only",
            m3=True, zne=False,
            zne_extrapolator=None, zne_noise_factors=None,
        ),
        MitigationSpec(
            name="zne_lin_135",
            m3=False, zne=True,
            zne_extrapolator="linear",
            zne_noise_factors=(1.0, 3.0, 5.0),
        ),
        MitigationSpec(
            name="m3_zne_lin_135",
            m3=True, zne=True,
            zne_extrapolator="linear",
            zne_noise_factors=(1.0, 3.0, 5.0),
        ),
        MitigationSpec(
            name="m3_zne_exp_135",
            m3=True, zne=True,
            zne_extrapolator="exponential",
            zne_noise_factors=(1.0, 3.0, 5.0),
        ),
        MitigationSpec(
            name="m3_zne_poly3_135",
            m3=True, zne=True,
            zne_extrapolator="polynomial_degree_3",
            zne_noise_factors=(1.0, 3.0, 5.0),
        ),
        MitigationSpec(
            name="m3_zne_lin_123",
            m3=True, zne=True,
            zne_extrapolator="linear",
            zne_noise_factors=(1.0, 2.0, 3.0),
        ),
        MitigationSpec(
            name="m3_zne_lin_12345",
            m3=True, zne=True,
            zne_extrapolator="linear",
            zne_noise_factors=(1.0, 2.0, 3.0, 4.0, 5.0),
        ),
    ]
