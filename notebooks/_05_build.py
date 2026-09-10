#!/usr/bin/env python
"""Build notebooks/05_L3_xas_simulator.ipynb programmatically.

Mirrors _04_build.py with the sys.path bootstrap fix for the editable-
install gotcha."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = []

    nb.cells.append(nbf.v4.new_markdown_cell(
        "# Notebook 05 - L3 NiO L-edge XAS (qEOM)\n\n"
        "Phase 5 deliverable. 18-qubit tapered SIAM, qEOM on H' = H + V_core, "
        "EDRIXS example_3 validation, two polarization channels.\n\n"
        "Reference spec: `docs/superpowers/specs/2026-05-27-siam-vqe-phase-5-qeom-xas-design.md`."
    ))

    nb.cells.append(nbf.v4.new_code_cell(
        "%matplotlib inline\n"
        "import sys\n"
        "from pathlib import Path\n"
        "# Editable-install bootstrap (mirrors _04_build.py).\n"
        "sys.path.insert(0, str(Path.cwd().parent.resolve()))\n"
        "import json\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from siam_vqe.xas import XASSpectrum\n"
        "from siam_vqe.analysis import plot_xas_spectrum\n"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load the L3 XAS production result\n\n"
        "Produced by `scripts/run_l3_xas_sim.py`. Re-running it here is "
        "expensive (~30 min); we just load the cached JSON."
    ))
    nb.cells.append(nbf.v4.new_code_cell(
        "with open('05_L3_xas_result.json') as fh:\n"
        "    result = json.load(fh)\n"
        "print(f\"Phase 5 pass: {result['phase5_pass']}\")\n"
        "for ch in ('lin_z', 'lin_xy'):\n"
        "    print(f\"  channel {ch}:\")\n"
        "    print(f\"    Layer 4 (spectral L2):  {result[ch]['layer4_spectral_weight']}\")\n"
        "    print(f\"    Layer 5 (sum-rule):    {result[ch]['layer5_sum_rule']}\")\n"
    ))

    for ch_name in ("lin_z", "lin_xy"):
        nb.cells.append(nbf.v4.new_markdown_cell(f"## 2. {ch_name} XAS spectrum"))
        nb.cells.append(nbf.v4.new_code_cell(
            f"ch = '{ch_name}'\n"
            "edrixs = np.load(Path('../data/edrixs_xas_l3_reference.npz'))\n"
            "p5_spec = XASSpectrum(\n"
            "    omega_eV=edrixs['omega_grid_eV'],\n"
            "    sigma=np.array(result[ch]['sigma']),\n"
            "    channel=ch,\n"
            "    Gamma_eV=float(edrixs['Gamma_eV']),\n"
            "    peak_energies=np.array(result[ch]['peak_energies']),\n"
            "    peak_weights=np.array(result[ch]['peak_weights']),\n"
            ")\n"
            "ed_spec = XASSpectrum(\n"
            "    omega_eV=edrixs['omega_grid_eV'],\n"
            "    sigma=np.array(result[ch]['edrixs_sigma']),\n"
            "    channel=ch,\n"
            "    Gamma_eV=float(edrixs['Gamma_eV']),\n"
            "    peak_energies=np.array([]),\n"
            "    peak_weights=np.array([]),\n"
            ")\n"
            "ax = plot_xas_spectrum(p5_spec, edrixs_spectrum=ed_spec)\n"
            "plt.show()\n"
        ))

    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 3. Closeout summary\n\n"
        "If `phase5_pass = True`, Phase 5 deliverable is complete. "
        "Otherwise, read each per-channel layer report for the failing gate."
    ))

    return nb


def main() -> None:
    here = Path(__file__).parent
    nb = build()
    out = here / "05_L3_xas_simulator.ipynb"
    with out.open("w") as fh:
        nbf.write(nb, fh)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
