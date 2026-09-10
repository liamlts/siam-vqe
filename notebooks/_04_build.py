#!/usr/bin/env python
"""Build notebooks/04_L3_nio_fulld_simulator.ipynb programmatically.

Mirrors the _03_build.py pattern: assembles a nbformat.NotebookNode and writes
it next to this script. Notebook is then executed via nbconvert with
--allow-errors so per-cell outputs survive any late failure.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = []

    nb.cells.append(nbf.v4.new_markdown_cell(
        "# Notebook 04 — L3 NiO full d⁸ + ligand bath ADAPT-VQE\n\n"
        "Phase 4 deliverable. 18-qubit tapered SIAM, operator-ADAPT-VQE, "
        "4-seed multistart, 6 validation layers, 2 figures.\n\n"
        "Reference spec: `docs/superpowers/specs/2026-05-26-siam-vqe-phase-4-l3-design.md`."
    ))

    nb.cells.append(nbf.v4.new_code_cell(
        "%matplotlib inline\n"
        "import sys\n"
        "from pathlib import Path\n"
        "# Bootstrap: the editable install can point at a different siam_vqe\n"
        "# checkout (e.g. the primary tree). Prepend this worktree's package root\n"
        "# so the notebook always imports the local L3 modules.\n"
        "sys.path.insert(0, str(Path.cwd().parent.resolve()))\n"
        "import json\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from siam_vqe.hamiltonian_l3 import L3Params\n"
        "from siam_vqe.reference_l3 import load_l3_reference, inspect_multiplets\n"
        "from siam_vqe.analysis import plot_adapt_convergence, plot_l3_observables\n"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 1. Reference spectrum\n\n"
        "Load the precomputed scipy-ED reference (committed at "
        "`data/nio_l3_reference.npz`) and inspect the low-lying spectrum to "
        "confirm a non-degenerate orbital singlet at the bottom of the (9,9) sector."
    ))
    nb.cells.append(nbf.v4.new_code_cell(
        "ref = load_l3_reference(Path('../data/nio_l3_reference.npz'))\n"
        "print(f'Ground energy: {ref.ground_energy:.6f} eV')\n"
        "for row in inspect_multiplets(ref):\n"
        "    print(f\"  state {row['index']}: E = {row['energy_eV']:.6f} eV, \"\n"
        "          f\"gap = {row['gap_eV']:.6f} eV\")\n"
        "print()\n"
        "print('ED reference observables:')\n"
        "for k, v in ref.observables.items():\n"
        "    print(f'  {k}: {v:.6f}')\n"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 2. Load production-run result\n\n"
        "The CLI driver `scripts/run_l3_adapt.py --mode production --seeds 4` "
        "writes `notebooks/04_L3_vqe_result.json`. Re-running it here is "
        "expensive (~30-60 min); we just load the cached JSON."
    ))
    nb.cells.append(nbf.v4.new_code_cell(
        "with open('04_L3_vqe_result.json') as fh:\n"
        "    result = json.load(fh)\n"
        "print(f\"Mode: {result['mode']}\")\n"
        "print(f\"Seeds: {result['n_seeds']}\")\n"
        "print(f\"Pool size: {result['pool_size']}\")\n"
        "print(f\"Best energy: {result['final_energy']:.6f} eV\")\n"
        "print(f\"ED reference: {ref.ground_energy:.6f} eV\")\n"
        "print(f\"Abs error: {abs(result['final_energy'] - ref.ground_energy)*1000:.2f} meV\")\n"
        "print(f\"Operators picked: {len(result['operators_picked'])} / max {result['config']['max_operators']}\")\n"
        "print(f\"Converged reason: {result['converged_reason']}\")\n"
        "print(f\"Wall time: {result['wall_time_seconds']:.0f} s\")\n"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Validation layers"))
    nb.cells.append(nbf.v4.new_code_cell(
        "for layer, payload in result['validation_layers'].items():\n"
        "    status = 'PASS' if payload['pass'] else 'FAIL'\n"
        "    print(f'  {layer}: {status}')\n"
        "    for k, v in payload.items():\n"
        "        if k == 'pass':\n"
        "            continue\n"
        "        if isinstance(v, (dict, list)):\n"
        "            print(f'    {k}: {type(v).__name__}({len(v)})')\n"
        "        else:\n"
        "            print(f'    {k}: {v}')\n"
        "    print()\n"
        "print(f\"Phase 4 pass = {result['phase4_pass']}\")\n"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Figure: ADAPT convergence trace"))
    nb.cells.append(nbf.v4.new_code_cell(
        "ax = plot_adapt_convergence(result['adapt_trace'],\n"
        "                             ed_reference=ref.ground_energy,\n"
        "                             output_path='../figures/04_adapt_convergence.pdf')\n"
        "plt.show()\n"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Figure: VQE vs ED observables"))
    nb.cells.append(nbf.v4.new_code_cell(
        "ax = plot_l3_observables(result['vqe_observables'],\n"
        "                          result['ed_observables'],\n"
        "                          output_path='../figures/04_l3_observables.pdf')\n"
        "plt.show()\n"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 6. Closeout summary\n\n"
        "If `phase4_pass = True`: Phase 4 complete.\n\n"
        "If `phase4_pass = False`: read the layer-by-layer payload. Energy gap > 50 meV "
        "with `max_operators` hit is a research finding (spec §6.2 honesty clause), "
        "NOT a bug — report final gap, dropped-gradient magnitudes, and what the pool "
        "didn't cover."
    ))

    return nb


def main() -> None:
    here = Path(__file__).parent
    nb = build()
    out = here / "04_L3_nio_fulld_simulator.ipynb"
    with out.open("w") as fh:
        nbf.write(nb, fh)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
