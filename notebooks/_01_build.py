"""Build notebooks/01_L0_toy_groundstate.ipynb programmatically.

Goal: exercise the six validation layers from the spec §5 on the L0 Hubbard
dimer at U=4, t=1, eps=0.

Validation layers covered in Phase 1 (layer 6 = noise mitigation is Phase 3):
    1. check_energy_match
    2. check_state_overlap
    3. check_ansatz_expressivity
    4. check_multistart_spread
    5. check_observable_agreement (n_total, S^2, double_occ)
"""

from __future__ import annotations

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "source": [
            "# 01 — L0 toy ground state (Hubbard dimer)\n",
            "\n",
            "**Goal:** VQE on the 2-site Hubbard dimer at U=4, t=1, eps=0.\n",
            "\n",
            "**Validation layers** (from spec §5):\n",
            "1. Energy match against ED (|ΔE| < 1 mHa)\n",
            "2. State overlap ≥ 0.99\n",
            "3. Ansatz expressivity (max overlap when classically optimized) ≥ 0.99\n",
            "4. Multi-start spread across 8 seeded starts < 5 mHa\n",
            "5. Observable agreement (n_total, S², double occupancy) within 5%\n",
            "\n",
            "All five layers must pass. Each is reported with numerical margin.\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "import json\n",
            "import time\n",
            "from pathlib import Path\n",
            "\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            "import siam_vqe as svqe\n",
            "from siam_vqe import (\n",
            "    check_ansatz_expressivity, check_energy_match, check_multistart_spread,\n",
            "    check_observable_agreement, check_state_overlap,\n",
            "    efficient_su2_ansatz, exact_diag, hubbard_dimer, observables_dimer,\n",
            "    plot_convergence, run_vqe, run_vqe_multistart, to_qubit_op,\n",
            ")\n",
            "print(f'siam_vqe {svqe.__version__}')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "source": ["## 1. Set up the L0 Hubbard dimer\n"],
    },
    {
        "cell_type": "code",
        "source": [
            "U, t, eps = 4.0, 1.0, 0.0\n",
            "num_particles = (1, 1)  # half-filling, Sz=0\n",
            "\n",
            "fop = hubbard_dimer(U=U, t=t, eps=eps)\n",
            "pop = to_qubit_op(fop, scheme='parity_tapered', num_particles=num_particles)\n",
            "print(f'Hubbard dimer (U={U}, t={t}, eps={eps})')\n",
            "print(f'qubits (parity_tapered): {pop.num_qubits}')\n",
            "print(f'Pauli terms: {len(pop)}')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "source": ["## 2. ED reference\n"],
    },
    {
        "cell_type": "code",
        "source": [
            "ed = exact_diag(pop, k=4)\n",
            "print(f'ED ground state energy E0 = {ed.energies[0]:.8f}')\n",
            "print(f'lowest 4 levels: {np.round(ed.energies, 6)}')\n",
            "# Analytic check:\n",
            "import math\n",
            "e_analytic = 2 * eps + U / 2 - math.sqrt((U / 2) ** 2 + 4 * t ** 2)\n",
            "print(f'analytic GS: {e_analytic:.8f}')\n",
            "print(f'|ED - analytic| = {abs(ed.energies[0] - e_analytic):.2e}')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "source": ["## 3. VQE run (EfficientSU2 + COBYLA)\n"],
    },
    {
        "cell_type": "code",
        "source": [
            "circuit, x0 = efficient_su2_ansatz(num_qubits=pop.num_qubits, reps=3, seed=20260524)\n",
            "print(f'ansatz: EfficientSU2 reps=3, {circuit.num_parameters} parameters')\n",
            "result = run_vqe(pop, circuit, x0, optimizer='COBYLA', maxiter=600)\n",
            "print(f'VQE energy: {result.energy:.8f}')\n",
            "print(f'walltime: {result.walltime_s:.2f}s, evals: {len(result.history)}')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "source": ["### Convergence plot\n"],
    },
    {
        "cell_type": "code",
        "source": [
            "fig, ax = plt.subplots(figsize=(8, 4))\n",
            "plot_convergence(result, ed_energy=float(ed.energies[0]), ax=ax)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
        ],
    },
    {
        "cell_type": "markdown",
        "source": ["## 4. Validation layers\n"],
    },
    {
        "cell_type": "code",
        "source": [
            "# Layer 1: energy match\n",
            "r_energy = check_energy_match(result.energy, float(ed.energies[0]), tol_hartree=1e-3)\n",
            "print(f'Layer 1 (energy match): {\"PASS\" if r_energy.passed else \"FAIL\"} - dE = {r_energy.delta:.2e}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# Layer 2: state overlap\n",
            "r_overlap = check_state_overlap(result, ed, threshold=0.99)\n",
            "print(f'Layer 2 (state overlap): {\"PASS\" if r_overlap.passed else \"FAIL\"} - overlap = {r_overlap.overlap:.6f}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# Layer 3: ansatz expressivity (classical max-overlap probe)\n",
            "r_express = check_ansatz_expressivity(\n",
            "    circuit=result.circuit, target_state=ed.vectors[:, 0],\n",
            "    threshold=0.99, seed=20260524, maxiter=300,\n",
            ")\n",
            "print(f'Layer 3 (ansatz expressivity): {\"PASS\" if r_express.passed else \"FAIL\"} - max overlap = {r_express.max_overlap:.6f}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# Layer 4: multi-start spread\n",
            "multistart = run_vqe_multistart(\n",
            "    pop, circuit, n_starts=8, ansatz_factory_seed_base=1000,\n",
            "    ansatz_num_qubits=pop.num_qubits, ansatz_reps=3,\n",
            "    optimizer='COBYLA', maxiter=400,\n",
            ")\n",
            "r_spread = check_multistart_spread(multistart, max_spread_hartree=5e-3)\n",
            "print(f'Layer 4 (multi-start spread): {\"PASS\" if r_spread.passed else \"FAIL\"}')\n",
            "print(f'  best={r_spread.best:.6f}, median={r_spread.median:.6f}, spread={r_spread.spread:.2e}')\n",
            "energies = [r.energy for r in multistart]\n",
            "print('  per-start energies:', np.round(energies, 6))\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# Layer 5: observable agreement\n",
            "obs_fops = observables_dimer()\n",
            "obs_reports = {}\n",
            "for name in ['n_total', 'S2', 'double_occ']:\n",
            "    obs_pop = to_qubit_op(obs_fops[name], scheme='parity_tapered', num_particles=num_particles)\n",
            "    rep = check_observable_agreement(result, ed, observable=obs_pop, rel_tol=0.05, name=name)\n",
            "    obs_reports[name] = rep\n",
            "    print(f'Layer 5 ({name}): {\"PASS\" if rep.passed else \"FAIL\"} - VQE={rep.vqe_value:.4f}, ED={rep.ed_value:.4f}, rel_err={rep.rel_error:.2%}')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "source": ["## 5. Persist results\n"],
    },
    {
        "cell_type": "code",
        "source": [
            "out = {\n",
            "    'system': {'U': U, 't': t, 'eps': eps, 'num_particles': list(num_particles)},\n",
            "    'ansatz': {'name': 'EfficientSU2', 'reps': 3, 'num_parameters': int(circuit.num_parameters)},\n",
            "    'optimizer': result.optimizer_name,\n",
            "    'maxiter': 600,\n",
            "    'vqe_energy': float(result.energy),\n",
            "    'ed_energy': float(ed.energies[0]),\n",
            "    'validations': {\n",
            "        'layer_1_energy_match': {'passed': r_energy.passed, 'delta': float(r_energy.delta), 'tol': r_energy.tol_hartree},\n",
            "        'layer_2_state_overlap': {'passed': r_overlap.passed, 'overlap': float(r_overlap.overlap), 'threshold': r_overlap.threshold},\n",
            "        'layer_3_ansatz_expressivity': {'passed': r_express.passed, 'max_overlap': float(r_express.max_overlap), 'threshold': r_express.threshold},\n",
            "        'layer_4_multistart_spread': {'passed': r_spread.passed, 'best': float(r_spread.best), 'median': float(r_spread.median), 'spread': float(r_spread.spread)},\n",
            "        'layer_5_observables': {name: {'passed': rep.passed, 'vqe': float(rep.vqe_value), 'ed': float(rep.ed_value), 'rel_err': float(rep.rel_error)} for name, rep in obs_reports.items()},\n",
            "    },\n",
            "}\n",
            "out_path = Path('01_L0_vqe_result.json')\n",
            "out_path.write_text(json.dumps(out, indent=2))\n",
            "print(f'wrote {out_path}')\n",
            "all_pass = r_energy.passed and r_overlap.passed and r_express.passed and r_spread.passed and all(r.passed for r in obs_reports.values())\n",
            "print(f'\\nALL FIVE LAYERS PASS: {all_pass}')\n",
        ],
    },
]


def make_cell(cell: dict, idx: int) -> dict:
    out = {
        "id": f"cell-{idx:02d}",
        "cell_type": cell["cell_type"],
        "metadata": {},
        "source": cell["source"],
    }
    if cell["cell_type"] == "code":
        out["outputs"] = []
        out["execution_count"] = None
    return out


nb = {
    "cells": [make_cell(c, i) for i, c in enumerate(cells)],
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent / "01_L0_toy_groundstate.ipynb"
out_path.write_text(json.dumps(nb, indent=1))
print(f"wrote {out_path}")
