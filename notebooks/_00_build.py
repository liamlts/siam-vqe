"""Build notebooks/00_setup_smoke_test.ipynb programmatically.

Running this script: python notebooks/_00_build.py
Output: notebooks/00_setup_smoke_test.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "source": [
            "# 00 — Setup smoke test\n",
            "\n",
            "Verifies all dependencies import, the simulator instantiates, ",
            "and a trivial VQE call completes. If this notebook runs top-to-bottom, ",
            "the project is correctly installed.\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "import sys\n",
            "print('Python', sys.version)\n",
            "import qiskit, qiskit_nature, qiskit_algorithms, qiskit_aer\n",
            "print('qiskit', qiskit.__version__)\n",
            "print('qiskit_nature', qiskit_nature.__version__)\n",
            "print('qiskit_algorithms', qiskit_algorithms.__version__)\n",
            "print('qiskit_aer', qiskit_aer.__version__)\n",
            "import siam_vqe\n",
            "print('siam_vqe', siam_vqe.__version__)\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "from siam_vqe import hubbard_dimer, to_qubit_op, exact_diag\n",
            "fop = hubbard_dimer(U=4.0, t=1.0)\n",
            "pop = to_qubit_op(fop, scheme='parity_tapered', num_particles=(1, 1))\n",
            "print(f'Hamiltonian: {pop.num_qubits} qubits, {len(pop)} Pauli terms')\n",
            "ed = exact_diag(pop, k=1)\n",
            "print(f'ED ground state energy: {ed.energies[0]:.6f}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "from siam_vqe import efficient_su2_ansatz, run_vqe\n",
            "circuit, x0 = efficient_su2_ansatz(num_qubits=pop.num_qubits, reps=2, seed=0)\n",
            "result = run_vqe(pop, circuit, x0, optimizer='COBYLA', maxiter=200)\n",
            "print(f'VQE energy: {result.energy:.6f}  ({len(result.history)} evals, {result.walltime_s:.2f}s)')\n",
            "print(f'Gap to ED: {result.energy - ed.energies[0]:.2e}')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "If everything above printed without errors and the VQE gap to ED ",
            "is small (typically <1e-3 for this trivial problem), the project is installed correctly.\n",
        ],
    },
]


def make_cell(cell: dict, idx: int) -> dict:
    out = {
        "cell_type": cell["cell_type"],
        "id": f"cell-{idx:02d}",
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

out_path = Path(__file__).parent / "00_setup_smoke_test.ipynb"
out_path.write_text(json.dumps(nb, indent=1))
print(f"wrote {out_path}")
