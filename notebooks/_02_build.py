"""Build notebooks/02_L1_nio_min_hardware.ipynb programmatically.

Phase 2 deliverable: L1 NiO SIAM (4 spin-orbitals → 2 qubits) with noiseless
VQE, FakeMarrakesh noisy sim, and one open-plan IBM Quantum hardware run
across 3 EstimatorV2 resilience levels.

All six validation layers from spec §6:
    1. Energy match against ED (|ΔE| < 1 mHa on noiseless)
    2. State overlap ≥ 0.99 on noiseless
    3. Ansatz expressivity ≥ 0.99
    4. Multi-start spread (N=8) best-to-median < 1 mHa
    5. Observable agreement (n_d, n_p, S²_d, double_occ_d) within 1% on noiseless
    6. Resilience-tier guardrail on hardware (informational)

The notebook is split into LOCAL cells (noiseless + FakeMarrakesh) and
HARDWARE cells (submit, retrieve, post-process). The local half must
pass layers 1-5 before the hardware half is executed.
"""

from __future__ import annotations

import json
from pathlib import Path

cells: list[dict] = []

# ---- Cell 1: Title + overview (markdown)
cells.append(
    {
        "cell_type": "markdown",
        "source": [
            "# 02 — L1 NiO single-orbital SIAM + IBM hardware run\n",
            "\n",
            "**Goal:** VQE on a 4-spin-orbital Anderson impurity reduction of NiO,\n",
            "with parameters from EDRIXS example_3 (Haverkort PRB 85, 165113).\n",
            "Execute on (a) noiseless simulator, (b) FakeMarrakesh noisy sim,\n",
            "(c) real IBM Quantum hardware at 3 resilience levels.\n",
            "\n",
            "**Honesty note:** this is *not* a literal d⁸ NiO calculation. It's a\n",
            "half-filled 2-mode toy reduction that inherits its parameters from\n",
            "the EDRIXS example_3 NiO Anderson impurity model.\n",
            "\n",
            "**Validation layers** (spec §6):\n",
            "1. Energy match against ED (|ΔE| < 1 mHa noiseless)\n",
            "2. State overlap ≥ 0.99 noiseless\n",
            "3. Ansatz expressivity ≥ 0.99\n",
            "4. Multi-start spread (N=8) best-to-median < 1 mHa\n",
            "5. Observable agreement (n_d, n_p, S²_d, double_occ_d) within 1%\n",
            "6. Hardware resilience-tier monotonicity (informational)\n",
            "\n",
            "Layers 1-5 are HARD GATES on noiseless. Hardware submission is\n",
            "skipped if any of those fail.\n",
        ],
    }
)

# ---- Cell 2: Imports + config (code)
cells.append(
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
            "    check_observable_agreement_multi, check_resilience_guardrail,\n",
            "    check_state_overlap, compare_energies, compute_l1_levels, exact_diag,\n",
            "    make_noisy_estimator, nio_l1_anderson, observables_l1, run_vqe,\n",
            "    run_vqe_multistart, to_qubit_op, uccsd_ansatz,\n",
            ")\n",
            "from siam_vqe.hardware import (\n",
            "    make_runtime_estimator, pick_backend, transpile_for_backend,\n",
            ")\n",
            "print(f'siam_vqe {svqe.__version__}')\n",
        ],
    }
)

# ---- Cell 3: L1 parameters (markdown)
cells.append(
    {
        "cell_type": "markdown",
        "source": [
            "## 1. L1 parameters (from EDRIXS example_3 + CT_imp_bath)\n",
            "\n",
            "U_dd, V_eg, Δ, 10Dq, ten_dq_bath: verbatim from example_3.\n",
            "(ε_d, ε_p) computed via closed-form CT_imp_bath port + eg crystal-field shift.\n",
        ],
    }
)

# ---- Cell 4: Load parameters
cells.append(
    {
        "cell_type": "code",
        "source": [
            "U = 7.3       # U_dd\n",
            "V = 2.06      # V_eg\n",
            "eps_d, eps_p = compute_l1_levels()\n",
            "print(f'U = {U} eV')\n",
            "print(f'V = {V} eV')\n",
            "print(f'eps_d = {eps_d:.6f} eV')\n",
            "print(f'eps_p = {eps_p:.6f} eV')\n",
            "print(f'eps_d - eps_p = {eps_d - eps_p:.6f} eV')\n",
        ],
    }
)

# ---- Cell 5: Build H and observables (markdown + code)
cells.append(
    {"cell_type": "markdown", "source": ["## 2. Build H + observables + ED reference\n"]}
)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "fop = nio_l1_anderson(U=U, V=V, eps_d=eps_d, eps_p=eps_p)\n",
            "num_particles = (1, 1)\n",
            "pop = to_qubit_op(fop, scheme='parity_tapered', num_particles=num_particles)\n",
            "obs = observables_l1()\n",
            "print(f'qubits (parity_tapered): {pop.num_qubits}')\n",
            "print(f'Pauli terms: {len(pop)}')\n",
            "\n",
            "ed = exact_diag(pop, k=4)\n",
            "print(f'ED ground state E0 = {ed.energies[0]:.8f} eV')\n",
            "print(f'lowest 4: {np.round(ed.energies, 6)}')\n",
        ],
    }
)

# ---- Cell 6: Noiseless VQE (markdown + multi-start)
cells.append(
    {
        "cell_type": "markdown",
        "source": ["## 3. Noiseless VQE (UCCSD + SLSQP, N=8 multi-start)\n"],
    }
)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "circuit, x0 = uccsd_ansatz(\n",
            "    num_spatial_orbitals=2,\n",
            "    num_particles=num_particles,\n",
            "    mapper_scheme='parity_tapered',\n",
            ")\n",
            "print(f'ansatz: UCCSD, {circuit.num_parameters} parameters')\n",
            "\n",
            "multistart = run_vqe_multistart(\n",
            "    pop, circuit, n_starts=8, optimizer='SLSQP', maxiter=200, seed=20260524,\n",
            ")\n",
            "best = multistart.best\n",
            "print(f'best E = {best.energy:.8f} eV  ({best.optimizer_name})')\n",
            "print(f'spread (best-to-median) = {multistart.spread_best_to_median:.2e}')\n",
        ],
    }
)

# ---- Cell 7: Validation layers 1-5 (markdown + code with assertion gate)
cells.append(
    {
        "cell_type": "markdown",
        "source": [
            "## 4. Validation layers 1-5 (hard gate)\n",
            "\n",
            "All five must pass before submitting any hardware job.\n",
        ],
    }
)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "# Layer 1: energy match\n",
            "em = check_energy_match(best.energy, ed.energies[0], abs_tol=1e-3)\n",
            "print(f'Layer 1 (energy match): passed={em.passed} |ΔE|={em.delta_e:.3e}')\n",
            "\n",
            "# Layer 2: state overlap\n",
            "ov = check_state_overlap(best, pop, ed.vectors[:, 0])\n",
            "print(f'Layer 2 (state overlap): passed={ov.passed} overlap²={ov.overlap_sq:.6f}')\n",
            "\n",
            "# Layer 3: ansatz expressivity\n",
            "exp = check_ansatz_expressivity(circuit, ed.vectors[:, 0], n_starts=8, seed=20260524)\n",
            "print(f'Layer 3 (expressivity): passed={exp.passed} max-overlap={exp.max_overlap:.6f}')\n",
            "\n",
            "# Layer 4: multi-start spread\n",
            "ms = check_multistart_spread(multistart, tol_mha=1.0)\n",
            "print(\n",
            "    f'Layer 4 (multi-start spread): passed={ms.passed} '\n",
            "    f'best-to-median={ms.spread_best_to_median:.3e}'\n",
            ")\n",
            "\n",
            "# Layer 5: observable agreement\n",
            "ob = check_observable_agreement_multi(best, pop, ed.vectors[:, 0], obs, num_particles=(1, 1), rel_tol=0.01)\n",
            "print(f'Layer 5 (observables): passed={ob.passed} max_rel_err={ob.max_rel_error:.3e}')\n",
            "for k, v in ob.values.items():\n",
            "    print(f'  {k}: VQE={v.vqe:.6f}  ED={v.ed:.6f}  rel_err={v.rel_error:.3e}')\n",
            "\n",
            "all_passed = em.passed and ov.passed and exp.passed and ms.passed and ob.passed\n",
            "assert all_passed, 'Noiseless validation failed; aborting before hardware submit.'\n",
            "print('\\nAll five noiseless layers PASSED — clear to proceed to FakeMarrakesh + hardware.')\n",
        ],
    }
)

# ---- Cell 8: FakeMarrakesh noisy sim (markdown + code)
cells.append(
    {"cell_type": "markdown", "source": ["## 5. FakeMarrakesh noisy simulator at x*\n"]}
)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "from qiskit_ibm_runtime.fake_provider import FakeMarrakesh\n",
            "from qiskit import transpile\n",
            "\n",
            "bound_circuit = circuit.assign_parameters(best.params)\n",
            "fake = FakeMarrakesh()\n",
            "isa_circ_fake = transpile(bound_circuit, backend=fake, optimization_level=3)\n",
            "isa_pop_fake = pop.apply_layout(isa_circ_fake.layout)\n",
            "noisy_est = make_noisy_estimator(fake, shots=8192, seed=20260524)\n",
            "\n",
            "job_fake = noisy_est.run([(isa_circ_fake, isa_pop_fake)])\n",
            "res_fake = job_fake.result()\n",
            "E_fake = float(res_fake[0].data.evs)\n",
            "E_fake_std = float(res_fake[0].data.stds) if hasattr(res_fake[0].data, 'stds') else 0.0\n",
            "print(f'FakeMarrakesh: E = {E_fake:.6f} ± {E_fake_std:.6f} eV')\n",
            "print(f'gap to ED:    |ΔE| = {abs(E_fake - ed.energies[0]):.6f} eV')\n",
        ],
    }
)

# ---- Cell 9: Partial JSON dump (after local stages)
cells.append(
    {"cell_type": "markdown", "source": ["## 6. Partial result snapshot (pre-hardware)\n"]}
)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "partial = {\n",
            "    'phase': 2,\n",
            "    'level': 'L1',\n",
            "    'parameters': {'U': U, 'V': V, 'eps_d': eps_d, 'eps_p': eps_p},\n",
            "    'ed_energy': float(ed.energies[0]),\n",
            "    'noiseless_energy': float(best.energy),\n",
            "    'noiseless_params': best.params.tolist(),\n",
            "    'multistart_spread': float(multistart.spread_best_to_median),\n",
            "    'fake_marrakesh_energy': E_fake,\n",
            "    'fake_marrakesh_std': E_fake_std,\n",
            "    'validation_layers': {\n",
            "        '1_energy_match': bool(em.passed),\n",
            "        '2_state_overlap': bool(ov.passed),\n",
            "        '3_expressivity': bool(exp.passed),\n",
            "        '4_multistart_spread': bool(ms.passed),\n",
            "        '5_observable_agreement': bool(ob.passed),\n",
            "    },\n",
            "}\n",
            "partial_path = Path('02_L1_partial_result.json')\n",
            "partial_path.write_text(json.dumps(partial, indent=2))\n",
            "print(f'wrote {partial_path}')\n",
        ],
    }
)

# ---- Cell 10: Hardware section header (markdown)
cells.append(
    {
        "cell_type": "markdown",
        "source": [
            "---\n",
            "\n",
            "## 7. Hardware execution\n",
            "\n",
            "**Queue expectation:** open-plan jobs can wait 12-48 h end-to-end.\n",
            "Job IDs are captured to `02_L1_job_ids.json` immediately after\n",
            "submission so this notebook can be killed and resumed.\n",
            "\n",
            "Three EstimatorV2 jobs at resilience_level = {0, 1, 2}.\n",
            "Each evaluates ⟨H⟩ at x* with 8192 shots.\n",
        ],
    }
)

# ---- Cell 11: Submit hardware jobs (code, comment-on-by-default for safety)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "# Set SUBMIT_HARDWARE = True to actually submit. Default False to prevent\n",
            "# accidental open-plan-budget burn when re-executing the notebook.\n",
            "SUBMIT_HARDWARE = False\n",
            "\n",
            "if SUBMIT_HARDWARE:\n",
            "    from qiskit_ibm_runtime import QiskitRuntimeService\n",
            "    service = QiskitRuntimeService()\n",
            "    backend = pick_backend(service, min_qubits=2)\n",
            "    print(f'backend: {backend.name} (pending_jobs={backend.status().pending_jobs})')\n",
            "\n",
            "    isa_circ_hw, isa_pop_hw = transpile_for_backend(bound_circuit, pop, backend)\n",
            "    print(f'ISA circuit: {isa_circ_hw.num_qubits} qubits, depth {isa_circ_hw.depth()}')\n",
            "\n",
            "    job_ids = {}\n",
            "    for rl in [0, 1, 2]:\n",
            "        est = make_runtime_estimator(backend, resilience_level=rl, default_shots=8192)\n",
            "        job = est.run([(isa_circ_hw, isa_pop_hw)])\n",
            "        job_ids[f'hw_L{rl}'] = job.job_id()\n",
            "        print(f'  resilience_level={rl}: job_id={job.job_id()}')\n",
            "\n",
            "    submission = {\n",
            "        'backend': backend.name,\n",
            "        'submitted_at': time.time(),\n",
            "        'job_ids': job_ids,\n",
            "        'shots': 8192,\n",
            "    }\n",
            "    Path('02_L1_job_ids.json').write_text(json.dumps(submission, indent=2))\n",
            "    print('wrote 02_L1_job_ids.json')\n",
            "else:\n",
            "    print('SUBMIT_HARDWARE = False — set to True and re-run this cell to submit.')\n",
        ],
    }
)

# ---- Cell 12: Retrieve results (code, looks up from job_ids.json)
cells.append(
    {"cell_type": "markdown", "source": ["## 8. Retrieve hardware results (blocking)\n"]}
)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "from qiskit_ibm_runtime import QiskitRuntimeService\n",
            "\n",
            "job_ids_path = Path('02_L1_job_ids.json')\n",
            "if not job_ids_path.exists():\n",
            "    raise FileNotFoundError(\n",
            "        '02_L1_job_ids.json not found — set SUBMIT_HARDWARE=True in the '\n",
            "        'previous cell and re-execute it first.'\n",
            "    )\n",
            "submission = json.loads(job_ids_path.read_text())\n",
            "service = QiskitRuntimeService()\n",
            "\n",
            "hw_results: dict[str, dict[str, float]] = {}\n",
            "for label, jid in submission['job_ids'].items():\n",
            "    job = service.job(jid)\n",
            "    print(f'  {label} ({jid}): status = {job.status()}')\n",
            "    res = job.result()\n",
            "    pub = res[0]\n",
            "    ev = float(pub.data.evs)\n",
            "    std = float(pub.data.stds) if hasattr(pub.data, 'stds') else 0.0\n",
            "    hw_results[label] = {'energy': ev, 'std': std}\n",
            "    print(f'    E = {ev:.6f} ± {std:.6f} eV')\n",
        ],
    }
)

# ---- Cell 13: Layer 6 + final comparison + JSON
cells.append(
    {
        "cell_type": "markdown",
        "source": ["## 9. Layer 6 guardrail + comparison plot + final JSON\n"],
    }
)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "# Layer 6 guardrail (informational; pass-or-flag, never aborts)\n",
            "guardrail = check_resilience_guardrail(\n",
            "    e_ed=float(ed.energies[0]),\n",
            "    e_hw_l0=hw_results['hw_L0']['energy'],\n",
            "    e_hw_l1=hw_results['hw_L1']['energy'],\n",
            "    e_hw_l2=hw_results['hw_L2']['energy'],\n",
            ")\n",
            "print(f'Layer 6 (resilience guardrail): passed={guardrail.passed}')\n",
            "print(f'  gap_L0 = {guardrail.gap_l0:.6f} eV')\n",
            "print(f'  gap_L1 = {guardrail.gap_l1:.6f} eV')\n",
            "print(f'  gap_L2 = {guardrail.gap_l2:.6f} eV')\n",
            "print(f'  notes: {guardrail.notes}')\n",
            "\n",
            "# Comparison figure\n",
            "all_energies = {\n",
            "    'ED': float(ed.energies[0]),\n",
            "    'noiseless': float(best.energy),\n",
            "    'FakeMarrakesh': E_fake,\n",
            "    'hw_L0': hw_results['hw_L0']['energy'],\n",
            "    'hw_L1': hw_results['hw_L1']['energy'],\n",
            "    'hw_L2': hw_results['hw_L2']['energy'],\n",
            "}\n",
            "uncertainties = {\n",
            "    'FakeMarrakesh': E_fake_std,\n",
            "    'hw_L0': hw_results['hw_L0']['std'],\n",
            "    'hw_L1': hw_results['hw_L1']['std'],\n",
            "    'hw_L2': hw_results['hw_L2']['std'],\n",
            "}\n",
            "fig = compare_energies(\n",
            "    all_energies, ed_energy=float(ed.energies[0]), uncertainties=uncertainties,\n",
            "    title=(\n",
            "        'L1 NiO SIAM — noiseless / FakeMarrakesh / hardware '\n",
            "        '(3 mitigation tiers) vs. ED'\n",
            "    ),\n",
            ")\n",
            "fig_dir = Path('../figures'); fig_dir.mkdir(exist_ok=True)\n",
            "fig.savefig(fig_dir / '02_L1_energy_comparison.png', dpi=150)\n",
            "fig.savefig(fig_dir / '02_L1_energy_comparison.pdf')\n",
            "plt.show()\n",
            "\n",
            "# Full result JSON\n",
            "final = dict(partial)\n",
            "final['backend'] = submission['backend']\n",
            "final['hw_results'] = hw_results\n",
            "final['validation_layers']['6_resilience_guardrail'] = bool(guardrail.passed)\n",
            "final['guardrail'] = {\n",
            "    'passed': bool(guardrail.passed),\n",
            "    'gap_l0': guardrail.gap_l0,\n",
            "    'gap_l1': guardrail.gap_l1,\n",
            "    'gap_l2': guardrail.gap_l2,\n",
            "    'notes': guardrail.notes,\n",
            "}\n",
            "Path('02_L1_vqe_result.json').write_text(json.dumps(final, indent=2))\n",
            "print('wrote 02_L1_vqe_result.json')\n",
        ],
    }
)

# ---- Build notebook JSON
notebook = {
    "cells": [
        {
            "cell_type": c["cell_type"],
            "metadata": {},
            "source": c["source"],
            **({"execution_count": None, "outputs": []} if c["cell_type"] == "code" else {}),
        }
        for c in cells
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "02_L1_nio_min_hardware.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {out}")
