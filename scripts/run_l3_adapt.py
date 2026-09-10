#!/usr/bin/env python
"""CLI: run L3 ADAPT-VQE on the NiO SIAM (smoke or production mode).

Examples
--------
Smoke (1 seed, 8 operators max, < 5 min):
    python scripts/run_l3_adapt.py --mode smoke --seeds 1 --max-ops 8

Production (4 seeds, full pool, grad threshold 1e-4 eV):
    python scripts/run_l3_adapt.py --mode production --seeds 4 --max-ops 30 \
        --grad-tol 1e-4 --output notebooks/04_L3_vqe_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Ensure the package root is on sys.path when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from siam_vqe.adapt_vqe import (
    AdaptConfig,
    apply_exp_iT,
    build_l3_multistart_seeds,
    build_uccsd_pool,
    hf_state_to_tapered_statevector,
    pool_to_tapered_paulis,
)
from siam_vqe.analysis import (
    check_layer2_overlap_l3,
    check_layer4_multistart_l3,
    check_layer5_observables_l3,
    check_layer6_adapt_trace,
)
from siam_vqe.hamiltonian_l3 import L3Params, nio_l3_hamiltonian
from siam_vqe.observables_l3 import (
    evaluate_observable_on_sector,
    make_l3_observables,
)
from siam_vqe.reference_l3 import (
    compute_l3_reference,
    load_l3_reference,
    save_l3_reference,
)
from siam_vqe.tapering_l3 import lift_tapered_to_full_sector, tapered_l3_pauli
from siam_vqe.vqe_runner import run_adapt_multistart


def main() -> None:
    parser = argparse.ArgumentParser(description="L3 ADAPT-VQE driver.")
    parser.add_argument("--mode", choices=["smoke", "production"], required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--max-ops", type=int, default=30)
    parser.add_argument("--grad-tol", type=float, default=1e-4)
    parser.add_argument("--inner-opt", choices=["cobyla", "bfgs", "slsqp"],
                        default="cobyla")
    parser.add_argument("--reference-path", type=Path,
                        default=Path("data/nio_l3_reference.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("notebooks/04_L3_vqe_result.json"))
    args = parser.parse_args()

    params = L3Params()
    print(f"[run_l3_adapt] Mode: {args.mode}")
    print(f"[run_l3_adapt] Params: F2={params.F2_dd}, F4={params.F4_dd}, "
          f"V_eg={params.V_eg}, V_t2g={params.V_t2g}")

    # 1. Load or compute reference.
    if args.reference_path.exists():
        ref = load_l3_reference(args.reference_path)
        print(f"[run_l3_adapt] Loaded reference NPZ: E_0 = {ref.ground_energy:.6f} eV")
    else:
        print("[run_l3_adapt] Computing reference from scratch ...")
        ref = compute_l3_reference(params, k_states=10)
        args.reference_path.parent.mkdir(parents=True, exist_ok=True)
        save_l3_reference(ref, args.reference_path)
        print(f"[run_l3_adapt] Saved reference NPZ: E_0 = {ref.ground_energy:.6f} eV")

    # 2. Build Hamiltonian + tapered Pauli form.
    t0 = time.time()
    H = nio_l3_hamiltonian(params)
    H_pauli = tapered_l3_pauli(H, num_particles=(9, 9))
    print(f"[run_l3_adapt] Tapered H built ({H_pauli.num_qubits} qubits, "
          f"{len(H_pauli)} Pauli terms) in {time.time() - t0:.1f} s")

    # 3. Build seeds first — the pool's occupied/virtual partition must
    # match the actual reference HF (d⁸ ³A_2g with virtuals at modes 1, 10),
    # not the lowest-by-index default.
    hf_seeds = build_l3_multistart_seeds(num_seeds=args.seeds)
    psi_seeds = [hf_state_to_tapered_statevector(hf) for hf in hf_seeds]

    # 4. Build pool from seed 0's occupation/virtual partition.
    ref_hf = hf_seeds[0]
    occupied = list(ref_hf.occupied)
    virtual = sorted(set(range(ref_hf.num_spin_orbitals)) - set(occupied))
    fermionic_pool = build_uccsd_pool(num_spin_orbitals=20,
                                       occupied=occupied, virtual=virtual)
    pool = pool_to_tapered_paulis(fermionic_pool, num_particles=(9, 9))
    print(f"[run_l3_adapt] Pool size = {len(pool)} operators "
          f"(virtuals = {virtual})")

    # 5. Run multistart ADAPT.
    config = AdaptConfig(
        gradient_threshold=args.grad_tol,
        max_operators=args.max_ops,
        inner_optimizer=args.inner_opt,
    )
    t1 = time.time()
    multi = run_adapt_multistart(H_pauli, pool, psi_seeds, config)
    wall_time = time.time() - t1
    print(f"[run_l3_adapt] Multistart done in {wall_time:.1f} s")
    print(f"[run_l3_adapt] best={multi.best_energy:.6f}, "
          f"median={multi.median_energy:.6f}, worst={multi.worst_energy:.6f}, "
          f"spread={multi.spread:.6f} eV")

    # 6. Validation layers on best seed.
    best = multi.per_seed[multi.best_seed_index]
    psi_best = apply_exp_iT(psi_seeds[multi.best_seed_index],
                             [pool[k] for k in best.operators_picked],
                             list(best.theta))
    psi_sector = lift_tapered_to_full_sector(psi_best,
                                               num_particles=(9, 9),
                                               num_spin_orbitals=20)
    norm = np.linalg.norm(psi_sector)
    psi_sector = psi_sector / norm if norm > 0 else psi_sector
    vqe_obs = {}
    for name, op in make_l3_observables(num_spin_orbitals=20).items():
        vqe_obs[name] = evaluate_observable_on_sector(
            op, psi_sector, basis=ref.basis, num_spin_orbitals=20,
        )

    # Layer 1: energy.
    layer1 = {
        "pass": abs(best.final_energy - ref.ground_energy) < 0.05,
        "vqe_energy": best.final_energy,
        "ed_energy": ref.ground_energy,
        "abs_err_eV": abs(best.final_energy - ref.ground_energy),
        "tolerance_eV": 0.05,
    }
    layer2 = check_layer2_overlap_l3(psi_best, ref, threshold=0.85)
    layer4 = check_layer4_multistart_l3(multi, layer1_passed=layer1["pass"],
                                          spread_threshold=0.1)
    layer5 = check_layer5_observables_l3(vqe_obs, ref, rel_tol=0.05, abs_tol=0.05)
    layer6 = check_layer6_adapt_trace(best.trace, noise_floor_eV=0.001)

    phase4_pass = (
        layer1["pass"]
        and (layer4["pass"])
        and layer6["pass"]
    )
    print(f"[run_l3_adapt] Phase 4 pass = {phase4_pass}")

    # 7. Dump JSON.
    out = {
        "mode": args.mode,
        "params": asdict(params),
        "config": asdict(config),
        "n_seeds": args.seeds,
        "wall_time_seconds": wall_time,
        "pool_size": len(pool),
        "best_seed_index": multi.best_seed_index,
        "final_energy": multi.best_energy,
        "median_energy": multi.median_energy,
        "worst_energy": multi.worst_energy,
        "spread": multi.spread,
        "operators_picked": list(best.operators_picked),
        "n_operators": len(best.operators_picked),
        "theta": best.theta.tolist(),
        "adapt_trace": list(best.trace),
        "converged_reason": best.converged_reason,
        "vqe_observables": vqe_obs,
        "ed_observables": dict(ref.observables),
        "validation_layers": {
            "layer1_energy": layer1,
            "layer2_overlap": layer2,
            "layer4_multistart": layer4,
            "layer5_observables": layer5,
            "layer6_adapt_trace": layer6,
        },
        "phase4_pass": phase4_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"[run_l3_adapt] Wrote {args.output}")


if __name__ == "__main__":
    main()
