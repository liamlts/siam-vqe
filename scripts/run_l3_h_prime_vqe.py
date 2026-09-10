#!/usr/bin/env python
"""ADAPT-VQE on H' in the (10, 9) sector -> psi'_GS for Phase 5 XAS.

Mirrors `run_l3_adapt.py` structure but on H' = H + V_core and starting
from the d^9 ^2E_g HF seed. Per Observation #19, derives the UCCSD pool's
(occupied, virtual) partition from seed 0 at runtime.

Example
-------
    python -u scripts/run_l3_h_prime_vqe.py --mode production \
        --seeds 4 --max-ops 30 --grad-tol 1e-4 \
        --output notebooks/05_L3_h_prime_vqe_result.json
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
    build_l3_xas_seeds,
    build_uccsd_pool,
    hf_state_to_tapered_statevector,
    pool_to_tapered_paulis,
)
from siam_vqe.core_hole import CoreHoleParams, tapered_l3_h_prime_pauli
from siam_vqe.hamiltonian_l3 import L3Params
from siam_vqe.reference_l3 import load_l3_reference
from siam_vqe.vqe_runner import run_adapt_multistart


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADAPT-VQE driver for H' (Phase 5 XAS final state)."
    )
    parser.add_argument("--mode", choices=["smoke", "production"], required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--max-ops", type=int, default=30)
    parser.add_argument("--grad-tol", type=float, default=1e-4)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("data/nio_l3_h_prime_reference.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/05_L3_h_prime_vqe_result.json"),
    )
    args = parser.parse_args()

    params = L3Params()
    ch = CoreHoleParams()
    print(f"[run_l3_h_prime_vqe] Mode: {args.mode}")
    print(f"[run_l3_h_prime_vqe] U_dc = {ch.U_dc}")

    # 1. Load H' ED reference.
    ref = load_l3_reference(args.reference_path)
    print(
        f"[run_l3_h_prime_vqe] H' ED ground = {ref.ground_energy:.6f} eV "
        f"in sector {ref.num_particles}"
    )

    # 2. Build tapered Pauli H' in the (10, 9) XAS final-state sector.
    t0 = time.time()
    H_prime_pauli = tapered_l3_h_prime_pauli(params, ch, num_particles=(10, 9))
    print(
        f"[run_l3_h_prime_vqe] H' tapered Pauli "
        f"({H_prime_pauli.num_qubits} qubits, {len(H_prime_pauli)} terms) "
        f"built in {time.time() - t0:.1f} s"
    )

    # 3. Build seeds first; the pool's occupied/virtual partition must match
    # the d^9 ^2E_g reference, not the lowest-by-index default (Obs #19).
    hf_seeds = build_l3_xas_seeds(num_seeds=args.seeds, num_particles=(10, 9))
    psi_seeds = [hf_state_to_tapered_statevector(hf) for hf in hf_seeds]

    # 4. Derive pool partition from seed 0.
    ref_hf = hf_seeds[0]
    occupied = list(ref_hf.occupied)
    virtual = sorted(set(range(ref_hf.num_spin_orbitals)) - set(occupied))
    fermi_pool = build_uccsd_pool(
        num_spin_orbitals=20, occupied=occupied, virtual=virtual,
    )
    pool = pool_to_tapered_paulis(fermi_pool, num_particles=(10, 9))
    print(
        f"[run_l3_h_prime_vqe] Pool size = {len(pool)} operators "
        f"(virtuals = {virtual})"
    )

    # 5. Multistart ADAPT.
    config = AdaptConfig(
        gradient_threshold=args.grad_tol,
        max_operators=args.max_ops,
        inner_optimizer="cobyla",
    )
    t1 = time.time()
    multi = run_adapt_multistart(H_prime_pauli, pool, psi_seeds, config)
    wall = time.time() - t1
    print(
        f"[run_l3_h_prime_vqe] Done in {wall:.1f}s -- "
        f"best={multi.best_energy:.6f}, median={multi.median_energy:.6f}, "
        f"worst={multi.worst_energy:.6f}, spread={multi.spread:.6f} eV"
    )

    # 6. Layer 1 (energy) validation against H' ED.
    err = abs(multi.best_energy - ref.ground_energy)
    layer1_pass = err < 0.05
    print(
        f"[run_l3_h_prime_vqe] |E_VQE - E_ED'| = {err * 1000:.2f} meV "
        f"(pass = {layer1_pass})"
    )

    # 7. Dump JSON result.
    best = multi.per_seed[multi.best_seed_index]
    out = {
        "mode": args.mode,
        "params": asdict(params),
        "core_hole": asdict(ch),
        "n_seeds": args.seeds,
        "wall_time_seconds": wall,
        "pool_size": len(pool),
        "best_seed_index": multi.best_seed_index,
        "best_energy": multi.best_energy,
        "median_energy": multi.median_energy,
        "worst_energy": multi.worst_energy,
        "spread": multi.spread,
        "ed_ground_energy": ref.ground_energy,
        "abs_err_eV": err,
        "layer1_pass": layer1_pass,
        "operators_picked": list(best.operators_picked),
        "theta": best.theta.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            out,
            f,
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o),
        )
    print(f"[run_l3_h_prime_vqe] Wrote {args.output}")


if __name__ == "__main__":
    main()
