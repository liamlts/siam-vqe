"""Phase 3 CLI: L2 NiO e_g² SIAM noisy-sim mitigation sweep.

Builds the L2 Kanamori Hamiltonian, runs noiseless VQE to a Phase-1-grade
result, then loops over the Phase 3 mitigation grid running each config on
AerSimulator.from_backend(FakeMarrakesh()).

Per-config results are written to <output-dir>/<spec.name>.json immediately
so a kill mid-sweep keeps progress. Re-running skips configs whose JSON
already exists.

Usage:
    python scripts/run_l2_noise_sweep.py \
        --output-dir notebooks/03_L2_sweep_results \
        --grid default \
        --backend fake-marrakesh \
        --seed 20260525

Flags:
    --shots <N>         Per-config shot count (default 8192).
    --quick             Set shots=2048 (fast iteration mode).
    --backend <name>    "fake-marrakesh" (default) or "ibm_marrakesh" (stretch).
    --grid <name>       "default" only for now; future-extensible.
    --seed <int>        Multistart seed (default 20260525).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Bootstrap sys.path so the script picks up *this* worktree's siam_vqe package
# rather than whatever the editable install points at (pytest handles this via
# rootdir; running `python scripts/...` directly does not).
_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

# Imports deferred to main() to keep --help fast and to allow alternative
# backend modules to be loaded lazily.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for per-config JSON outputs.",
    )
    p.add_argument(
        "--grid", default="default", choices=["default"],
        help="Mitigation grid factory name.",
    )
    p.add_argument(
        "--backend", default="fake-marrakesh",
        choices=["fake-marrakesh", "ibm_marrakesh"],
        help="Substrate for the sweep. ibm_marrakesh is the stretch hardware path.",
    )
    p.add_argument("--shots", type=int, default=8192)
    p.add_argument("--quick", action="store_true",
                   help="Set shots=2048 for fast iteration (overrides --shots).")
    p.add_argument("--seed", type=int, default=20260525)
    return p.parse_args()


def _scrub_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Drop non-JSON-serializable fields from estimator metadata."""
    safe: dict[str, Any] = {}
    for k, v in meta.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = repr(v)[:200]
    return safe


def main() -> int:
    args = parse_args()

    if args.quick:
        args.shots = 2048

    print(f"Phase 3 sweep — output_dir={args.output_dir}, backend={args.backend}, shots={args.shots}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Imports (deferred from module top) ---
    import numpy as np

    from siam_vqe.ansatz import uccsd_ansatz
    from siam_vqe.hamiltonian import nio_l2_kanamori, observables_l2
    from siam_vqe.mappings import to_qubit_op
    from siam_vqe.reference_ed import exact_diag
    from siam_vqe.reference_edrixs import compute_l2_levels
    from siam_vqe.vqe_runner import run_vqe_multistart

    # --- Step 1: Compute L2 levels ---
    print("Step 1: compute_l2_levels()")
    levels = compute_l2_levels()
    print(f"  U={levels.U:.4f}, U'={levels.U_prime:.4f}, J_H={levels.J_H:.4f}")
    print(f"  V={levels.V:.4f}, ε_d={levels.eps_d:.4f}, ε_p={levels.eps_p:.4f}")

    # --- Step 2: Build H ---
    print("Step 2: build L2 Hamiltonian")
    fop = nio_l2_kanamori(
        U=levels.U, U_prime=levels.U_prime, J_H=levels.J_H,
        V=levels.V, eps_d=levels.eps_d, eps_p=levels.eps_p,
    )
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(3, 3))
    print(f"  qubit count: {pop.num_qubits} (expect 6)")

    # --- Step 3: ED ---
    print("Step 3: scipy.sparse ED")
    ed = exact_diag(pop, k=5)
    print(f"  GS energy: {ed.energies[0]:.6f} eV")
    print(f"  Lowest 5 energies: {ed.energies}")

    # --- Step 4: Pre-VQE polarization check (spec risk #5) ---
    psi_ed = ed.vectors[:, 0]
    max_weight = float(np.max(np.abs(psi_ed) ** 2))
    if max_weight > 0.95:
        print(f"  WARNING: GS is trivially polarized (max basis weight = {max_weight:.3f}). "
              "Phase 3 risk #5 — consider gauge adjustment.")
    else:
        print(f"  GS multi-configuration weight OK (max basis weight = {max_weight:.3f}).")

    # --- Step 5: Noiseless VQE multistart (with reps=2 retry if reps=1 fails Layer 1 or 4) ---
    from qiskit import QuantumCircuit

    from siam_vqe.vqe_runner import MultistartResult

    def _run_multistart_at_reps(reps: int) -> tuple[QuantumCircuit, MultistartResult]:
        ansatz_circuit, _ = uccsd_ansatz(
            num_spatial_orbitals=4, num_particles=(3, 3),
            mapper_scheme="parity_tapered", reps=reps,
        )
        result = run_vqe_multistart(
            pop, ansatz_circuit, n_starts=8,
            optimizer="SLSQP", maxiter=300, seed=args.seed,
        )
        return ansatz_circuit, result

    multistart_seed = args.seed
    print("Step 5: noiseless VQE (8 starts, UCCSD reps=1)")
    ansatz_circuit, multistart = _run_multistart_at_reps(reps=1)
    print(f"  reps=1: best={multistart.best.energy:.6f}, "
          f"spread={multistart.spread_best_to_median:.2e}")

    # Quick layer-1 + layer-4 pre-check; retry with reps=2 if either fails
    from siam_vqe.analysis import check_energy_match, check_multistart_spread
    pre_layer1 = check_energy_match(multistart.best.energy, float(ed.energies[0]), tol_hartree=1e-3)
    pre_layer4 = check_multistart_spread(multistart, tol_mha=1.0)
    if not (pre_layer1.passed and pre_layer4.passed):
        print(
            f"  reps=1 failed pre-check (layer-1={pre_layer1.passed}, layer-4={pre_layer4.passed}); "
            "retrying with reps=2 (Phase 3 spec risk #1)"
        )
        ansatz_circuit, multistart = _run_multistart_at_reps(reps=2)
        print(
            f"  reps=2: best={multistart.best.energy:.6f}, "
            f"spread={multistart.spread_best_to_median:.2e}"
        )

    x_star = multistart.best.params

    # --- Step 6: Validation layers 1-5 (noiseless) ---
    print("Step 6: validation layers 1-5 (noiseless)")
    from siam_vqe.analysis import (
        check_ansatz_expressivity,
        check_observable_agreement_multi,
        check_state_overlap,
    )
    layer1 = check_energy_match(multistart.best.energy, float(ed.energies[0]), tol_hartree=1e-3)
    layer2 = check_state_overlap(multistart.best, pop, psi_ed, threshold=0.99)
    layer3 = check_ansatz_expressivity(
        ansatz_circuit, psi_ed, threshold=0.99, seed=multistart_seed, n_starts=4
    )
    layer4 = check_multistart_spread(multistart, tol_mha=1.0)
    obs_dict = observables_l2()
    layer5 = check_observable_agreement_multi(
        multistart.best, pop, psi_ed, obs_dict, num_particles=(3, 3), rel_tol=0.01
    )
    for i, layer_passed in enumerate(
        [layer1.passed, layer2.passed, layer3.passed, layer4.passed, layer5.passed],
        start=1,
    ):
        print(f"  Layer {i}: passed={layer_passed}")

    # --- Diagnostic context for soft layers ---
    if not (layer2.passed and layer3.passed and layer5.passed):
        # Recompute the GS-cluster info to explain why
        cluster_tol = 1e-3
        cluster_size = int(np.sum(np.abs(ed.energies - ed.energies[0]) < cluster_tol))
        cluster_span_uev = float(
            (ed.energies[cluster_size - 1] - ed.energies[0]) * 1e6
        ) if cluster_size > 1 else 0.0
        print(
            f"  [info] GS cluster: {cluster_size} eigenvalues within {cluster_tol*1e3:.0f} meV of E_0 "
            f"(span {cluster_span_uev:.1f} µeV; next genuine gap = "
            f"{float((ed.energies[cluster_size] - ed.energies[0]) * 1e3):.2f} meV). "
            f"Layers 2/3/5 may fire on basis-rotation artifacts within this manifold — see "
            f"docs/superpowers/notes/2026-05-25-l2-degeneracy.md."
        )

    # Hard gate: only layers 1 + 4 are scientifically meaningful for the noisy sweep
    hard_passed = layer1.passed and layer4.passed
    if not hard_passed:
        print(
            f"ABORT: hard gate failed (layer1={layer1.passed}, layer4={layer4.passed}). "
            "Hardware/noisy sweep would be misleading."
        )
        return 1
    if not (layer2.passed and layer3.passed and layer5.passed):
        print(
            "  [warn] Layers 2/3/5 failed but are treated as warnings — see L2-degeneracy note."
        )

    # --- Step 7: Build noisy simulator ---
    print("Step 7: build noisy simulator")
    if args.backend == "fake-marrakesh":
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        backend_or_sim = AerSimulator.from_backend(FakeMarrakesh())
        backend_name = "FakeMarrakesh"
    elif args.backend == "ibm_marrakesh":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        from siam_vqe.hardware import pick_backend
        backend_or_sim = pick_backend(service, min_qubits=6)
        backend_name = backend_or_sim.name
    else:
        raise ValueError(f"Unknown backend: {args.backend!r}")

    # --- Step 8: Transpile circuit ---
    print("Step 8: transpile circuit + observable")
    from siam_vqe.hardware import transpile_for_backend
    bound_circuit = ansatz_circuit.assign_parameters(x_star)
    isa_circuit, isa_pop = transpile_for_backend(bound_circuit, pop, backend_or_sim)
    print(f"  ISA circuit depth: {isa_circuit.depth()}, "
          f"2Q-gate count: {sum(1 for instr in isa_circuit.data if instr.operation.num_qubits == 2)}")

    # --- Step 9: Mitigation sweep ---
    print("Step 9: mitigation sweep")
    from dataclasses import replace

    from siam_vqe.mitigation import make_default_grid
    from siam_vqe.noise import make_mitigated_estimator
    grid = make_default_grid()

    # Override shot count from CLI
    grid = [replace(spec, shots=args.shots) for spec in grid]

    sweep_manifest: dict[str, Any] = {
        "backend": backend_name,
        "shots_per_config": args.shots,
        "configs": {},
        "ed_energy": float(ed.energies[0]),
        "noiseless_vqe_energy": float(multistart.best.energy),
        "x_star": x_star.tolist(),
        "wall_started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for spec in grid:
        out_path = args.output_dir / f"{spec.name}.json"
        if out_path.exists():
            print(f"  [skip] {spec.name} (already exists at {out_path})")
            sweep_manifest["configs"][spec.name] = {"status": "skipped_existing"}
            continue

        print(f"  [run]  {spec.name}: m3={spec.m3}, zne={spec.zne}, "
              f"extrap={spec.zne_extrapolator}, factors={spec.zne_noise_factors}")
        t0 = time.perf_counter()
        try:
            estimator = make_mitigated_estimator(backend_or_sim, spec)
            result = estimator.run([(isa_circuit, isa_pop)]).result()
            energy = float(result[0].data.evs)
            std = float(result[0].data.stds)
            meta = dict(result[0].metadata) if hasattr(result[0], "metadata") else {}

            per_config: dict[str, Any] = {
                "status": "ok",
                "energy": energy,
                "std": std,
                "metadata": _scrub_metadata(meta),
                "walltime_s": time.perf_counter() - t0,
                "spec": {
                    "name": spec.name, "m3": spec.m3, "zne": spec.zne,
                    "zne_extrapolator": spec.zne_extrapolator,
                    "zne_noise_factors": list(spec.zne_noise_factors) if spec.zne_noise_factors else None,
                    "shots": spec.shots,
                },
            }
            print(f"    ⟨H⟩ = {energy:.4f} ± {std:.4f}  ({per_config['walltime_s']:.1f} s)")
        except Exception as e:
            per_config = {
                "status": "error",
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "walltime_s": time.perf_counter() - t0,
                "spec": {
                    "name": spec.name, "m3": spec.m3, "zne": spec.zne,
                    "zne_extrapolator": spec.zne_extrapolator,
                    "zne_noise_factors": list(spec.zne_noise_factors) if spec.zne_noise_factors else None,
                    "shots": spec.shots,
                },
            }
            print(f"    ERROR: {type(e).__name__}: {e}")

        with out_path.open("w") as f:
            json.dump(per_config, f, indent=2)
        sweep_manifest["configs"][spec.name] = per_config

    sweep_manifest["wall_ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest_path = args.output_dir / "_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(sweep_manifest, f, indent=2)
    print(f"\nSweep complete. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
