#!/usr/bin/env python
"""Build the H' scipy-ED reference NPZ at `data/nio_l3_h_prime_reference.npz`.

Runs once; output is committed to the repo so the production runs don't
have to ED 18-qubit H' from scratch each time.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root = parent of this scripts/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from siam_vqe.core_hole import CoreHoleParams
from siam_vqe.hamiltonian_l3 import L3Params
from siam_vqe.reference_l3 import compute_l3_xas_reference, save_l3_reference


def main() -> None:
    params = L3Params()
    ch = CoreHoleParams()
    print("[build_h_prime_ed_reference] Computing H' ED reference in (10, 9) ...")
    # Sector (10, 9) dim = C(10, 10) * C(10, 9) = 1 * 10 = 10; ARPACK
    # requires k < N - 1. Cap k_states at 8.
    ref = compute_l3_xas_reference(params, ch, num_particles=(10, 9), k_states=8)
    out = _PROJECT_ROOT / "data" / "nio_l3_h_prime_reference.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_l3_reference(ref, out)
    print(f"[build_h_prime_ed_reference] Saved: E_0 = {ref.ground_energy:.6f} eV -> {out}")


if __name__ == "__main__":
    main()
