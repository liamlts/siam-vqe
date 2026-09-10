"""Driver smoke test for the L3 XAS simulator (Phase 5 Task 18).

Requires:
- notebooks/04_L3_vqe_result.json (Phase 4 ψ_GS replay payload)
- notebooks/05_L3_h_prime_vqe_result.json (Phase 5 ψ'_GS replay payload)
- data/edrixs_xas_l3_reference.npz (EDRIXS-style reference NPZ)

If any of these are missing the test is skipped.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_run_l3_xas_sim_produces_json(tmp_path):
    here = Path(__file__).resolve().parent.parent
    script = here / "scripts" / "run_l3_xas_sim.py"
    needed = [
        here / "notebooks/04_L3_vqe_result.json",
        here / "notebooks/05_L3_h_prime_vqe_result.json",
        here / "data/edrixs_xas_l3_reference.npz",
    ]
    if not all(p.exists() for p in needed):
        pytest.skip("Prerequisite artifacts missing.")

    out = tmp_path / "xas.json"
    cmd = [
        sys.executable,
        str(script),
        "--output",
        str(out),
        "--figures-dir",
        str(tmp_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, cwd=here
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    data = json.loads(out.read_text())
    assert "phase5_pass" in data
