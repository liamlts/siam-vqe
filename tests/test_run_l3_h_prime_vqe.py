"""Smoke test for the H' ADAPT-VQE driver."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_h_prime_vqe_smoke(tmp_path):
    """`--mode smoke --seeds 1 --max-ops 2` should complete quickly and
    emit a valid JSON result with the expected top-level keys."""
    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "run_l3_h_prime_vqe.py"
    )
    out = tmp_path / "smoke.json"
    cmd = [
        sys.executable,
        str(script),
        "--mode", "smoke",
        "--seeds", "1",
        "--max-ops", "2",
        "--output", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert out.exists()
    data = json.loads(out.read_text())
    assert "best_energy" in data
    assert data["n_seeds"] == 1
