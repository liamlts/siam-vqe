"""Smoke test for run_l3_adapt.py CLI driver."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_l3_adapt_smoke_mode_produces_json(tmp_path):
    """`--mode smoke --seeds 1 --max-ops 2` should complete in < 10 min and
    emit a valid JSON result."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_l3_adapt.py"
    out = tmp_path / "smoke.json"
    cmd = [sys.executable, str(script), "--mode", "smoke",
           "--seeds", "1", "--max-ops", "2", "--output", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert out.exists()
    data = json.loads(out.read_text())
    assert "final_energy" in data
    assert "operators_picked" in data
    assert "validation_layers" in data
