"""L1 XAS driver smoke — statevector pre-flight.

Phase 5b Task 3c smoke test: verifies the L1 XAS hardware driver script runs
to completion in statevector mode and emits a JSON payload with the expected
schema (backend, channels, peak_energy_per_channel, etc).

The driver is exercised on the statevector backend only here; the
FakeMarrakesh and real-hardware backends are validated manually (Steps 6 and
Task 3d, respectively) — both take minutes to hours and are unsuited to the
unit-test loop.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_l1_xas_hw_statevector_smoke(tmp_path):
    """The driver runs to completion on statevector backend and produces a JSON
    payload with peak_energy_per_channel and per-channel spectra."""
    repo = Path(__file__).resolve().parent.parent
    script = repo / "scripts" / "run_l1_xas_hw.py"
    assert script.exists(), "run_l1_xas_hw.py not yet created"

    out = tmp_path / "smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(script),
            "--backend",
            "statevector",
            "--shots",
            "1024",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=540,
        cwd=repo,
    )
    assert result.returncode == 0, (
        f"driver failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    payload = json.loads(out.read_text())
    for key in (
        "backend",
        "channels",
        "peak_energy_per_channel",
        "sigma_by_channel",
        "omega_grid",
        "delta_e_gs",
        "energy_ground_state",
        "energy_h_prime_ground_state",
        "wall_time_s",
    ):
        assert key in payload, f"missing key {key!r} in output JSON"

    assert payload["backend"] == "statevector"
    assert "lin_z" in payload["peak_energy_per_channel"]
    assert "lin_xy" in payload["peak_energy_per_channel"]
    assert "lin_z" in payload["sigma_by_channel"]
    assert "lin_xy" in payload["sigma_by_channel"]
