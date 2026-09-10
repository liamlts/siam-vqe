"""Render production figures + summary against the full-shot sweep.

Standalone twin of the analysis notebook for the Task 18 production data.
Reads notebooks/03_L2_sweep_results_full/_manifest.json, runs Layer 6 + 7,
saves figures with _full suffix, writes 03_L2_summary_full.json. The smoke
artifacts remain in place for A/B comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Bootstrap sys.path for direct invocation (editable install points elsewhere)
_PKG_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_PARENT))

from siam_vqe.analysis import (  # noqa: E402
    compare_mitigations,
    plot_zne_extrapolation_curves,
    check_stack_monotonicity,
    check_mitigation_effectiveness,
)


NB_DIR = Path(__file__).resolve().parent
SWEEP_DIR = NB_DIR / "03_L2_sweep_results_full"
FIG_DIR = NB_DIR.parent / "figures"
SUMMARY_PATH = NB_DIR / "03_L2_summary_full.json"

assert SWEEP_DIR.exists(), f"missing {SWEEP_DIR}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Load manifest + per-config results
manifest = json.loads((SWEEP_DIR / "_manifest.json").read_text())
ed_energy = manifest["ed_energy"]
results = manifest["configs"]
print(f"Loaded {len(results)} configs from {SWEEP_DIR.name}")
print(f"ED reference: {ed_energy:.6f} eV")
print(f"Sweep wall time: {manifest['wall_started']} -> {manifest['wall_ended']}")

# --- Layer 7: mitigation effectiveness (hard gate)
l7 = check_mitigation_effectiveness(results, ed_energy)
print()
print("=== Layer 7 (hard gate) ===")
print(f"  passed:      {l7.passed}")
print(f"  best config: {l7.best_config}")
print(f"  best ratio:  {l7.best_ratio:.4f}")
print("  per-config ratios:")
for name, r in sorted(l7.per_config_ratio.items(), key=lambda kv: kv[1]):
    print(f"    {name:24s}  {r:.4f}")

# --- Layer 6: stack monotonicity (pass-or-flag)
# Compare m3_only -> m3_zne_<best nonlinear> as in the smoke run analysis.
# Use the same comparator as the smoke build: m3_only vs m3_zne_exp_135.
l6 = check_stack_monotonicity(
    e_ed=ed_energy,
    e_m3=results["m3_only"]["energy"],
    e_m3_zne=results["m3_zne_exp_135"]["energy"],
)
print()
print("=== Layer 6 (stack monotonicity, m3_only vs m3_zne_exp_135) ===")
print(f"  passed: {l6.passed}")
print(f"  gap M3:        {l6.gap_m3:.4f} eV")
print(f"  gap M3+ZNE:    {l6.gap_m3_zne:.4f} eV")
print(f"  notes:         {l6.notes}")

# --- Mitigation comparison figure
fig1 = compare_mitigations(
    results,
    ed_energy=ed_energy,
    title="L2 NiO e_g² SIAM — mitigation comparison (FakeMarrakesh, 8192 shots)",
)
png1 = FIG_DIR / "03_L2_mitigation_comparison_full.png"
pdf1 = FIG_DIR / "03_L2_mitigation_comparison_full.pdf"
fig1.savefig(png1, dpi=150, bbox_inches="tight", facecolor="white")
fig1.savefig(pdf1, bbox_inches="tight", facecolor="white")
print(f"\nwrote {png1.name} + {pdf1.name}")

# --- ZNE extrapolation curves
# Build diagnostics dict in the shape plot_zne_extrapolation_curves expects.
diagnostics: dict[str, dict] = {}
for name, res in results.items():
    md = res.get("metadata", {})
    if "zne_raw_values" not in md:
        continue
    diagnostics[name] = {
        "noise_factors": md["zne_noise_factors"],
        "raw_values": md["zne_raw_values"],
        "extrapolator": md["zne_extrapolator"],
        "extrapolated": res["energy"],
    }
fig2 = plot_zne_extrapolation_curves(
    diagnostics,
    title="L2 NiO e_g² SIAM — ZNE extrapolation curves (FakeMarrakesh, 8192 shots)",
)
png2 = FIG_DIR / "03_L2_zne_curves_full.png"
pdf2 = FIG_DIR / "03_L2_zne_curves_full.pdf"
fig2.savefig(png2, dpi=150, bbox_inches="tight", facecolor="white")
fig2.savefig(pdf2, bbox_inches="tight", facecolor="white")
print(f"wrote {png2.name} + {pdf2.name}")

# --- Production summary JSON
summary = {
    "phase": 3,
    "level": "L2",
    "shots_per_config": manifest["shots_per_config"],
    "ed_energy": ed_energy,
    "noiseless_vqe_energy": manifest["noiseless_vqe_energy"],
    "layer_6_passed": l6.passed,
    "layer_6_gap_m3": l6.gap_m3,
    "layer_6_gap_m3_zne": l6.gap_m3_zne,
    "layer_6_notes": l6.notes,
    "layer_7_passed": l7.passed,
    "best_mitigation_config": l7.best_config,
    "best_mitigation_ratio": l7.best_ratio,
    "layer_7_per_config_ratio": l7.per_config_ratio,
    "sweep_dir": str(SWEEP_DIR),
    "wall_started": manifest["wall_started"],
    "wall_ended": manifest["wall_ended"],
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
print(f"wrote {SUMMARY_PATH.name}")

print()
print("done.")
