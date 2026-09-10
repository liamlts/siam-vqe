"""Build notebooks/03_L2_noise_analysis.ipynb programmatically.

Phase 3 deliverable: L2 NiO e_g² SIAM (FakeMarrakesh noise study).
Reads per-config sweep JSONs, runs Layer 6 (stack monotonicity) and
Layer 7 (mitigation effectiveness), renders comparison + ZNE figures.

Style: flat cells.append({...}) mirroring _02_build.py — no helper functions.

Usage
-----
python notebooks/_03_build.py --sweep-dir notebooks/03_L2_sweep_results \\
    [--out-path notebooks/03_L2_noise_analysis.ipynb]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Build 03_L2_noise_analysis.ipynb")
parser.add_argument(
    "--sweep-dir",
    required=True,
    type=Path,
    help="Path to 03_L2_sweep_results/ (contains _manifest.json + per-config JSONs)",
)
parser.add_argument(
    "--out-path",
    type=Path,
    default=Path("siam_vqe/notebooks/03_L2_noise_analysis.ipynb"),
    help="Output .ipynb path (default: siam_vqe/notebooks/03_L2_noise_analysis.ipynb)",
)
args = parser.parse_args()

out_path: Path = args.out_path.resolve()

# Resolve figures dir relative to out_path (notebooks/ -> parent/figures/)
# Used only to pre-create the directory at build time; paths are resolved
# at notebook execution time via Path.cwd() so they are portable.
figures_dir: Path = out_path.parent.parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Compute notebook-relative sweep dir name for runtime resolution.
# args.sweep_dir may be absolute or relative; normalise to a name/subpath
# relative to the notebook directory (notebooks/).
sweep_dir_resolved: Path = args.sweep_dir.resolve()
try:
    sweep_dir_rel: str = str(sweep_dir_resolved.relative_to(out_path.parent))
except ValueError:
    # sweep_dir is outside notebooks/ — fall back to relative-from-cwd form
    sweep_dir_rel = str(sweep_dir_resolved.relative_to(out_path.parent.parent))
    sweep_dir_rel = f"../{sweep_dir_rel}"

# ---------------------------------------------------------------------------
# Build cells
# ---------------------------------------------------------------------------

cells: list[dict[str, object]] = []

# ---- Cell 1: Title (markdown)
cells.append(
    {
        "cell_type": "markdown",
        "source": [
            "# 03 — L2 NiO e_g² SIAM: FakeMarrakesh noise study\n",
            "\n",
            "**Phase 3 deliverable.** Reads per-config sweep results from\n",
            "`03_L2_sweep_results/`, runs Layer 6 (stack monotonicity M3 → M3+ZNE)\n",
            "and Layer 7 (mitigation effectiveness — hard gate), and renders\n",
            "the comparison bar chart + ZNE extrapolation curves.\n",
            "\n",
            "**Configs swept:** no_mit, m3_only, zne_lin_135, m3_zne_lin_135,\n",
            "m3_zne_exp_135, m3_zne_poly3_135, m3_zne_lin_123, m3_zne_lin_12345.\n",
        ],
    }
)

# ---- Cell 2: Imports + load manifest (code)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Jupyter sets cwd to the notebook directory; the project root is one level up.\n",
            "_PROJECT_ROOT = Path.cwd().parent  # siam_vqe/, from notebooks/ cwd\n",
            "if str(_PROJECT_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(_PROJECT_ROOT))\n",
            "\n",
            "import json\n",
            "\n",
            "import matplotlib\n",
            "matplotlib.use('Agg')  # non-interactive backend for nbconvert\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            f"sweep_dir = Path.cwd() / '{sweep_dir_rel}'\n",
            "\n",
            "manifest = json.loads((sweep_dir / '_manifest.json').read_text())\n",
            "ed_energy: float = manifest['ed_energy']\n",
            "noiseless_energy: float = manifest['noiseless_vqe_energy']\n",
            "x_star: list[float] = manifest['x_star']\n",
            "\n",
            "print(f'ED energy:        {ed_energy:.8f} eV')\n",
            "print(f'noiseless VQE:    {noiseless_energy:.8f} eV')\n",
            "print(f'|ΔE| noiseless:   {abs(noiseless_energy - ed_energy):.2e} eV')\n",
            "print(f'sweep started:    {manifest[\"wall_started\"]}')\n",
            "print(f'sweep ended:      {manifest[\"wall_ended\"]}')\n",
            "print(f'configs in manifest: {list(manifest[\"configs\"].keys())}')\n",
        ],
    }
)

# ---- Cell 3: Load per-config sweep results (markdown)
cells.append(
    {
        "cell_type": "markdown",
        "source": ["## Load per-config sweep results\n"],
    }
)

# ---- Cell 4: Load each config JSON (code)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "configs: dict[str, dict] = {}\n",
            "\n",
            "for name in manifest['configs']:\n",
            "    p = sweep_dir / f'{name}.json'\n",
            "    if not p.exists():\n",
            "        print(f'  {name}: FILE MISSING — skipping')\n",
            "        continue\n",
            "    res = json.loads(p.read_text())\n",
            "    configs[name] = res\n",
            "    if res.get('status') == 'ok':\n",
            "        print(f'  {name}: E = {res[\"energy\"]:.6f} ± {res[\"std\"]:.6f} eV')\n",
            "    else:\n",
            "        print(f'  {name}: ERROR — {res.get(\"error\", \"unknown\")}')\n",
            "\n",
            "n_ok = sum(1 for r in configs.values() if r.get('status') == 'ok')\n",
            "n_err = len(configs) - n_ok\n",
            "print(f'\\nLoaded {len(configs)} configs: {n_ok} ok, {n_err} errors')\n",
        ],
    }
)

# ---- Cell 5: Layer 6 markdown
cells.append(
    {
        "cell_type": "markdown",
        "source": [
            "## Layer 6: stack monotonicity (M3 → M3+ZNE)\n",
            "\n",
            "Pass-or-flag (not a hard gate). Checks that adding ZNE on top of M3\n",
            "does not regress the gap to ED relative to M3 alone.\n",
        ],
    }
)

# ---- Cell 6: Layer 6 check (code)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "from siam_vqe.analysis import check_stack_monotonicity\n",
            "\n",
            "m3_ok = configs.get('m3_only', {}).get('status') == 'ok'\n",
            "m3_zne_ok = configs.get('m3_zne_lin_135', {}).get('status') == 'ok'\n",
            "\n",
            "if m3_ok and m3_zne_ok:\n",
            "    l6 = check_stack_monotonicity(\n",
            "        e_ed=ed_energy,\n",
            "        e_m3=configs['m3_only']['energy'],\n",
            "        e_m3_zne=configs['m3_zne_lin_135']['energy'],\n",
            "    )\n",
            "    print(f'Layer 6 (stack monotonicity): passed={l6.passed}')\n",
            "    print(f'  gap_m3       = {l6.gap_m3:.6f} eV')\n",
            "    print(f'  gap_m3+zne   = {l6.gap_m3_zne:.6f} eV')\n",
            "    print(f'  notes: {l6.notes}')\n",
            "else:\n",
            "    print('Layer 6: SKIPPED — m3_only or m3_zne_lin_135 not ok')\n",
            "    l6 = None\n",
        ],
    }
)

# ---- Cell 7: Layer 7 markdown
cells.append(
    {
        "cell_type": "markdown",
        "source": [
            "## Layer 7: mitigation effectiveness (Phase 3 headline)\n",
            "\n",
            "**Hard gate.** At least one mitigation config must have\n",
            "|E_mit - E_ED| / |E_no_mit - E_ED| < 1.0.\n",
        ],
    }
)

# ---- Cell 8: Layer 7 check (code)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "from siam_vqe.analysis import check_mitigation_effectiveness\n",
            "\n",
            "l7 = check_mitigation_effectiveness(configs, ed_energy=ed_energy)\n",
            "print(f'Layer 7 (mitigation effectiveness): passed={l7.passed}')\n",
            "print(f'  best_config:  {l7.best_config}')\n",
            "print(f'  best_ratio:   {l7.best_ratio:.6f}')\n",
            "print()\n",
            "print('Per-config ratios (sorted, ascending):')\n",
            "for name, ratio in sorted(l7.per_config_ratio.items(), key=lambda kv: kv[1]):\n",
            "    print(f'  {name:30s}: {ratio:.6f}')\n",
            "\n",
            "assert l7.passed, f'Layer 7 FAILED — best ratio {l7.best_ratio:.4f} >= 1.0'\n",
        ],
    }
)

# ---- Cell 9: Figure 1 markdown
cells.append(
    {
        "cell_type": "markdown",
        "source": ["## Figure 1: mitigation comparison bar chart\n"],
    }
)

# ---- Cell 10: Figure 1 code
cells.append(
    {
        "cell_type": "code",
        "source": [
            "from siam_vqe.analysis import compare_mitigations\n",
            "\n",
            "fig_mit_png = str(Path.cwd().parent / 'figures' / '03_L2_mitigation_comparison.png')\n",
            "fig_mit_pdf = str(Path.cwd().parent / 'figures' / '03_L2_mitigation_comparison.pdf')\n",
            "\n",
            "fig1 = compare_mitigations(\n",
            "    configs,\n",
            "    ed_energy=ed_energy,\n",
            "    title='L2 NiO e_g\\u00b2 SIAM — mitigation comparison (FakeMarrakesh)',\n",
            ")\n",
            "fig1.savefig(fig_mit_png, dpi=150)\n",
            "fig1.savefig(fig_mit_pdf)\n",
            "plt.close(fig1)\n",
            "print(f'Saved {fig_mit_png}')\n",
            "print(f'Saved {fig_mit_pdf}')\n",
        ],
    }
)

# ---- Cell 11: Figure 2 markdown
cells.append(
    {
        "cell_type": "markdown",
        "source": ["## Figure 2: ZNE extrapolation curves\n"],
    }
)

# ---- Cell 12: Figure 2 code — flat ZNE metadata extraction
cells.append(
    {
        "cell_type": "code",
        "source": [
            "from siam_vqe.analysis import plot_zne_extrapolation_curves\n",
            "\n",
            "fig_zne_png = str(Path.cwd().parent / 'figures' / '03_L2_zne_curves.png')\n",
            "fig_zne_pdf = str(Path.cwd().parent / 'figures' / '03_L2_zne_curves.pdf')\n",
            "\n",
            "diagnostics: dict[str, dict] = {}\n",
            "for name, res in configs.items():\n",
            "    if res.get('status') != 'ok':\n",
            "        continue\n",
            "    md = res.get('metadata', {})\n",
            "    if 'zne_raw_values' in md:\n",
            "        diagnostics[name] = {\n",
            "            'noise_factors': md['zne_noise_factors'],\n",
            "            'raw_values': md['zne_raw_values'],\n",
            "            'extrapolator': md['zne_extrapolator'],\n",
            "            'extrapolated': res['energy'],\n",
            "        }\n",
            "\n",
            "print(f'ZNE diagnostics populated for {len(diagnostics)} configs:')\n",
            "for name in diagnostics:\n",
            "    print(f'  {name}')\n",
            "\n",
            "fig2 = plot_zne_extrapolation_curves(\n",
            "    diagnostics,\n",
            "    title='L2 NiO e_g\\u00b2 SIAM — ZNE extrapolation curves (FakeMarrakesh)',\n",
            ")\n",
            "fig2.savefig(fig_zne_png, dpi=150)\n",
            "fig2.savefig(fig_zne_pdf)\n",
            "plt.close(fig2)\n",
            "print(f'Saved {fig_zne_png}')\n",
            "print(f'Saved {fig_zne_pdf}')\n",
        ],
    }
)

# ---- Cell 13: Summary JSON markdown
cells.append(
    {
        "cell_type": "markdown",
        "source": ["## Write 03_L2_summary.json\n"],
    }
)

# ---- Cell 14: Write summary JSON (code)
cells.append(
    {
        "cell_type": "code",
        "source": [
            "summary = {\n",
            "    'phase': 3,\n",
            "    'level': 'L2',\n",
            "    'ed_energy': ed_energy,\n",
            "    'noiseless_vqe_energy': noiseless_energy,\n",
            "    'best_mitigation_config': l7.best_config,\n",
            "    'best_mitigation_ratio': l7.best_ratio,\n",
            "    'layer_7_passed': bool(l7.passed),\n",
            "    'layer_7_per_config_ratio': l7.per_config_ratio,\n",
            "    'sweep_dir': str(sweep_dir),\n",
            "    'sweep_manifest_started': manifest['wall_started'],\n",
            "    'sweep_manifest_ended': manifest['wall_ended'],\n",
            "}\n",
            "\n",
            "summary_path = sweep_dir.parent / '03_L2_summary.json'\n",
            "summary_path.write_text(json.dumps(summary, indent=2))\n",
            "print(f'Wrote {summary_path}')\n",
            "print(json.dumps(summary, indent=2))\n",
        ],
    }
)

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------

notebook: dict[str, object] = {
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

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=2) + "\n")
print(f"wrote {out_path}")
