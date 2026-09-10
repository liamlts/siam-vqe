#!/usr/bin/env bash
# launch_l2_noise_sweep_full.sh — kick off the Phase 3 Task 18 production sweep
#
# Runs the L2 noise sweep at the default 8192-shot setting against
# FakeMarrakesh with seed 20260525 (matching the smoke run for direct A/B).
# Output goes to siam_vqe/notebooks/03_L2_sweep_results_full/<spec>.json.
# Log goes to siam_vqe/logs/sweep_full_<timestamp>.log.
#
# Wall time budget: ~3-4 hours (5 of 8 configs use M3 + multi-noise-factor
# ZNE; m3_zne_lin_12345 with 5 factors at 8192 shots is the slowest at
# probably 30-45 min).
#
# Uses nohup so it survives terminal close. Skip-if-exists logic in the
# sweep loop means you can safely Ctrl-C and re-launch — completed configs
# will be skipped on the next run.
#
# Run from anywhere; the script cd's to the worktree's siam_vqe/ root.

set -euo pipefail

WORKTREE_ROOT="${WORKTREE_ROOT:-$(git rev-parse --show-toplevel)}"
PKG_ROOT="${WORKTREE_ROOT}/siam_vqe"
PYTHON="/opt/local/bin/python3.13"

# Verify worktree state before launching
cd "${WORKTREE_ROOT}"
BRANCH=$(git branch --show-current)
HEAD_SHA=$(git rev-parse --short HEAD)
if [[ "${BRANCH}" != "siam-vqe/phase-3-l2-noise-study" ]]; then
    echo "ERROR: expected branch 'siam-vqe/phase-3-l2-noise-study', got '${BRANCH}'" >&2
    echo "Refusing to launch the sweep. Resolve the branch state first." >&2
    exit 1
fi

# Verify python + mthree + the package
"${PYTHON}" -c "import mthree, qiskit, qiskit_aer, qiskit_ibm_runtime, numpy; print('deps OK')" || {
    echo "ERROR: required Python deps missing. Run 'pip install --user mthree>=2.6'." >&2
    exit 2
}

cd "${PKG_ROOT}"
mkdir -p notebooks/03_L2_sweep_results_full logs

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
LOG_FILE="${PKG_ROOT}/logs/sweep_full_${TIMESTAMP}.log"
PID_FILE="${PKG_ROOT}/logs/sweep_full.pid"

# If a previous run is still alive, refuse to start a second one
if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "${OLD_PID}" >/dev/null 2>&1; then
        echo "ERROR: an existing sweep is still running (PID ${OLD_PID})." >&2
        echo "Use 'kill ${OLD_PID}' to stop it first, or just wait." >&2
        echo "Tail its log: tail -f ${PKG_ROOT}/logs/sweep_full_*.log" >&2
        exit 3
    fi
    rm -f "${PID_FILE}"
fi

echo "Launching Phase 3 Task 18 production sweep…"
echo "  Worktree:    ${WORKTREE_ROOT}"
echo "  Branch:      ${BRANCH} @ ${HEAD_SHA}"
echo "  Python:      ${PYTHON}"
echo "  Output dir:  ${PKG_ROOT}/notebooks/03_L2_sweep_results_full/"
echo "  Log file:    ${LOG_FILE}"
echo "  Seed:        20260525 (matches smoke run for direct A/B)"
echo "  Shots:       8192 (default)"
echo ""

nohup "${PYTHON}" -u scripts/run_l2_noise_sweep.py \
    --output-dir notebooks/03_L2_sweep_results_full \
    --backend fake-marrakesh \
    --seed 20260525 \
    > "${LOG_FILE}" 2>&1 &

SWEEP_PID=$!
echo "${SWEEP_PID}" > "${PID_FILE}"

echo "PID ${SWEEP_PID} — disowned, will survive terminal close."
echo ""
echo "Monitor:    tail -f ${LOG_FILE}"
echo "Check progress: ls ${PKG_ROOT}/notebooks/03_L2_sweep_results_full/*.json | wc -l"
echo "Stop early: kill ${SWEEP_PID}"
echo ""
echo "When the sweep finishes (8 JSONs + _manifest.json in the output dir),"
echo "regenerate figures + summary against the full data by rerunning the"
echo "analysis notebook against notebooks/03_L2_sweep_results_full/ — see"
echo "docs/superpowers/notes/2026-05-25-task18-launch.md for the resume steps."
