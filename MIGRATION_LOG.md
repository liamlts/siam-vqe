# Qiskit 2.x migration — change log

Branch `migrate/qiskit-2x`, 2026-09-10.

## Why this was small

The migration touches 7 files and about 25 lines. That is not an oversight — it
is a consequence of Phase 5b. The single largest Qiskit-2.0 breaking change for
this codebase, the removal of `backend.run()` in `qiskit_ibm_runtime`, was
already absorbed then: the hardware-reachable path was unified on a
`_sample_counts(mode, ...)` seam over `SamplerV2`. By the time Qiskit 2.x
arrived, the expensive part had been paid.

The whole dependency set resolves on **qiskit 2.5.2** with no conflicts, and the
package imported and ran against it before a single line was changed.

## Dependency versions

| Package | Before | After | Note |
|---|---|---|---|
| qiskit | `~=1.2` (1.4.5) | `>=2.3,<3` (2.5.2) | capped below 3.0 — see risk below |
| qiskit-nature | `~=0.7` (0.7.2) | `>=0.8,<0.9` (0.8.0) | 0.8 declares `qiskit<3.0,>=1.4` |
| qiskit-ibm-runtime | `>=0.25` (0.44.1) | `>=0.49` (0.49.0) | 0.49 *requires* `qiskit>=2.3` |
| qiskit-aer | `>=0.15` (0.17.2) | `>=0.17` (0.17.2) | version unchanged, floor raised |
| qiskit-algorithms | `>=0.3` (0.4.0) | `>=0.4` (0.4.0) | version unchanged, floor raised |
| mthree | `>=2.6` (3.0.0) | `>=2.6` (3.0.0) | untouched |
| Python | `>=3.11` | `>=3.12` | forced by numpy, not by Qiskit |

`qiskit-ibm-runtime` 0.49 is what actually forces Qiskit 2.x; nothing else in
the set requires it.

## Code changes

### `siam_vqe/ansatz.py` — the only real API change

`EfficientSU2` (the class) is deprecated as of Qiskit 2.1 and slated for removal
in 3.0. Replaced with the `efficient_su2()` function form. The class form
required a `.decompose()` call, which the function form does not, so that line
is gone.

Verified equivalent before switching, across `(nq, reps, entanglement)` =
`(2,2,linear)`, `(4,3,linear)`, `(6,2,full)`:

- identical parameter count, circuit depth, and operation count
- **bit-identical statevectors** under the same bound parameter vector
  (`|<a|b>| = 1.000000000000`, `np.allclose` true)

This was the only deprecation warning in the entire codebase originating from
`siam_vqe` itself. After the change there are none.

### `siam_vqe/noise.py` — docstring accuracy

Two corrections, no behaviour change. The API-note version banner said
"Qiskit ≥1.2 / Aer ≥0.15". More substantively, the module claimed
`AerSimulator.from_backend()` extracts "calibration / readout / pulse-error
data" — with the Pulse package removed in Qiskit 2.0 that is no longer true,
and the `Target` is now the sole source. Corrected rather than left to mislead.

### `siam_vqe/hardware.py` — docstring accuracy

Resilience-level note referenced `qiskit-ibm-runtime >= 0.25`; now 0.49.

## Configuration changes

### `pyproject.toml`

Dependency pins per the table above, plus:

- `requires-python` 3.11 → 3.12; `[tool.ruff] target-version` py311 → py312;
  `[tool.mypy] python_version` 3.11 → 3.12.
- **Three `filterwarnings` entries deleted.** They suppressed
  `Instruction.condition`, `DAGCircuit.calibrations`, and
  `QuantumCircuit.calibrations` — all APIs that Qiskit 2.0 removed outright, so
  the suppressions were dead code matching warnings that can no longer be
  emitted.
- **One `filterwarnings` entry added**, for the live issue described under
  risks below.

### `.github/workflows/test.yml`

Python 3.11 → 3.12. No other change; the workflow already installed the latest
resolvable stack, so it picks up Qiskit 2.x from the pins.

### `README.md`

Install section now states Python 3.12+ and Qiskit 2.x rather than advertising
the migration as pending. The roadmap entry for it is removed, since it is done.

## The Python floor: why 3.12

Nothing in Qiskit forces this. Qiskit 2.5.2 declares `>=3.10`, and
qiskit-nature 0.8 and qiskit-ibm-runtime 0.49 both declare `>=3.10`. The
migration works on 3.11.

**numpy** is the constraint. numpy 2.5.3 requires Python `>=3.12`; on 3.11 the
newest available wheel is 2.4.6. That is not merely cosmetic: numpy 2.5's type
stubs use PEP 695 `type` statements, so `mypy` targeting 3.11 fails to parse
them outright —

```
numpy/__init__.pyi:737: error: Type statement is only supported in Python 3.12 and greater
```

Staying on 3.11 was a legitimate option (CI would cap at numpy 2.4.6 and
type-check cleanly). Bumping was chosen deliberately so the project tracks the
current scientific stack rather than only the current Qiskit.

## Known risk, deliberately not fixed here

**qiskit-nature 0.8 will break on Qiskit 3.0.** Its `UCCSD` and `HartreeFock`
are still built on `BlueprintCircuit`, deprecated in Qiskit 2.1 and scheduled
for removal in 3.0:

```
DeprecationWarning: The class ``qiskit.circuit.library.blueprintcircuit.BlueprintCircuit``
is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0.
```

Nothing in `siam_vqe` touches `BlueprintCircuit` — this is entirely inside
qiskit-nature, and there is no local fix. Two consequences, both intentional:

1. The qiskit pin is capped at `<3` rather than left open.
2. A `filterwarnings` entry suppresses it, replacing the three dead ones. It is
   annotated in `pyproject.toml` so the reason is not lost, and it should be
   deleted once qiskit-nature moves off `BlueprintCircuit`.

This is the direct successor to the problem this migration solved, and it is
upstream's to resolve.

## Verification

Run against qiskit 2.5.2 / aer 0.17.2 / nature 0.8.0 / algorithms 0.4.0 /
ibm-runtime 0.49.0 / mthree 3.0.0 in a clean virtual environment.

CI numbers below are from the same GitHub Actions runner type, so the
comparison is like-for-like: `main` on Python 3.11 / qiskit 1.4.5 versus this
branch on Python 3.12.14 / qiskit 2.5.2.

| Gate | Qiskit 1.4.5 (baseline) | Qiskit 2.5.2 |
|---|---|---|
| Full suite | 256 passed / 1 xfailed | **256 passed / 1 xfailed** |
| Pytest wall time (CI) | 1112.93 s (18m32s) | **633.60 s (10m33s)** |
| Warnings (CI) | 71 | **10** |
| `ruff check siam_vqe` | clean | clean |
| `mypy siam_vqe` | clean (target 3.11) | clean (target 3.12), 20 files |
| Deprecations owned by `siam_vqe` | 1 (`EfficientSU2`) | **0** |

An unplanned result worth recording: the suite got **43% faster** (1113 s →
634 s) and warnings dropped 86% (71 → 10). Neither was a goal of this work.
The speedup is upstream — Qiskit 2.x internals plus the removal of the
deprecated shim paths the 1.x stack was routing through — not any change to
test scope, since the test counts are identical.

The test result is identical to the baseline — no test was skipped, xfailed, or
rewritten to accommodate 2.x. The single `x` is the pre-existing xfail.

Two honest caveats on the local verification, neither of which CI inherits:

1. The local interpreter was Python 3.13, not the 3.12 floor now declared. CI
   runs 3.12 and is the binding check.
2. The full-suite run was launched after the `ansatz.py` and `pyproject`
   dependency/`filterwarnings` changes but before the Python-floor bump and the
   docstring corrections. Those later edits are build metadata and comments
   with no effect on test execution; `ruff` and `mypy` above were re-run
   against the final state.

## Not in scope

- `run_vqe_multistart`'s legacy path still builds a throwaway ansatz solely to
  size `x0`, and raises if it mismatches the real one.
- `StatevectorEstimator(seed=...)` in `run_vqe` remains a no-op at exact
  precision.
- CI still runs a single Python version with no scheduled run.
- **The private `research-cowork` copy of this package still pins
  `qiskit~=1.2`.** Once this lands, the two diverge and the private tree becomes
  the stale one. That inversion is worth resolving.
