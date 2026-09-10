"""Tests for run_adapt_multistart driver."""
from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from siam_vqe.adapt_vqe import AdaptConfig
from siam_vqe.vqe_runner import AdaptMultistartResult, run_adapt_multistart


def test_run_adapt_multistart_returns_multi_result():
    """Tiny driver smoke: 2 seeds on H_2 (4 qubits) with hand-built pool."""
    H = SparsePauliOp.from_list([
        ("IIII", -0.81054), ("IIIZ", 0.17219), ("IIZI", -0.22575),
        ("IZII", 0.17219), ("ZIII", -0.22575), ("IIZZ", 0.12091),
        ("IZIZ", 0.16893), ("IZZI", 0.16615), ("ZIIZ", 0.16615),
        ("ZIZI", 0.17464), ("ZZII", 0.12091), ("XXXX", 0.04523),
        ("YYYY", 0.04523), ("XXYY", -0.04523), ("YYXX", -0.04523),
    ])
    pool = [
        SparsePauliOp.from_list([("YXXX", 1.0)]),
        SparsePauliOp.from_list([("XYXX", 1.0)]),
        SparsePauliOp.from_list([("XXYX", 1.0)]),
        SparsePauliOp.from_list([("XXXY", 1.0)]),
    ]
    psi_seeds = [
        np.eye(16, dtype=complex)[3],   # |0011>
        np.eye(16, dtype=complex)[5],   # |0101> (different seed)
    ]
    config = AdaptConfig(gradient_threshold=1e-6, max_operators=8,
                         inner_max_iter=100)
    result = run_adapt_multistart(H, pool, psi_seeds, config)
    assert isinstance(result, AdaptMultistartResult)
    assert len(result.per_seed) == 2
    assert result.best_seed_index in {0, 1}
    assert result.best_energy <= result.median_energy <= result.worst_energy


def test_run_adapt_multistart_spread_computed_correctly():
    """Best/median/worst should match np.min/median/max of per-seed energies."""
    H = SparsePauliOp.from_list([("IIII", -1.0)])  # trivial diagonal
    pool = [SparsePauliOp.from_list([("XIII", 1.0)])]
    psi_seeds = [
        np.eye(16, dtype=complex)[i] for i in range(3)
    ]
    config = AdaptConfig(gradient_threshold=1e-6, max_operators=2,
                         inner_max_iter=50)
    result = run_adapt_multistart(H, pool, psi_seeds, config)
    energies = sorted(r.final_energy for r in result.per_seed)
    assert result.best_energy == pytest.approx(energies[0])
    assert result.worst_energy == pytest.approx(energies[-1])
