"""Tests for siam_vqe.analysis — validation layers and plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import pytest

from siam_vqe.analysis import (
    EnergyMatchReport,
    ExpressivityReport,
    MultiStartReport,
    ObservableReport,
    OverlapReport,
    check_ansatz_expressivity,
    check_energy_match,
    check_multistart_spread,
    check_observable_agreement,
    check_state_overlap,
    plot_convergence,
)
from siam_vqe.ansatz import efficient_su2_ansatz
from siam_vqe.hamiltonian import hubbard_dimer, observables_dimer
from siam_vqe.mappings import to_qubit_op
from siam_vqe.reference_ed import exact_diag
from siam_vqe.vqe_runner import run_vqe, run_vqe_multistart


@pytest.fixture
def vqe_and_ed(dimer_params: dict[str, float]) -> tuple[object, object, object]:
    """Shared converged VQE + ED result on the Hubbard dimer."""
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    circuit, x0 = efficient_su2_ansatz(num_qubits=pop.num_qubits, reps=3, seed=20260524)
    vqe = run_vqe(pop, circuit, x0, optimizer="COBYLA", maxiter=600)
    ed = exact_diag(pop, k=1)
    return vqe, ed, pop


def test_check_energy_match_passes(vqe_and_ed: tuple[object, object, object]) -> None:
    vqe, ed, _ = vqe_and_ed
    report = check_energy_match(vqe.energy, ed.energies[0], tol_hartree=1e-3)
    assert isinstance(report, EnergyMatchReport)
    assert report.passed is True
    assert abs(report.delta) < 1e-3
    # Check backward-compat aliases.
    assert report.delta_e == report.delta
    # Check abs_tol kwarg alias.
    report2 = check_energy_match(vqe.energy, ed.energies[0], abs_tol=1e-3)
    assert report2.passed is True


def test_check_state_overlap_returns_overlap(vqe_and_ed: tuple[object, object, object]) -> None:
    vqe, ed, _pop = vqe_and_ed
    report = check_state_overlap(vqe, ed, threshold=0.95)
    assert isinstance(report, OverlapReport)
    assert 0.0 <= report.overlap <= 1.0
    # On a 2-qubit dimer with EfficientSU2 reps=3, we expect very high overlap.
    assert report.overlap > 0.95
    # Check overlap_sq alias.
    assert report.overlap_sq == report.overlap
    # Check notebook-API call: (vqe, pop, psi_ed_vector).
    report2 = check_state_overlap(vqe, _pop, ed.vectors[:, 0], threshold=0.95)
    assert report2.overlap == pytest.approx(report.overlap, abs=1e-10)


def test_check_observable_agreement_n_total(
    vqe_and_ed: tuple[object, object, object],
    dimer_params: dict[str, float],
) -> None:
    vqe, ed, _pop = vqe_and_ed
    n_total_fop = observables_dimer()["n_total"]
    n_total_pop = to_qubit_op(n_total_fop, scheme="parity_tapered", num_particles=(1, 1))
    report = check_observable_agreement(
        vqe, ed, observable=n_total_pop, rel_tol=0.05, name="n_total"
    )
    assert isinstance(report, ObservableReport)
    assert report.vqe_value == pytest.approx(2.0, abs=0.1)
    assert report.ed_value == pytest.approx(2.0, abs=1e-8)
    assert report.passed is True


def test_check_multistart_spread(dimer_params: dict[str, float]) -> None:
    fop = hubbard_dimer(**dimer_params)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    circuit, _ = efficient_su2_ansatz(num_qubits=pop.num_qubits, reps=2, seed=0)
    multistart = run_vqe_multistart(
        pop, circuit, n_starts=4, ansatz_factory_seed_base=100,
        ansatz_num_qubits=pop.num_qubits, ansatz_reps=2,
        optimizer="COBYLA", maxiter=150,
    )
    # Original API: pass MultistartResult (or list), max_spread_hartree kwarg.
    report = check_multistart_spread(multistart, max_spread_hartree=0.1)
    assert isinstance(report, MultiStartReport)
    assert report.n_starts == 4
    assert report.passed is True
    # Notebook API: tol_mha kwarg (100 mHa threshold).
    report2 = check_multistart_spread(multistart, tol_mha=100.0)
    assert report2.passed is True
    # spread_best_to_median alias.
    assert report.spread_best_to_median >= 0.0


def test_plot_convergence_returns_axes(vqe_and_ed: tuple[object, object, object]) -> None:
    vqe, ed, _ = vqe_and_ed
    fig, ax = plt.subplots()
    plot_convergence(vqe, ed_energy=float(ed.energies[0]), ax=ax)
    assert ax.has_data()
    plt.close(fig)


def test_check_ansatz_expressivity_efficient_su2(
    vqe_and_ed: tuple[object, object, object],
) -> None:
    vqe, ed, _ = vqe_and_ed
    report = check_ansatz_expressivity(
        circuit=vqe.circuit,
        target_state=ed.vectors[:, 0],
        threshold=0.99,
        seed=20260524,
        maxiter=300,
    )
    assert isinstance(report, ExpressivityReport)
    # EfficientSU2 reps=3 on 2 qubits IS expressive enough for the dimer GS.
    assert report.max_overlap > 0.99
    assert report.passed is True


# ---------------------------------------------------------------------------
# Task 6: compare_energies + check_resilience_guardrail
# ---------------------------------------------------------------------------

from matplotlib.figure import Figure  # noqa: E402

from siam_vqe.analysis import (  # noqa: E402
    ResilienceGuardrailReport,
    check_resilience_guardrail,
    compare_energies,
)


def test_compare_energies_returns_figure() -> None:
    results = {"ED": -5.0, "noiseless": -5.0, "FakeMarrakesh": -4.8, "hw_L1": -4.7}
    fig = compare_energies(results, ed_energy=-5.0)
    assert isinstance(fig, Figure)
    # Figure should have one axes and one line representing the ED horizontal ref.
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # ED line is an axhline; check at least one Line2D with horizontal data.
    has_axhline = any(
        len(set(line.get_ydata())) == 1 for line in ax.get_lines()
    )
    assert has_axhline


def test_compare_energies_with_uncertainties() -> None:
    results = {"ED": -5.0, "noiseless": -5.0, "hw_L1": -4.7}
    uncertainties = {"hw_L1": 0.05}
    fig = compare_energies(results, ed_energy=-5.0, uncertainties=uncertainties)
    assert isinstance(fig, Figure)


def test_resilience_guardrail_passes_when_monotone() -> None:
    """|E_hw_L2 - E_ED| <= |E_hw_L1 - E_ED| <= |E_hw_L0 - E_ED| -> passes."""
    rep = check_resilience_guardrail(
        e_ed=-5.0, e_hw_l0=-4.0, e_hw_l1=-4.5, e_hw_l2=-4.8
    )
    assert isinstance(rep, ResilienceGuardrailReport)
    assert rep.passed is True
    assert rep.gap_l0 == pytest.approx(1.0)
    assert rep.gap_l1 == pytest.approx(0.5)
    assert rep.gap_l2 == pytest.approx(0.2)


def test_resilience_guardrail_fails_when_zne_worse_than_m3() -> None:
    """Mitigation level 2 producing a worse result than level 1 -> guardrail fails."""
    rep = check_resilience_guardrail(
        e_ed=-5.0, e_hw_l0=-4.0, e_hw_l1=-4.5, e_hw_l2=-4.3
    )
    assert rep.passed is False
    assert "L2 worse than L1" in rep.notes


def test_check_observable_agreement_multi_accepts_num_particles() -> None:
    """The multi variant must accept num_particles to support L2 (3,3) sector,
    not hardcode (1,1) from the L1 path."""
    from siam_vqe.analysis import check_observable_agreement_multi
    from siam_vqe.hamiltonian import observables_l1, nio_l1_anderson
    from siam_vqe.mappings import to_qubit_op
    from siam_vqe.ansatz import uccsd_ansatz
    from siam_vqe.reference_ed import exact_diag
    from siam_vqe.vqe_runner import run_vqe

    fop = nio_l1_anderson(U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5)
    pop = to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    ed = exact_diag(pop)
    psi_ed = ed.vectors[:, 0]

    ansatz_circuit, x0 = uccsd_ansatz(
        num_spatial_orbitals=2, num_particles=(1, 1), mapper_scheme="parity_tapered"
    )
    vqe = run_vqe(pop, ansatz_circuit, x0, optimizer="SLSQP", maxiter=200, seed=42)

    obs_dict = observables_l1()
    report = check_observable_agreement_multi(
        vqe, pop, psi_ed, obs_dict, num_particles=(1, 1), rel_tol=0.05
    )
    assert report.passed
    assert "n_d_total" in report.values


def test_check_observable_agreement_multi_threads_num_particles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify num_particles is actually passed to to_qubit_op, not silently
    dropped or hardcoded to (1, 1).

    The original physics test passed `num_particles=(1, 1)` which is exactly
    the value the old hardcoded multi path used — so it could not detect a
    silent fallback. This spy-based test patches the to_qubit_op symbol the
    multi function actually calls and asserts the kwarg is threaded through.
    """
    import siam_vqe.analysis as _analysis_module
    from siam_vqe.analysis import check_observable_agreement_multi
    from siam_vqe.ansatz import uccsd_ansatz
    from siam_vqe.hamiltonian import nio_l1_anderson, observables_l1
    from siam_vqe.mappings import to_qubit_op as _real_to_qubit_op
    from siam_vqe.reference_ed import exact_diag
    from siam_vqe.vqe_runner import run_vqe

    captured: list[dict] = []

    def spy(fop, scheme, num_particles=None, **kwargs):
        captured.append({"scheme": scheme, "num_particles": num_particles})
        return _real_to_qubit_op(
            fop, scheme=scheme, num_particles=num_particles, **kwargs
        )

    # Build the L1 (1,1) fixture using the real to_qubit_op (before patching).
    fop = nio_l1_anderson(U=7.3, V=2.06, eps_d=2.5, eps_p=-2.5)
    pop = _real_to_qubit_op(fop, scheme="parity_tapered", num_particles=(1, 1))
    ed = exact_diag(pop)
    psi_ed = ed.vectors[:, 0]
    ansatz_circuit, x0 = uccsd_ansatz(
        num_spatial_orbitals=2, num_particles=(1, 1), mapper_scheme="parity_tapered"
    )
    vqe = run_vqe(pop, ansatz_circuit, x0, optimizer="SLSQP", maxiter=200, seed=42)
    obs_dict = observables_l1()

    # The multi function imports `to_qubit_op as _to_qubit_op` at module top
    # (after the M2 cleanup). Patch that bound name on the analysis module.
    monkeypatch.setattr(_analysis_module, "_to_qubit_op", spy)

    # First call: (1, 1) — should pass physics AND the spy should see (1, 1).
    captured.clear()
    check_observable_agreement_multi(
        vqe, pop, psi_ed, obs_dict, num_particles=(1, 1), rel_tol=0.05
    )
    assert len(captured) >= 1, (
        "to_qubit_op should have been called for FermionicOp observables"
    )
    assert all(c["num_particles"] == (1, 1) for c in captured), (
        f"Expected num_particles=(1,1) in all to_qubit_op calls; got {captured}"
    )

    # Second call: (2, 1) — physics doesn't matter (state won't match obs);
    # we only need the spy to see the new value, proving the parameter
    # actually threads through instead of being silently overridden.
    captured.clear()
    try:
        check_observable_agreement_multi(
            vqe, pop, psi_ed, obs_dict, num_particles=(2, 1), rel_tol=0.05
        )
    except Exception:
        # Mismatched num_particles can fail to_qubit_op (tapering shape mismatch)
        # for a (1,1)-sector pop. That is fine — we only care that the spy was
        # invoked with the new value before any failure.
        pass
    assert len(captured) >= 1, (
        "to_qubit_op should have been called even with mismatched num_particles"
    )
    assert all(c["num_particles"] == (2, 1) for c in captured), (
        f"Expected num_particles=(2,1) in all to_qubit_op calls; got {captured}"
    )
