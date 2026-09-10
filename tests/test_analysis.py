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
    plot_adapt_convergence,
    plot_convergence,
    plot_l3_observables,
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


# ---------------------------------------------------------------------------
# Tasks 22-23: L3 layer checkers (Layer 2, 4, 5, 6)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from siam_vqe.analysis import (  # noqa: E402
    check_layer2_overlap_l3,
    check_layer5_observables_l3,
)
from siam_vqe.hamiltonian_l3 import L3Params  # noqa: E402
from siam_vqe.reference_l3 import compute_l3_reference  # noqa: E402


def test_layer2_overlap_l3_passes_on_ed_ground_state():
    """If the VQE statevector is the exact ED ground state, overlap = 1 >= 0.85."""
    p = L3Params()
    ref = compute_l3_reference(p, k_states=2)
    # Build a fake VQE result whose tapered statevector lifts back to ref.ground_vector
    from siam_vqe.tapering_l3 import project_full_to_tapered

    psi_full = np.zeros(2**20, dtype=complex)
    for i, occ in enumerate(ref.basis):
        psi_full[occ] = ref.ground_vector[i]
    psi_tapered = project_full_to_tapered(psi_full, num_particles=(9, 9))

    result = check_layer2_overlap_l3(psi_tapered, ref, threshold=0.85)
    assert result["pass"] is True
    assert result["overlap"] == pytest.approx(1.0, abs=1e-6)


def test_layer2_overlap_l3_softens_on_near_degeneracy():
    """If E_1 - E_0 < 10 meV, soften: overlap measured against projector onto
    the near-degenerate subspace."""
    from siam_vqe.tapering_l3 import project_full_to_tapered

    p = L3Params()
    ref = compute_l3_reference(p, k_states=2)
    # Simulate near-degeneracy by patching excited_energies (test-only override).
    from dataclasses import replace
    fake_ref = replace(ref, excited_energies=np.array([ref.ground_energy + 0.005]))

    # Use a tapered vector that has support in the (9,9) sector so the lift
    # is non-zero and the gap check (rather than leak check) governs the mode.
    psi_full = np.zeros(2**20, dtype=complex)
    for i, occ in enumerate(ref.basis):
        psi_full[occ] = ref.ground_vector[i]
    psi_tapered = project_full_to_tapered(psi_full, num_particles=(9, 9))
    result = check_layer2_overlap_l3(psi_tapered, fake_ref, threshold=0.85)
    assert result["mode"] == "soft_degeneracy"


def test_layer5_observables_l3_passes_when_within_5pct():
    """<n_d>, <S^2>, <n_p> all within 5% relative -> layer passes."""
    p = L3Params()
    ref = compute_l3_reference(p, k_states=2)
    # Fake VQE observables: exact match (zero relative error).
    vqe_obs = dict(ref.observables)
    result = check_layer5_observables_l3(vqe_obs, ref, rel_tol=0.05, abs_tol=0.05)
    assert result["pass"] is True


def test_layer5_observables_l3_abs_tol_fallback_when_ed_obs_near_zero():
    """When |<O>_ED| < 0.01, switch to absolute tolerance < 0.05."""
    ref_obs = {"n_d": 8.0, "n_p": 10.0, "S2": 2.0,
               "n_d_3z2": 1.5, "n_d_x2y2": 1.4, "n_d_xz": 1.7,
               "n_d_yz": 1.7, "n_d_xy": 1.7}
    from siam_vqe.reference_l3 import L3Reference
    ref = L3Reference(
        ground_energy=-1.0, ground_vector=np.zeros(100, dtype=complex),
        excited_energies=np.array([0.0]), sector_dim=100,
        sector=(9, 9), basis=tuple(range(100)),
        observables=ref_obs,
    )
    # All exact except <n_d_x2y2> where ED is near zero. (Substitute small value.)
    ref_obs_zero = dict(ref_obs)
    ref_obs_zero["n_d_x2y2"] = 0.005
    ref_zero = L3Reference(
        ground_energy=ref.ground_energy,
        ground_vector=ref.ground_vector,
        excited_energies=ref.excited_energies,
        sector_dim=ref.sector_dim, sector=ref.sector, basis=ref.basis,
        observables=ref_obs_zero,
    )
    vqe_obs = dict(ref_obs_zero)
    vqe_obs["n_d_x2y2"] = 0.03  # abs err 0.025 < 0.05
    result = check_layer5_observables_l3(vqe_obs, ref_zero, rel_tol=0.05, abs_tol=0.05)
    assert result["pass"] is True
    assert result["details"]["n_d_x2y2"]["mode"] == "abs_tol"


from siam_vqe.adapt_vqe import AdaptResult  # noqa: E402
from siam_vqe.analysis import (  # noqa: E402
    check_layer4_multistart_l3,
    check_layer6_adapt_trace,
)
from siam_vqe.vqe_runner import AdaptMultistartResult  # noqa: E402


def _fake_adapt_result(E):
    return AdaptResult(
        final_energy=E, theta=np.array([0.1]),
        operators_picked=(0,),
        trace=({"iteration": 0, "energy": E, "g_max": 0.0, "k_star": 0,
                "n_operators": 1, "event": "operator_added"},),
        converged_reason="gradient",
    )


def test_layer4_multistart_l3_passes_when_spread_under_100mev():
    per_seed = tuple(_fake_adapt_result(E) for E in [-1.5, -1.49, -1.48, -1.47])
    multi = AdaptMultistartResult(
        per_seed=per_seed, best_seed_index=0,
        best_energy=-1.5, median_energy=-1.485, worst_energy=-1.47,
        spread=0.03,
    )
    result = check_layer4_multistart_l3(multi, layer1_passed=True,
                                         spread_threshold=0.1)
    assert result["pass"] is True
    assert result["mode"] == "strict"


def test_layer4_multistart_l3_soft_warn_when_spread_fails_but_best_passes():
    """L2 soft-gate carryover: spread > threshold but best seed still passes."""
    per_seed = tuple(_fake_adapt_result(E) for E in [-1.5, -1.0, -0.9, -0.8])
    multi = AdaptMultistartResult(
        per_seed=per_seed, best_seed_index=0,
        best_energy=-1.5, median_energy=-0.95, worst_energy=-0.8,
        spread=0.7,
    )
    result = check_layer4_multistart_l3(multi, layer1_passed=True,
                                         spread_threshold=0.1)
    assert result["pass"] is True
    assert result["mode"] == "soft_cluster"
    assert "warn" in result


def test_layer4_multistart_l3_fails_when_layer1_also_fails():
    per_seed = tuple(_fake_adapt_result(E) for E in [-1.5, -1.0, -0.9, -0.8])
    multi = AdaptMultistartResult(
        per_seed=per_seed, best_seed_index=0,
        best_energy=-1.5, median_energy=-0.95, worst_energy=-0.8,
        spread=0.7,
    )
    result = check_layer4_multistart_l3(multi, layer1_passed=False,
                                         spread_threshold=0.1)
    assert result["pass"] is False


def test_layer6_adapt_trace_monotonic_passes_on_clean_descent():
    trace = (
        {"iteration": 0, "energy": -1.0, "g_max": 0.5, "n_operators": 1,
         "event": "operator_added"},
        {"iteration": 1, "energy": -1.2, "g_max": 0.3, "n_operators": 2,
         "event": "operator_added"},
        {"iteration": 2, "energy": -1.5, "g_max": 0.1, "n_operators": 3,
         "event": "operator_added"},
        {"iteration": 3, "energy": -1.5, "g_max": 0.001, "n_operators": 3,
         "event": "converged"},
    )
    result = check_layer6_adapt_trace(trace, noise_floor_eV=0.001)
    assert result["pass"] is True


def test_layer6_adapt_trace_fails_on_runaway_growth():
    """Energy goes UP after operator append -> unphysical (bug indicator)."""
    trace = (
        {"iteration": 0, "energy": -1.0, "g_max": 0.5, "n_operators": 1,
         "event": "operator_added"},
        {"iteration": 1, "energy": -0.7, "g_max": 0.3, "n_operators": 2,
         "event": "operator_added"},
    )
    result = check_layer6_adapt_trace(trace, noise_floor_eV=0.001)
    assert result["pass"] is False
    assert "violations" in result


def test_plot_adapt_convergence_creates_pdf(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    trace = (
        {"iteration": 0, "energy": -1.0, "g_max": 0.5, "n_operators": 1,
         "event": "operator_added"},
        {"iteration": 1, "energy": -1.2, "g_max": 0.3, "n_operators": 2,
         "event": "operator_added"},
        {"iteration": 2, "energy": -1.4, "g_max": 0.1, "n_operators": 3,
         "event": "operator_added"},
        {"iteration": 3, "energy": -1.45, "g_max": 0.001, "n_operators": 3,
         "event": "converged"},
    )
    output = tmp_path / "adapt_convergence.pdf"
    ax = plot_adapt_convergence(trace, ed_reference=-1.46, output_path=output)
    import matplotlib.axes
    assert isinstance(ax, matplotlib.axes.Axes)
    assert output.exists()


def test_plot_l3_observables_creates_pdf(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    vqe_obs = {"n_d": 7.8, "n_p": 10.2, "S2": 1.95,
               "n_d_3z2": 1.5, "n_d_x2y2": 1.4, "n_d_xz": 1.7,
               "n_d_yz": 1.6, "n_d_xy": 1.6}
    ed_obs = {"n_d": 7.85, "n_p": 10.15, "S2": 1.99,
              "n_d_3z2": 1.52, "n_d_x2y2": 1.42, "n_d_xz": 1.69,
              "n_d_yz": 1.62, "n_d_xy": 1.60}
    output = tmp_path / "obs.pdf"
    ax = plot_l3_observables(vqe_obs, ed_obs, output_path=output)
    import matplotlib.axes
    assert isinstance(ax, matplotlib.axes.Axes)
    assert output.exists()


def test_check_layer_xas_peak_energies_passes_when_all_match():
    """Each Phase-5 peak within 50 meV of an EDRIXS peak → pass."""
    from siam_vqe.analysis import check_layer_xas_peak_energies

    phase5_peaks = np.array([1.0, 3.5, 6.0])
    edrixs_peaks = np.array([1.02, 3.49, 6.01])
    report = check_layer_xas_peak_energies(phase5_peaks, edrixs_peaks,
                                            tolerance_eV=0.05)
    assert report["pass"] is True
    assert report["max_residual_eV"] < 0.05


def test_check_layer_xas_peak_energies_fails_with_missing_peak():
    """A peak in EDRIXS with no Phase-5 counterpart fails the gate."""
    from siam_vqe.analysis import check_layer_xas_peak_energies

    phase5_peaks = np.array([1.0])
    edrixs_peaks = np.array([1.0, 3.5])  # extra peak in EDRIXS
    report = check_layer_xas_peak_energies(phase5_peaks, edrixs_peaks,
                                            tolerance_eV=0.05)
    assert report["pass"] is False


def test_check_layer_xas_spectral_weight_l2_distance():
    from siam_vqe.analysis import check_layer_xas_spectral_weight

    omega = np.linspace(0, 10, 101)
    sigma_phase5 = np.exp(-((omega - 5)**2) / 2)
    sigma_edrixs = sigma_phase5 * 1.02  # 2% bias
    report = check_layer_xas_spectral_weight(sigma_phase5, sigma_edrixs,
                                              tolerance_frac=0.05)
    assert report["pass"] is True
    assert 0.01 < report["l2_distance_frac"] < 0.03


def test_check_layer_xas_sum_rule_within_one_percent():
    from siam_vqe.analysis import check_layer_xas_sum_rule

    sum_weights = 2.001
    expected = 2.000
    report = check_layer_xas_sum_rule(sum_weights, expected,
                                       tolerance_frac=0.01)
    assert report["pass"] is True


def test_plot_xas_spectrum_writes_file(tmp_path):
    from siam_vqe.analysis import plot_xas_spectrum
    from siam_vqe.xas import XASSpectrum

    omega = np.linspace(0, 10, 101)
    spec = XASSpectrum(
        omega_eV=omega,
        sigma=np.exp(-((omega - 5)**2)),
        channel="lin_z",
        Gamma_eV=0.5,
        peak_energies=np.array([5.0]),
        peak_weights=np.array([1.0]),
    )
    out = tmp_path / "xas.pdf"
    plot_xas_spectrum(spec, edrixs_spectrum=None, output_path=str(out))
    assert out.exists()
