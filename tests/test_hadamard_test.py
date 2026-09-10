"""Production Hadamard-test wrapper — Re + Im, simulator, same-sector."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def test_hadamard_re_basic():
    """⟨1|c†|0⟩ = ⟨1|(X−iY)/2|0⟩ = 1.0 + 0j on 1 qubit."""
    from siam_vqe.noise import compute_hadamard_test

    qc_left = QuantumCircuit(1)
    qc_left.x(0)
    qc_right = QuantumCircuit(1)
    op = SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)])

    re, im, err_re, err_im = compute_hadamard_test(
        qc_left=qc_left, qc_right=qc_right, operator=op, shots=8192,
    )
    assert abs(re - 1.0) < 3 * err_re + 0.05
    assert abs(im) < 3 * err_im + 0.05


def test_hadamard_im_basic():
    """⟨+|iX|+⟩ = i; Re=0, Im=1 on 1 qubit."""
    from siam_vqe.noise import compute_hadamard_test

    qc = QuantumCircuit(1)
    qc.h(0)
    op = SparsePauliOp.from_list([("X", 1j)])

    re, im, err_re, err_im = compute_hadamard_test(
        qc_left=qc, qc_right=qc, operator=op, shots=8192,
    )
    assert abs(re) < 3 * err_re + 0.05
    assert abs(im - 1.0) < 3 * err_im + 0.05


def test_hadamard_vs_statevector_4qubit():
    """Random 4-qubit ψ_left, ψ_right, O: Hadamard test ≈ statevector ⟨ψ_left|O|ψ_right⟩."""
    from siam_vqe.noise import compute_hadamard_test

    rng = np.random.default_rng(42)
    qc_left = QuantumCircuit(4)
    for q in range(4):
        qc_left.ry(rng.uniform(0, np.pi), q)
    for q in range(3):
        qc_left.cx(q, q + 1)

    qc_right = QuantumCircuit(4)
    for q in range(4):
        qc_right.rx(rng.uniform(0, np.pi), q)
    for q in range(3):
        qc_right.cx(q, q + 1)

    paulis = ["IIIX", "IIXI", "IXII", "XIII", "IIIY", "YIII", "IZIZ"]
    coeffs = (rng.standard_normal(len(paulis)) + 1j * rng.standard_normal(len(paulis))) * 0.3
    op = SparsePauliOp.from_list(list(zip(paulis, coeffs, strict=True)))

    sv_left = Statevector(qc_left).data
    sv_right = Statevector(qc_right).data
    expected = sv_left.conj() @ op.to_matrix() @ sv_right

    re, im, err_re, err_im = compute_hadamard_test(
        qc_left=qc_left, qc_right=qc_right, operator=op, shots=16384,
    )
    assert abs(re - expected.real) < 5 * err_re + 0.05
    assert abs(im - expected.imag) < 5 * err_im + 0.05


def test_hadamard_cross_sector_4qubit_jw():
    """Cross-sector matrix element ⟨n=3|c†_0|n=2⟩ via Hadamard test
    matches Slater rule on a 4-qubit JW system."""
    from siam_vqe.noise import compute_hadamard_test

    # ψ_right = |1100⟩ (n=2: qubits 2,3 occupied; qubits 0,1 unoccupied)
    qc_right = QuantumCircuit(4)
    qc_right.x(2)
    qc_right.x(3)

    # ψ_left = |1101⟩ (n=3: add particle on qubit 0)
    qc_left = QuantumCircuit(4)
    qc_left.x(0)
    qc_left.x(2)
    qc_left.x(3)

    # c†_0 in JW = (X_0 − i Y_0) / 2  (qubits 1,2,3 idle → identity)
    op = SparsePauliOp.from_list([("IIIX", 0.5), ("IIIY", -0.5j)])

    re, im, err_re, err_im = compute_hadamard_test(
        qc_left=qc_left, qc_right=qc_right, operator=op, shots=8192,
    )
    # ⟨1101|c†_0|1100⟩ = 1 (Slater rule: c† on unoccupied site gives +1)
    assert abs(re - 1.0) < 3 * err_re + 0.05
    assert abs(im) < 3 * err_im + 0.05


def test_hadamard_antihermitian_warning():
    """An anti-Hermitian operator triggers the Observation #20 warning."""
    import warnings
    from siam_vqe.noise import compute_hadamard_test

    qc = QuantumCircuit(2)
    qc.h(0)
    # T = c†_0 c_1 − c†_1 c_0 (anti-Hermitian; JW: (XY − YX)/2 + (something Z))
    # Simpler: just iY (anti-Hermitian and nonzero)
    op = SparsePauliOp.from_list([("IY", 1j)])
    op_mat = op.to_matrix()
    assert np.allclose(op_mat, -op_mat.conj().T)  # confirm anti-Hermitian

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_hadamard_test(qc_left=qc, qc_right=qc, operator=op, shots=1024)
        assert any("Observation #20" in str(rec.message) for rec in w)


def test_compute_hadamard_test_isa_backend_path() -> None:
    """The isa_backend branch (full ISA transpile, translator) runs locally on
    a FakeMarrakesh target and returns finite Re/Im. Guards the hardware path."""
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from siam_vqe.noise import compute_hadamard_test

    fb = FakeMarrakesh()
    qc_left = QuantumCircuit(1)
    qc_left.x(0)
    qc_right = QuantumCircuit(1)  # |0>
    op = SparsePauliOp.from_list([("Z", 1.0)])
    re, im, e_re, e_im = compute_hadamard_test(
        qc_left=qc_left, qc_right=qc_right, operator=op,
        shots=512, mode=fb, isa_backend=fb,
    )
    assert np.isfinite(re) and np.isfinite(im)
    assert e_re >= 0.0 and e_im >= 0.0


def test_hadamard_with_mitigated_estimator_fakemarrakesh():
    """Hadamard test under M3+ZNE on FakeMarrakesh recovers the noiseless
    matrix element within mitigation tolerance."""
    pytest.importorskip("qiskit_ibm_runtime")
    pytest.importorskip("mthree")
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    from siam_vqe.mitigation import MitigationSpec
    from siam_vqe.noise import (
        compute_hadamard_test_mitigated,
        make_mitigated_estimator,
    )

    # AerSimulator.from_backend(FakeMarrakesh) is the existing pattern from
    # tests/test_noise.py — raw FakeMarrakesh.run() is not used by
    # MitigatedEstimator's sampler path. The noise model still comes from the
    # FakeMarrakesh snapshot.
    fake = FakeMarrakesh()
    sim = AerSimulator.from_backend(fake)
    spec = MitigationSpec(
        name="m3_zne_poly3_123",
        m3=True, zne=True,
        zne_extrapolator="polynomial_degree_3",
        zne_noise_factors=(1.0, 2.0, 3.0),
        shots=4096,
    )
    mit_est = make_mitigated_estimator(sim, spec)

    # Same 1-qubit ⟨1|c†|0⟩ = 1 test as Task 2a
    qc_left = QuantumCircuit(1)
    qc_left.x(0)
    qc_right = QuantumCircuit(1)
    op = SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)])

    re, im, _err_re, _err_im = compute_hadamard_test_mitigated(
        qc_left=qc_left, qc_right=qc_right, operator=op,
        shots=4096, mitigated_estimator=mit_est,
    )
    # FakeMarrakesh has nontrivial noise; M3+ZNE should recover within 0.15
    assert abs(re - 1.0) < 0.15
    assert abs(im) < 0.15
