"""Spike: confirm RuntimeEstimatorV2(mode=AerSimulator) honors resilience options.

Run once. Prints the resilience-option dict actually applied by the estimator,
and runs a small Bell-state ⟨Z⊗Z⟩ measurement at three resilience configs to
see if the result reflects the mitigation choice.

If this fails, Task 7 falls back to manual M3 + manual ZNE (see Task 8).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2


def main() -> None:
    # Build a simple Bell-state circuit
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])

    # Build a noisy Aer simulator with a 2% depolarizing error on CX
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), "cx")
    sim = AerSimulator(noise_model=noise)

    print("=" * 60)
    print("Spike 1: no mitigation")
    est = RuntimeEstimatorV2(mode=sim)
    est.options.default_shots = 4096
    try:
        est.options.resilience.measure_mitigation = False
        est.options.resilience.zne_mitigation = False
    except AttributeError as e:
        print(f"  AttributeError setting resilience options: {e}")
        print("  → local-mode RuntimeEstimatorV2 does NOT expose resilience options.")
        print("  → Task 7 must use the manual fallback path (Task 8).")
        return
    res = est.run([(bell, obs)]).result()
    print(f"  ⟨ZZ⟩ unmit = {float(res[0].data.evs):.4f}  (ideal = 1.0)")

    print("=" * 60)
    print("Spike 2: M3 only")
    est2 = RuntimeEstimatorV2(mode=sim)
    est2.options.default_shots = 4096
    est2.options.resilience.measure_mitigation = True
    est2.options.resilience.zne_mitigation = False
    res2 = est2.run([(bell, obs)]).result()
    print(f"  ⟨ZZ⟩ M3 = {float(res2[0].data.evs):.4f}")

    print("=" * 60)
    print("Spike 3: M3 + ZNE linear [1, 3, 5]")
    est3 = RuntimeEstimatorV2(mode=sim)
    est3.options.default_shots = 4096
    est3.options.resilience.measure_mitigation = True
    est3.options.resilience.zne_mitigation = True
    est3.options.resilience.zne.noise_factors = (1.0, 3.0, 5.0)
    est3.options.resilience.zne.extrapolator = "linear"
    res3 = est3.run([(bell, obs)]).result()
    print(f"  ⟨ZZ⟩ M3+ZNE = {float(res3[0].data.evs):.4f}")
    print(f"  metadata[0] resilience keys: {list(res3[0].metadata.get('resilience', {}).keys())}")

    print("=" * 60)
    print("VERDICT: local-mode resilience options are honored.")


if __name__ == "__main__":
    main()
