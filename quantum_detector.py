import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import OneClassSVM

from metrics_calculator import calculate_metrics

IBM_TOKEN = ""


def angle_encoding_feature_map(num_qubits):
    params = ParameterVector('x', num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    return qc


def get_sampler(shots=1024):
    if IBM_TOKEN:
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=IBM_TOKEN, overwrite=True)
        service = QiskitRuntimeService()

        backends = service.backends(operational=True, simulator=False, min_num_qubits=8)
        if not backends:
            raise Exception("No IBM backends available")

        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=8)
        print(f"\nUsing IBM Quantum backend: {backend.name}")
        print(f"Queue depth: {backend.status().pending_jobs}")

        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = shots
    else:
        print(f"\nUsing local Quantum backend simulator")
        sampler = StatevectorSampler(default_shots=shots)

    return sampler


def run_quantum(x_train, x_test, y_train, y_test, subset=None):
    print("\n" + "=" * 50)
    print("Quantum: Fidelity Kernel")
    print("=" * 50 + "\n")

    x_train_normal = x_train[y_train == 1][:subset]
    x_test_subset = x_test[:subset]
    y_test_subset = y_test[:subset]

    print(f"Training samples (normal): {len(x_train_normal)}")
    print(
        f"Test samples: {len(x_test_subset)} (normal: {np.sum(y_test_subset == 1)}, anomaly: {np.sum(y_test_subset == 0)})")

    feature_map = angle_encoding_feature_map(8)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    quantum_kernel._sampler = get_sampler()

    print("\nComputing training kernel...")
    k_train = quantum_kernel.evaluate(x_train_normal)

    print("\nComputing test kernel...")
    k_test = quantum_kernel.evaluate(x_test_subset, x_train_normal)

    print("\nTraining model...")
    model = OneClassSVM(kernel='precomputed', nu=0.3)
    model.fit(k_train)

    print("\nTesting...")
    predictions = model.predict(k_test)

    return calculate_metrics(predictions, y_test_subset)
