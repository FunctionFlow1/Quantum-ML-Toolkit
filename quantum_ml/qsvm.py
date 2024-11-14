from qiskit import QuantumCircuit, Aer, transpile
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import QSVMClassifier
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap
import numpy as np

class QuantumSVM:
    def __init__(self, feature_dimension, seed=42):
        algorithm_globals.random_seed = seed
        self.feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=2)
        self.kernel = QuantumKernel(feature_map=self.feature_map, quantum_instance=QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024, seed_simulator=seed, seed_transpiler=seed))
        self.qsvm = QSVMClassifier(quantum_kernel=self.kernel, optimizer=COBYLA(maxiter=100))

    def train(self, X_train, y_train):
        print("Training Quantum SVM...")
        self.qsvm.fit(X_train, y_train)
        print("Quantum SVM training complete.")

    def predict(self, X_test):
        print("Making predictions with Quantum SVM...")
        return self.qsvm.predict(X_test)

    def score(self, X_test, y_test):
        print("Evaluating Quantum SVM...")
        return self.qsvm.score(X_test, y_test)

if __name__ == "__main__":
    # Generate some dummy data for binary classification
    num_samples = 20
    feature_dim = 2
    X = np.random.rand(num_samples, feature_dim) * 2 - 1 # Features between -1 and 1
    y = (X[:, 0] + X[:, 1] > 0).astype(int) # Simple linear separation

    # Split data
    split_idx = int(0.7 * num_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    qsvm_model = QuantumSVM(feature_dimension=feature_dim)
    qsvm_model.train(X_train, y_train)

    predictions = qsvm_model.predict(X_test)
    accuracy = qsvm_model.score(X_test, y_test)

    print("
True labels:", y_test)
    print("Predictions:", predictions)
    print(f"Accuracy: {accuracy:.4f}")
