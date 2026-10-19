'''
Fidelity quantum kernel with the IQP feature map (Section II and III of the paper).

The kernel between two inputs x1, x2 is k(x1, x2) = |<0| U(x2)^dagger U(x1) |0>|^2, where U is the IQP
embedding (qml.IQPEmbedding) with a linear or full entanglement pattern and `n_repeats` repetitions.
The bandwidth b is not applied here: the input data are rescaled by b (times pi) before calling the kernel,
see the scripts in experiments/QK_computation/.
'''

import numpy as np
import jax

import pennylane as qml


class QuantumKernel:
    """IQP feature-map fidelity kernel simulated with PennyLane (default.qubit, JAX interface).

    Parameters:
        n_qubits: number of qubits, equal to the number of features D.
        ent_type: entanglement pattern of the ZZ terms, 'linear' (nearest neighbours) or 'full' (all pairs).
        n_repeats: number of repetitions r of the feature map.
    """

    def __init__(self, n_qubits=7, ent_type='linear', n_repeats=1):
        self.n_qubits = n_qubits
        self.ent_type = ent_type
        self.n_repeats = n_repeats

        # Simulator and projector on |0...0>: the kernel is the expectation value of this projector
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.projector = np.zeros((2**n_qubits, 2**n_qubits))
        self.projector[0, 0] = 1

        # Precompile the quantum circuit
        self.circuit = self._create_circuit()

    def feature_map(self, x):
        """IQP embedding of x with the chosen entanglement pattern."""
        if self.ent_type == 'linear':
            patterns = [[i, i + 1] for i in range(self.n_qubits - 1)]
        elif self.ent_type == 'full':
            patterns = [[i, j] for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]
        else:
            raise ValueError('Entanglement type not recognized')

        qml.IQPEmbedding(x, range(self.n_qubits), self.n_repeats, pattern=patterns)

    def _create_circuit(self):
        """Circuit U(x2)^dagger U(x1) |0>, measured on the |0...0> projector, JIT-compiled with JAX."""
        @jax.jit
        @qml.qnode(self.dev, interface="jax")
        def circuit(x1, x2):
            self.feature_map(x1)
            qml.adjoint(self.feature_map)(x2)
            return qml.expval(qml.Hermitian(self.projector, wires=range(self.n_qubits)))
        return circuit

    def kernel(self, x1, x2):
        """Quantum kernel between two inputs."""
        return self.circuit(x1, x2)

    def kernel_matrix(self, A, B):
        """Kernel matrix K[i, j] = k(A[i], B[j]), shape (len(A), len(B)), vectorised with jax.vmap."""
        vectorized_kernel = jax.vmap(
            jax.vmap(self.kernel, in_axes=(None, 0)),
            in_axes=(0, None)
        )
        return vectorized_kernel(A, B)
