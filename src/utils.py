'''
Classical RBF kernel and kernel diagnostics used by the evaluation scripts.
'''

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import sqrtm, inv


def Compute_rbf_kernel(X, X_train):
    """Classical RBF kernel K[i, j] = exp(-gamma ||X[i] - X_train[j]||^2) with the default
    gamma = 1 / (D * Var(X_train)) (scikit-learn's gamma='scale').

    gamma is always computed on the training set (second argument), so the train kernel
    Compute_rbf_kernel(X_train, X_train) and the test kernel Compute_rbf_kernel(X_test, X_train)
    use the same gamma.
    """
    X_train = np.asarray(X_train)
    n_ft = X_train.shape[1]
    gamma = 1 / (n_ft * X_train.var())
    return rbf_kernel(X, X_train, gamma=gamma)


def compute_gd(classic_kernel, quantum_kernel):
    """Geometric difference g = sqrt(|| sqrt(K_c) K_q^-1 sqrt(K_c) ||_2) between a classical and a quantum kernel.
    Diagnostic only, not used in the paper."""
    sqrt_classic = sqrtm(classic_kernel)
    gd_matrix = sqrt_classic.dot(inv(quantum_kernel)).dot(sqrt_classic)
    spectral_norm = np.linalg.norm(gd_matrix, ord=2)
    return np.sqrt(spectral_norm)


def Kernel_concentration(matrix):
    """Variance of the off-diagonal kernel entries (kernel concentration diagnostic)."""
    non_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)].flatten()
    return np.var(non_diagonal)
