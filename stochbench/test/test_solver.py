import numpy as np
from scipy import sparse

from stochbench.solvers import grad_i_linreg, grad_i_linreg_sparse


def test_grad_linreg_sparse():
    np.random.seed(0)
    n_samples, n_features = 100, 100
    A = np.random.randn(n_samples, n_features)
    b = np.random.randn(n_samples)
    x = np.random.randn(n_features)

    A[A < 0.1] = 0
    A_sparse = sparse.csr_matrix(A)
    grad1 = grad_i_linreg(x, A, b, 12)
    grad2 = grad_i_linreg_sparse(x, A_sparse.data, A_sparse.indices,
                                 A_sparse.indptr, b, 12)
    np.testing.assert_allclose(grad1, grad2)
