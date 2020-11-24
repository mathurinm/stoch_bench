import numpy as np
from scipy import sparse

from stochbench.solvers import (grad_i_linreg, grad_i_linreg_sparse,
                                grad_i_logreg, grad_i_logreg_sparse)


def test_sparse_gradients():
    np.random.seed(0)
    n_samples, n_features = 100, 100
    A = np.random.randn(n_samples, n_features)
    b = np.random.randn(n_samples)
    x = np.random.randn(n_features)

    A[A < -1] = 0
    A[A > 1] = 0
    A_sparse = sparse.csr_matrix(A)
    grad1 = grad_i_linreg(x, A, b, 12)
    grad2 = grad_i_linreg_sparse(x, A_sparse.data, A_sparse.indices,
                                 A_sparse.indptr, b, 12)
    np.testing.assert_allclose(grad1, grad2)

    b = np.sign(b)
    grad1 = grad_i_logreg(x, A, b, 12)
    grad2 = grad_i_logreg_sparse(x, A_sparse.data, A_sparse.indices,
                                 A_sparse.indptr, b, 12)
    np.testing.assert_allclose(grad1, grad2)
