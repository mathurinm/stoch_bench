import numpy as np
from scipy import sparse
from numba import njit
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from libsvmdata import fetch_libsvm
# from celer.datasets import make_correlated_data

matplotlib.rcParams["text.usetex"] = True


@njit
def grad_i_linreg(x, A, b, i):
    return A[i] * (A[i] @ x - b[i])


@njit
def grad_i_linreg_sparse(x, A_data, A_indices, A_indptr, b, i):
    grad = np.zeros_like(x)
    Ai_x = 0.
    startptr, endptr = A_indptr[i], A_indptr[i + 1]
    for idx in range(startptr, endptr):
        Ai_x += A_data[idx] * x[A_indices[idx]]
    grad[A_indices[startptr:endptr]] = A_data[startptr:endptr] * (Ai_x - b[i])
    return grad


def grad_linreg(x, A, b):
    return A.T @ (A @ x - b) / A.shape[0]


def loss_linreg(x, A, b):
    return norm(A @ x - b) ** 2 / (2. * A.shape[0])


@njit
def grad_i_logreg(x, A, b, i):
    return - A[i] * b[i] / (1. + np.exp(b[i] * (A[i] @ x)))


@njit
def grad_i_logreg_sparse(x, A_data, A_indices, A_indptr, b, i):
    grad = np.zeros_like(x)
    Ai_x = 0.
    startptr, endptr = A_indptr[i], A_indptr[i + 1]
    for idx in range(startptr, endptr):
        Ai_x += A_data[idx] * x[A_indices[idx]]
    scalar = - b[i] / (1. + np.exp(b[i] * Ai_x))
    grad[A_indices[startptr:endptr]] = A_data[startptr:endptr] * scalar
    return grad


def grad_logreg(x, A, b):
    return -b * A.T @ (1. / (1. + np.exp(b * (A @ x)))) / A.shape[0]


def loss_logreg(x, A, b):
    return np.sum(np.log(1 + np.exp(-b * (A @ x))))


def gd(A, b, loss, grad, n_epochs, step, verbose=False):
    x = np.zeros(A.shape[1])
    losses = [loss(x, A, b)]
    grad_i_calls = [0]

    for epoch in range(n_epochs):
        x -= step * grad(x, A, b)
        losses.append(loss(x, A, b))
        grad_i_calls.append((epoch + 1) * A.shape[0])
        if verbose:
            print(f"Epoch {epoch + 1}, loss: {losses[-1]}")
    return x, np.array(losses), np.array(grad_i_calls)


def agd(A, b, loss, grad, n_epochs, step, verbose=False):
    x = np.zeros(A.shape[1])
    x_old = x.copy()
    y = x.copy()
    losses = [loss(x, A, b)]
    grad_i_calls = [0]
    t_new = 1

    for epoch in range(n_epochs):
        x_old[:] = x
        x = y - step * grad(y, A, b)
        t = t_new
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        y = x + (t - 1) / t_new * (x - x_old)
        losses.append(loss(x, A, b))
        grad_i_calls.append((epoch + 1) * A.shape[0])
        if verbose:
            print(f"Epoch {epoch + 1}, loss: {losses[-1]}")

    return x, np.array(losses), np.array(grad_i_calls)


@njit
def _sgd_epoch(A, b, x, grad_i, samples, steps):
    for i in samples:
        x -= steps[i] * grad_i(x, A, b, i)


@njit
def _sgd_epoch_sparse(A_data, A_indices, A_indptr, b, x, grad_i_sparse,
                      samples, steps):
    for i in samples:
        x -= steps[i] * grad_i_sparse(x, A_data, A_indices, A_indptr, b, i)


def sgd(A, b, loss, grad_i, grad_i_sparse, n_epochs, step, verbose=False):
    np.random.seed(0)
    n_samples, n_features = A.shape
    x = np.zeros(n_features)
    losses = [loss(x, A, b)]
    grad_i_calls = [0]

    for epoch in range(n_epochs):
        samples = np.random.choice(n_samples, n_samples)
        steps = step / np.sqrt(epoch * n_samples + np.arange(1, n_samples + 1))
        if sparse.issparse(A):
            _sgd_epoch_sparse(A.data, A.indices, A.indptr, b, x, grad_i_sparse,
                              samples, steps)
        else:
            _sgd_epoch(A, b, x, grad_i, samples, steps)

        losses.append(loss(x, A, b))
        grad_i_calls.append((epoch + 1) * n_samples)
        if verbose:
            print(f"Epoch {epoch + 1}, loss: {losses[-1]}")
    return x, np.array(losses), np.array(grad_i_calls)


@njit
def _saga_epoch(A, b, x, grad_i, samples, step, memory_grad, grad_mean):
    for i in samples:
        grad = grad_i(x, A, b, i)
        x -= step * (grad - memory_grad[i] + grad_mean)
        grad_mean += (grad - memory_grad[i]) / len(b)
        memory_grad[i] = grad


@njit
def _saga_epoch_sparse(
        A_data, A_indices, A_indptr, b, x, grad_i_sparse, samples, step,
        memory_grad, grad_mean):
    for i in samples:
        grad = grad_i_sparse(x, A_data, A_indices, A_indptr, b, i)
        x -= step * (grad - memory_grad[i] + grad_mean)
        grad_mean += (grad - memory_grad[i]) / len(b)
        memory_grad[i] = grad


def saga(A, b, loss, grad_i, grad_i_sparse, n_epochs, step, verbose=False):
    np.random.seed(0)
    n_samples, n_features = A.shape
    memory_grad = np.zeros((n_samples, n_features))
    x = np.zeros(n_features)
    losses = [loss(x, A, b)]
    grad_i_calls = [0]
    grad_mean = np.zeros(n_features)
    for epoch in range(n_epochs):
        samples = np.random.choice(n_samples, n_samples)
        if sparse.issparse(A):
            _saga_epoch_sparse(A.data, A.indices, A.indptr, b, x,
                               grad_i_sparse, samples, step,
                               memory_grad, grad_mean)
        else:
            _saga_epoch(A, b, x, grad_i, samples, step, memory_grad, grad_mean)

        losses.append(loss(x, A, b))
        grad_i_calls.append((epoch + 1) * n_samples)
        if verbose:
            print(f"Epoch {epoch + 1}, loss: {losses[-1]}")
    return x, np.array(losses), np.array(grad_i_calls)


@njit
def _svrg_epoch(A, b, x, grad_i, samples, step, x_ref, grad_full):
    for i in samples:
        x -= step * (grad_i(x, A, b, i) - grad_i(x_ref, A, b, i)
                     + grad_full)  # 2 grad_i calls


@njit
def _svrg_epoch_sparse(A_data, A_indices, A_indptr, b, x, grad_i_sparse,
                       samples, step, x_ref, grad_full):
    for i in samples:
        x -= step * (grad_i_sparse(x, A_data, A_indices, A_indptr, b, i)
                     - grad_i_sparse(x_ref, A_data, A_indices, A_indptr, b, i)
                     + grad_full)  # 2 grad_i calls


def svrg(A, b, loss, grad_i, grad_i_sparse, grad, n_epochs, step, m,
         verbose=False):
    """1 full gradient, m * n_samples stochastic gradients:
    1 iteration costs  (2m + 1) * n_samples)"""
    np.random.seed(0)
    n_samples, n_features = A.shape

    x = np.zeros(n_features)
    x_ref = np.zeros(n_features)
    losses = [loss(x, A, b)]
    grad_i_calls = [0]
    for epoch in range(n_epochs):
        x_ref[:] = x
        grad_full = grad(x_ref, A, b)  # n_samples grad_i calls
        samples = np.random.choice(n_samples, m * n_samples)
        if sparse.issparse(A):
            _svrg_epoch_sparse(A.data, A.indices, A.indptr, b, x,
                               grad_i_sparse, samples, step, x_ref, grad_full)
        else:
            _svrg_epoch(A, b, x, grad_i, samples, step, x_ref, grad_full)

        losses.append(loss(x, A, b))
        grad_i_calls.append(n_samples * (epoch + 1) * (1 + 2 * m))
        if verbose:
            print(f"Epoch {epoch + 1}, loss: {losses[-1]}")
    return x, np.array(losses), np.array(grad_i_calls)


def solver(A, b, pb, algo, algo_params):
    if pb == "linreg":
        loss, grad = loss_linreg, grad_linreg,
        grad_i, grad_i_sparse = grad_i_linreg, grad_i_linreg_sparse
    elif pb == "logreg":
        loss, grad = loss_logreg, grad_logreg
        grad_i, grad_i_sparse = grad_i_logreg, grad_i_logreg_sparse
    else:
        raise ValueError(f"Unsupported problem {pb}")

    if algo == "gd":
        return gd(A, b, loss, grad, **algo_params)
    elif algo == "agd":
        return agd(A, b, loss, grad, **algo_params)
    elif algo == "sgd":
        return sgd(A, b, loss, grad_i, grad_i_sparse, **algo_params)
    elif algo == "svrg":
        return svrg(A, b, loss, grad_i, grad_i_sparse, grad, **algo_params)
    elif algo == "saga":
        return saga(A, b, loss, grad_i, grad_i_sparse, **algo_params)
    else:
        raise ValueError(f"Unsupported algo {algo}")
