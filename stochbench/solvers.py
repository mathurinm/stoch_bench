import numpy as np
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
    return A.T @ (A @ x - b) / len(A)


def loss_linreg(x, A, b):
    return norm(A @ x - b) ** 2 / (2. * len(A))


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
    return -b * A.T @ (1. / (1. + np.exp(b * (A @ x)))) / len(A)


@njit
def loss_logreg(x, A, b):
    return np.sum(np.log(1 + np.exp(-b * (A @ x))))


def gd(A, b, loss, grad, n_epochs, step):
    x = np.zeros(A.shape[1])
    losses = []
    grad_i_calls = []

    for it in range(n_epochs):
        x -= step * grad(x, A, b)
        losses.append(loss(x, A, b))
        grad_i_calls.append(it * len(A))
    return x, np.array(losses), np.array(grad_i_calls)


def agd(A, b, loss, grad, n_epochs, step):
    x = np.zeros(A.shape[1])
    x_old = x.copy()
    y = x.copy()
    losses = []
    grad_i_calls = []
    t_new = 1

    for it in range(n_epochs):
        x_old[:] = x
        x = y - step * grad(y, A, b)
        t = t_new
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        y = x + (t - 1) / t_new * (x - x_old)
        losses.append(loss(x, A, b))
        grad_i_calls.append(it * len(A))

    return x, np.array(losses), np.array(grad_i_calls)


@njit
def sgd(A, b, loss, grad_i, n_epochs, step):
    np.random.seed(0)
    n_samples, n_features = A.shape
    x = np.zeros(n_features)
    losses = []
    grad_i_calls = []

    for it in range(n_epochs * n_samples):
        i = np.random.choice(n_samples)
        x -= step / np.sqrt(it + 1) * grad_i(x,  A, b, i)

        if it % n_samples == 0:
            losses.append(loss(x, A, b))
            grad_i_calls.append(it)
    return x, np.array(losses), np.array(grad_i_calls)


@njit
def saga(A, b, loss, grad_i, n_epochs, step):
    np.random.seed(0)
    n_samples, n_features = A.shape
    memory_grad = np.zeros((n_samples, n_features))
    x = np.zeros(n_features)
    losses = []
    grad_i_calls = []
    grad_mean = np.zeros(n_features)
    for it in range(n_epochs * n_samples):
        i = np.random.choice(n_samples)
        grad = grad_i(x, A, b, i)
        x -= step * (grad - memory_grad[i] + grad_mean)
        # udpdate table of gradient and its mean:
        grad_mean += (grad - memory_grad[i]) / n_samples
        memory_grad[i] = grad
        if it % n_samples == 0:
            losses.append(loss(x, A, b))
            grad_i_calls.append(it)
    return x, np.array(losses), np.array(grad_i_calls)


@njit
def svrg(A, b, loss, grad_i, grad, n_epochs, step, m):
    """1 full gradient, m * n_samples stochastic gradients:
    1 iteration costs  (2m + 1 * n_samples)"""
    np.random.seed(0)
    n_samples, n_features = A.shape

    x = np.zeros(n_features)
    x_ref = np.zeros(n_features)
    losses = []
    grad_i_calls = []
    for it in range(n_epochs):
        x_ref[:] = x
        grad_full = grad(x_ref, A, b)  # n_samples grad_i calls
        for t in range(m * n_samples):
            i = np.random.choice(n_samples)
            x -= step * (grad_i(x, A, b, i) - grad_i(x_ref, A, b, i)
                         + grad_full)  # 2 grad_i calls
            if t % n_samples == 0:
                losses.append(loss(x, A, b))
                grad_i_calls.append(
                    it * n_samples * (1 + 2 * m) + 2 * t)
    return x, np.array(losses), np.array(grad_i_calls)


def solver(A, b, pb, algo, algo_params):
    if pb == "linreg":
        loss, grad, grad_i = loss_linreg, grad_linreg, grad_i_linreg
    elif pb == "logreg":
        loss, grad, grad_i = loss_logreg, grad_logreg, grad_i_logreg
    else:
        raise ValueError(f"Unsupported problem {pb}")
    if algo == "gd":
        return gd(A, b, loss, grad, **algo_params)
    elif algo == "agd":
        return agd(A, b, loss, grad, **algo_params)
    elif algo == "sgd":
        return sgd(A, b, loss, grad_i, **algo_params)
    elif algo == "svrg":
        return svrg(A, b, loss, grad_i, grad, **algo_params)
    elif algo == "saga":
        return saga(A, b, loss, grad_i, **algo_params)
    else:
        raise ValueError(f"Unsupported algo {algo}")
