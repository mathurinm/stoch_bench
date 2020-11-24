import numpy as np
from numba import njit
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from libsvmdata import fetch_libsvm
# from celer.datasets import make_correlated_data


@njit
def grad_i_linreg(x, A, b, i):
    return A[i] * (A[i] @ x - b[i])


@njit
def grad_linreg(x, A, b):
    return A.T @ (A @ x - b) / len(A)


@njit
def loss_linreg(x, A, b):
    return norm(A @ x - b) ** 2 / (2. * len(A))


@ njit
def grad_i_logreg(x, A, b, i):
    return - A[i] * b[i] / (1. + np.exp(b[i] * A[i] @ x))


@ njit
def grad_logreg(x, A, b):
    return -b * A.T @ (1. / (1. + np.exp(b * (A @ b)))) / len(A)


@ njit
def loss_logreg(x, A, b):
    return np.sum(np.log(1 + np.exp(-b * A @ x)))


def gd(A, b, loss, grad, n_epochs, step):
    x = np.zeros(n_features)
    losses = [norm(b) ** 2 / (2 * len(b))]
    grad_i_calls = [0]

    for it in range(n_epochs):
        x -= step * grad(x, A, b)
        losses.append(loss(x, A, b))
        grad_i_calls.append(it * len(A))
    return x, np.array(losses), np.array(grad_i_calls)


@ njit
def sgd(A, b, loss, grad_i, n_epochs, step):
    np.random.seed(0)
    n_samples, n_features = A.shape
    x = np.zeros(n_features)
    losses = [norm(b) ** 2 / (2 * len(b))]
    grad_i_calls = [0]

    for it in range(n_epochs * n_samples):
        i = np.random.choice(n_samples)
        x -= step / np.sqrt(it + 1) * grad_i(x,  A, b, i)

        if it % n_samples == 0:
            losses.append(loss(x, A, b))
            grad_i_calls.append(it)
    return x, np.array(losses), np.array(grad_i_calls)


@ njit
def saga(A, b, loss, grad_i, n_epochs, step):
    np.random.seed(0)
    n_samples, n_features = A.shape
    memory_grad = np.zeros((n_samples, n_features))
    x = np.zeros(n_features)
    losses = [norm(b) ** 2 / (2 * len(b))]
    grad_i_calls = [0]
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


@ njit
def svrg(A, b, loss, grad_i, grad, m, n_epochs, step):
    """1 full gradient, m * n_samples stochastic gradients:
    1 iteration costs  (2m + 1 * n_samples)"""
    np.random.seed(0)
    n_samples, n_features = A.shape

    x = np.zeros(n_features)
    x_ref = np.zeros(n_features)
    losses = [norm(b) ** 2 / (2 * len(b))]
    grad_i_calls = [0]
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


np.random.seed(0)
n_samples, n_features = 1000, 500
A = np.random.randn(n_samples, n_features)
b = A @ np.random.randn(n_features) + 0.5 * np.random.randn(n_samples)


n_epochs = 100


step_sgd = 0.01

store_every = n_samples
print('sgd')
x_sgd, loss_sgd, calls_sgd = sgd(
    A, b, loss_linreg, grad_i_linreg,
    n_epochs, step=step_sgd)

print('saga')
step_saga = 1 / np.max(norm(A, axis=1) ** 2) / 3.
x_saga, loss_saga, calls_saga = saga(
    A, b, loss_linreg, grad_i_linreg, n_epochs,
    step=step_saga)


m_svrg = 1
print('svrg')
x_svrg, loss_svrg, calls_svrg = svrg(
    A, b, loss_linreg, grad_i_linreg, grad_linreg, m=m_svrg,
    n_epochs=n_epochs // (2 * m_svrg + 1), step=step_saga)


print('gd')
step_gd = step = 1. / (norm(A, ord=2) ** 2 / n_samples)
x_gd, loss_gd, calls_gd = gd(A, b, loss_linreg, grad_linreg,
                             n_epochs, step=step_gd)


clf = LinearRegression(fit_intercept=False, normalize=False).fit(A, b)
best_obj = norm(b - clf.predict(A)) ** 2 / (2 * len(A))

plt.close('all')
plt.figure()
plt.ylabel(r"$P(x_k) - P(x^*)$")
plt.semilogy(calls_saga, loss_saga - best_obj, label="SAGA")
plt.semilogy(calls_sgd, loss_sgd - best_obj, label=f"SGD, step={step_sgd:.3f}")
plt.semilogy(calls_svrg, loss_svrg - best_obj, label=f"SVRG, m={m_svrg}")
plt.semilogy(calls_gd, loss_gd - best_obj, label="GD")
plt.title(f"Linreg problem, n={n_samples}, d={n_features}")

plt.xlabel(r'# calls to $\nabla \psi_i$')
plt.legend()
plt.show(block=False)
