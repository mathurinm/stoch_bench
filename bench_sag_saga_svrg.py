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
def grad_linreg(x, A, b):
    return A.T @ (A @ x - b) / len(A)


@njit
def loss_linreg(x, A, b):
    return norm(A @ x - b) ** 2 / (2. * len(A))


@njit
def grad_i_logreg(x, A, b, i):
    return - A[i] * b[i] / (1. + np.exp(b[i] * (A[i] @ x)))


@njit
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


np.random.seed(0)
data = "simu"
pb = "logreg"

if data == "simu":
    n_samples, n_features = 1000, 500
    A = np.random.randn(n_samples, n_features)
    b = A @ np.random.randn(n_features) + 0.5 * np.random.randn(n_samples)
else:
    A, b = fetch_libsvm("news20")

if pb == "linreg":
    b -= b.mean()
    b /= np.std(b)
    gamma = 1
elif pb == "logreg":
    b = np.sign(b)
    gamma = 4


n_epochs = 100


step_sgd = 0.01
step_saga = gamma / np.max(norm(A, axis=1) ** 2) / 3.
step_gd = step = gamma / (norm(A, ord=2) ** 2 / n_samples)

m_svrg = 3

algos = ["sgd", "saga", "svrg", "gd", "agd"]
labels = [r"SGD, step=$%.2f / \sqrt{t}$" % step_sgd,
          "SAGA",
          "SVRG, $m=%d$" % m_svrg,
          "GD",
          "AGD"]
params = dict()
params["sgd"] = {'n_epochs': n_epochs, 'step': step_sgd}
params["saga"] = {'n_epochs': n_epochs, 'step': step_saga}
params["gd"] = {'n_epochs': n_epochs, 'step': step_gd}
params["agd"] = {'n_epochs': n_epochs, 'step': step_gd}
params["svrg"] = {'n_epochs': n_epochs // (2 * m_svrg + 1),
                  'step': step_saga,
                  'm': m_svrg}

all_x = dict()
all_loss = dict()
all_calls = dict()


for algo in algos:
    print(algo)
    all_x[algo], all_loss[algo], all_calls[algo] = solver(
        A, b, pb=pb, algo=algo, algo_params=params[algo])


clf = LinearRegression(fit_intercept=False, normalize=False).fit(A, b)
best_obj = norm(b - clf.predict(A)) ** 2 / (2 * len(A))


plt.close('all')
plt.figure()
for algo, label in zip(algos, labels):
    plt.semilogy(all_calls[algo], all_loss[algo] - best_obj, label=label)

plt.title(f"{pb} problem, n={n_samples}, d={n_features}")
plt.ylabel(r"$P(x_k) - P(x^*)$")
plt.xlabel(r'number of calls to $\nabla \psi_i$')
plt.legend()
plt.show(block=False)
