import numpy as np
import matplotlib
from scipy import sparse
from numpy.linalg import norm
import matplotlib.pyplot as plt
from libsvmdata import fetch_libsvm

from stochbench import solver

np.random.seed(0)
# data = "simu"
data = "news20"
pb = "logreg"

if data == "simu":
    n_samples, n_features = 1000, 500
    A = np.random.randn(n_samples, n_features)
    b = A @ np.random.randn(n_features) + 0.5 * np.random.randn(n_samples)
else:
    A, b = fetch_libsvm(data)
    A = A.tocsr()
    A = A[:1000]


if pb == "linreg":
    b -= b.mean()
    b /= np.std(b)
    gamma = 1
elif pb == "logreg":
    b = np.sign(b)
    gamma = 4


n_epochs = 100


step_sgd = 0.01
if sparse.issparse(A):
    step_saga = gamma / np.max(sparse.linalg.norm(A, axis=1) ** 2) / 3.
    step_gd = gamma / (sparse.linalg.svds(A, k=1)[1][0] ** 2 / len(b))

else:
    step_saga = gamma / np.max(norm(A, axis=1) ** 2) / 3.
    step_gd = gamma / (norm(A, ord=2) ** 2 / len(b))

m_svrg = 3

# algos = ["sgd", "saga", "svrg", "gd", "agd"]
algos = ["sgd"]
labels = [r"SGD, step=$%.2f / \sqrt{t}$" % step_sgd,
          "SAGA",
          "SVRG, $m=%d$" % m_svrg,
          "GD",
          "AGD"]
params = dict()
params["sgd"] = {'n_epochs': n_epochs, 'step': step_sgd, 'verbose': True}
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
