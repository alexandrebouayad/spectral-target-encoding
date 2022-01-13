from math import sqrt

import numpy as np
from scipy.optimize import minimize
from scipy.stats import betabinom
from sklearn.base import BaseEstimator, TransformerMixin


class HBBM(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        mu_prior=0.5,
        nu_prior=0.0001,
        max_iter=1000,
        tol=0.0001,
        method="spectral",
        solver=None
    ):
        self.mu_prior = mu_prior
        self.nu_prior = nu_prior
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.solver = solver

    @staticmethod
    def _counts(X, y):
        _, counts_total = np.unique(X, return_counts=True)
        n_categories = len(counts_total)

        X_0 = X[y == 0]
        X_1 = X[y == 1]
        categories_0, counts_0 = np.unique(X_0, return_counts=True)
        categories_1, counts_1 = np.unique(X_1, return_counts=True)

        counts = np.zeros((n_categories, 3))
        counts[categories_0, 0] = counts_0
        counts[categories_1, 1] = counts_1
        counts[:, -1] = counts_total

        return counts

    @staticmethod
    def nloglike(rv_params, X, y):
        counts = HBBM._counts(X, y)
        a, b = rv_params

        return -sum(betabinom.logpmf(k, n, a, b) for _, k, n in counts)

    def _spectral_fit(self, X, y=None):
        _, counts_1, counts_total = self._counts(X, y).T.copy()

        mu = self.mu_prior
        nu = self.nu_prior
        a = mu * nu
        mu2p = mu * (a + 1) / (nu + 1)
        n_iter = 0

        while n_iter < self.max_iter:
            mu_post = (a + counts_1) / (nu + counts_total)
            mu2p_post = mu_post * (a + counts_1 + 1) / (nu + counts_total + 1)

            mu_prev, mu = mu, mu_post.mean()
            mu2p_prev, mu2p = mu2p, mu2p_post.mean()
            nu = (mu - mu2p) / (mu2p - mu ** 2)
            a = nu * mu
            n_iter += 1

            delta_mu = mu - mu_prev
            delta_rmu2p = sqrt(mu2p) - sqrt(mu2p_prev)
            distance = sqrt(delta_mu ** 2 + delta_rmu2p ** 2)

            if distance < self.tol:
                break

        self.a_ = a
        self.b_ = nu * (1 - mu)
        self.ps_ = (a + counts_1) / (nu + counts_total)
        self.n_iter_ = n_iter

    def _mle_fit(self, X, y):
        _, counts_1, counts_total = self._counts(X, y).T.copy()

        mu = self.mu_prior
        nu = self.nu_prior
        a = mu * nu
        b = (1 - mu) * nu

        minimum = minimize(
            self.nloglike,
            [a, b],
            args=(X, y),
            method=self.solver,
            bounds=[(0, None), (0, None)],
            options={"maxiter": self.max_iter, "disp": False},
        )

        a, b = minimum.x
        nu = a + b

        self.a_ = a
        self.b_ = b
        self.ps_ = (a + counts_1) / (nu + counts_total)
        self.n_iter_ = minimum.nit

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.method == "spectral":
            self._spectral_fit(X, y)

        elif self.method == "mle":
            self._mle_fit(X, y)

        return self

    def score(self, X, y):
        return -self.nloglike([self.a_, self.b_], X, y)
