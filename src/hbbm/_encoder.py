import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OrdinalEncoder

from ._model import HBBM


class HBBMEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        mu_prior=0.5,
        nu_prior=0.0001,
        categories="auto",
        handle_unknown=True,
        max_iter=1000,
        tol=0.0001,
        method="spectral",
        solver=None,
    ):
        self.mu_prior = mu_prior
        self.nu_prior = nu_prior
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.solver = solver

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        self._ordinal_encoder = OrdinalEncoder(
            categories=self.categories,
            dtype=int,
        )

        if self.handle_unknown:
            self._ordinal_encoder.set_params(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )

        hbbm = HBBM(
            mu_prior=self.mu_prior,
            nu_prior=self.nu_prior,
            max_iter=self.max_iter,
            tol=self.tol,
            method=self.method,
            solver=self.solver,
        )

        X_ordinal = self._ordinal_encoder.fit_transform(X)
        mapping = []
        n_iter = []

        for X_i in X_ordinal.T:
            hbbm_i = clone(hbbm)
            hbbm_i.fit(X_i, y)
            mu_i = hbbm_i.a_ / (hbbm_i.a_ + hbbm_i.b_)
            mapping_i = np.append(hbbm_i.ps_, mu_i)

            mapping.append(mapping_i)
            n_iter.append(hbbm_i.n_iter_)

        self.mapping_ = mapping
        self.n_iter_ = np.array(n_iter)

        return self

    def transform(self, X):
        X_ordinal = self._ordinal_encoder.transform(X)

        X_encoded = [
            mapping_i[X_i]
            for mapping_i, X_i in zip(self.mapping_, X_ordinal.T)
        ]

        return np.array(X_encoded).T.copy()
