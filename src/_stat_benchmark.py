import pickle
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import beta, binom

from ._frozendict import frozendict
from ._logging import LoggedEstimator, Logger
from .setup import MODELS


class StatBenchmark:
    def __init__(self, *, n_samplings=10, seed=None):
        self.n_samplings = n_samplings
        self.seed = seed

        self._rng = None
        self._results = OrderedDict()
        self._logger = Logger()

    def _run(self, model, *, a, b, n_categories, n_observations):
        beta_rvs = beta.rvs(a, b, size=n_categories, random_state=self._rng)

        binom_rvs = [
            binom.rvs(1, p, size=n_observations, random_state=self._rng)
            for p in beta_rvs
        ]

        X = [[i] * n_observations for i in range(n_categories)]
        X = np.concatenate(X)
        X = X.reshape((-1, 1))
        y = np.concatenate(binom_rvs)

        logger = self._logger
        logged_model = LoggedEstimator(model, logger)
        logged_model.fit(X, y)

        a_ = logged_model.estimator_.a_
        b_ = logged_model.estimator_.b_
        ps_ = logged_model.estimator_.ps_

        nloglike = -logged_model.score(X, y)
        fit_time = logged_model.fit_time_
        n_iter = logged_model.estimator_.n_iter_

        return {
            "a_": a_,
            "b_": b_,
            "ps_": ps_,
            "nloglike": nloglike,
            "n_iter": n_iter,
            "fit_time": fit_time,
        }

    def run(self, model_table, sample_table):
        table = [
            (frozendict(model_row), frozendict(sample_row))
            for model_row in model_table
            for sample_row in sample_table
        ]

        table = [row for row in table if row not in self._results]
        table = OrderedDict.fromkeys(table)
        n_rows = len(table)

        if not table:
            return

        for i, row in enumerate(table, start=1):
            model_row, sample_row = row

            logger = self._logger
            logger.count.reset()

            logger.head_msg = "[Fit " f"{i}" "/" f"{n_rows}" "; "
            logger.head_msg += "{}" "/" f"{self.n_samplings}" "]"

            model_msgs = [f"{key}={val}" for key, val in model_row.items()]
            sample_msgs = [f"{key}={val}" for key, val in sample_row.items()]
            logger.body_msgs = model_msgs + sample_msgs

            model_params = {
                key.removeprefix("model__"): val
                for key, val in model_row.items()
                if key.startswith("model__")
            }

            model_name = model_row["model"]
            model = MODELS[model_name]
            model.set_params(**model_params)

            self._rng = np.random.default_rng(self.seed)
            results = defaultdict(list)

            for _ in range(self.n_samplings):
                sample_results = self._run(model, **sample_row)

                for key, val in sample_results.items():
                    results[key].append(val)

            results = {key: np.array(vals) for key, vals in results.items()}
            self._results[row] = results

    def score(self, scoring):
        model_keys = [model_row for model_row, _ in self._results]
        model_keys = pd.DataFrame(model_keys, dtype=object)
        model_keys = model_keys.sort_index(axis=1)

        sample_keys = [sample_row for _, sample_row in self._results]
        sample_keys = pd.DataFrame(sample_keys, dtype=object)
        sample_keys = sample_keys.sort_index(axis=1)

        scores = [
            scoring(res, sample_row)
            for (_, sample_row), res in self._results.items()
        ]

        scores = pd.DataFrame(scores, dtype=object)
        scores = pd.concat([model_keys, sample_keys, scores], axis=1)

        return scores

    def load(self, backup_file):
        with open(backup_file, "rb") as fp:
            self._results = pickle.load(fp)

    def dump(self, backup_file):
        with open(backup_file, "wb") as fp:
            pickle.dump(self._results, fp)
