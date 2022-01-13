import pickle
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ._frozendict import frozendict
from ._logging import LoggedEstimator, Logger
from .setup import CLASSIFIERS, ENCODERS, GRIDS

NUM_PREPROC = make_pipeline(SimpleImputer(), StandardScaler())


class MLBenchmark:
    def __init__(
        self,
        dataset_file,
        *,
        drop=None,
        num=None,
        cat_na="na",
        target="target",
        predict="1",
        scoring="roc_auc",
        n_splits=10,
        test_size=0.2,
        seed=None,
    ):
        self.drop = drop or []
        self.num = num or []
        self.cat_na = cat_na
        self.target = target
        self.predict = predict
        self.scoring = scoring
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = seed

        usecols = lambda col: col not in self.drop
        df_header = pd.read_csv(dataset_file, usecols=usecols, nrows=0)
        df_columns = df_header.columns
        X_columns = [col for col in df_columns if col != self.target]
        self.cat = [col for col in X_columns if col not in self.num]

        dtypes = {
            **{col: float for col in self.num},
            **{col: str for col in self.cat},
            self.target: str,
        }

        df = pd.read_csv(dataset_file, usecols=df_columns, dtype=dtypes)
        df = df.dropna(subset=[self.target])
        df[self.cat] = df[self.cat].fillna(self.cat_na)

        self.X = df[X_columns]
        self.y = df[self.target].map(lambda y: y == self.predict)
        self.y = self.y.astype(int)

        self._transform = {}
        self._encoder_results = OrderedDict()
        self._pipe_results = OrderedDict()

        self._shuffle_split = ShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self._logger = Logger()

    def encoder_run(self, table):
        table = [dict(row) for row in table if row["encoder"] != "drop"]

        for row in table:
            row.pop("cv_grid", None)
            row.pop("cv_scoring", None)

        table = [frozendict(row) for row in table]
        table = [row for row in table if row not in self._encoder_results]
        table = OrderedDict.fromkeys(table)
        n_rows = len(table)

        if not table:
            return

        for i, row in enumerate(table, start=1):
            encoder_params = {
                key.removeprefix("encoder__"): val
                for key, val in row.items()
                if key.startswith("encoder__")
            }

            encoder_name = row["encoder"]
            encoder = ENCODERS[encoder_name]
            encoder.set_params(**encoder_params)

            logger = self._logger
            logger.count.reset()

            logger.head_msg = "[Encoder " f"{i}" "/" f"{n_rows}" "; "
            logger.head_msg += "{}" "/" f"{self.n_splits}" "]"
            logger.body_msgs = [f"{key}={val}" for key, val in row.items()]

            transformer_lst = [
                ("encoder", encoder, self.cat),
                ("num_preproc", NUM_PREPROC, self.num),
            ]

            transformer = ColumnTransformer(transformer_lst)
            logged_transformer = LoggedEstimator(transformer, logger)

            results = cross_validate(
                estimator=logged_transformer,
                X=self.X,
                y=self.y,
                cv=self._shuffle_split,
                scoring=lambda *_: np.nan,
                verbose=0,
                return_estimator=True,
            )

            fit_transformers = [t.estimator_ for t in results["estimator"]]
            self._transform[row] = [t.transform for t in fit_transformers]

            n_iter = [
                getattr(t.named_transformers_.encoder, "n_iter_", np.nan)
                for t in fit_transformers
            ]

            self._encoder_results[row] = {
                "n_iter": np.array(n_iter),
                "fit_time": results["fit_time"],
            }

    def _drop_run(self, classifier):
        if not self.num:
            return {"fit_time": np.nan}

        logger = self._logger
        logged_classifier = LoggedEstimator(classifier, logger)

        return cross_validate(
            estimator=logged_classifier,
            X=self.X[self.num],
            y=self.y,
            scoring=self.scoring,
            cv=self._shuffle_split,
            verbose=0,
        )

    def _nocv_run(self, encoder_row, classifier):
        encoder_row = dict(encoder_row)
        encoder_row.pop("cv_grid", None)
        encoder_row.pop("cv_scoring", None)
        encoder_row = frozendict(encoder_row)

        logger = self._logger
        logged_classifier = LoggedEstimator(classifier, logger)

        split_iterable = self._shuffle_split.split(self.X)
        transform_splits = zip(self._transform[encoder_row], split_iterable)

        results = defaultdict(list)

        for transform, split in transform_splits:
            split_results = cross_validate(
                estimator=logged_classifier,
                X=transform(self.X),
                y=self.y,
                scoring=self.scoring,
                cv=[split],
                verbose=0,
            )

            for key, val in split_results.items():
                results[key].extend(val)

        results = {key: np.array(vals) for key, vals in results.items()}
        results["fit_time"] += self._encoder_results[encoder_row]["fit_time"]

        return results

    def _cv_run(self, encoder_row, classifier):
        cv_grid_name = encoder_row["cv_grid"]
        cv_grid = GRIDS[cv_grid_name]
        cv_scoring = encoder_row["cv_scoring"]

        encoder_params = {
            key.removeprefix("encoder__"): val
            for key, val in encoder_row.items()
            if key.startswith("encoder__")
        }

        encoder_name = encoder_row["encoder"]
        encoder = ENCODERS[encoder_name]
        encoder.set_params(**encoder_params)

        transformer_lst = [
            ("encoder", encoder, self.cat),
            ("num_preproc", NUM_PREPROC, self.num),
        ]

        transformer = ColumnTransformer(transformer_lst)
        steps = [("transformer", transformer), ("classifier", classifier)]
        pipe = Pipeline(steps)

        param_grid = {
            f"transformer__{key}": vals for key, vals in cv_grid.items()
        }

        grid_search_cv = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=self.scoring,
            refit=cv_scoring,
            verbose=0,
        )

        logger = self._logger
        logged_grid_search_cv = LoggedEstimator(grid_search_cv, logger)

        return cross_validate(
            estimator=logged_grid_search_cv,
            X=self.X,
            y=self.y,
            scoring=self.scoring,
            cv=self._shuffle_split,
            verbose=0,
        )

    def run(self, encoder_table, classifier_table):
        self.encoder_run(encoder_table)

        encoder_table = [dict(row) for row in encoder_table]

        for row in encoder_table:
            row["cv_grid"] = row.get("cv_grid")
            row["cv_scoring"] = row.get("cv_scoring")

        table = [
            (frozendict(encoder_row), frozendict(classifier_row))
            for encoder_row in encoder_table
            for classifier_row in classifier_table
        ]

        table = [row for row in table if row not in self._pipe_results]
        table = OrderedDict.fromkeys(table)
        n_rows = len(table)

        if not table:
            return

        for i, row in enumerate(table, start=1):
            encoder_row, classifier_row = row

            logger = self._logger
            logger.count.reset()

            logger.head_msg = "[Pipe " f"{i}" "/" f"{n_rows}" "; "
            logger.head_msg += "{}" "/" f"{self.n_splits}" "]"

            enc_msgs = [f"{key}={val}" for key, val in encoder_row.items()]
            clf_msgs = [f"{key}={val}" for key, val in classifier_row.items()]
            logger.body_msgs = enc_msgs + clf_msgs

            cv_grid_name = encoder_row["cv_grid"]
            encoder_name = encoder_row["encoder"]
            classifier_name = classifier_row["classifier"]

            classifier_params = {
                key.removeprefix("classifier__"): val
                for key, val in classifier_row.items()
                if key.startswith("classifier__")
            }

            classifier_name = classifier_row["classifier"]
            classifier = CLASSIFIERS[classifier_name]
            classifier.set_params(**classifier_params)

            if encoder_name == "drop":
                results = self._drop_run(classifier)

            elif cv_grid_name:
                results = self._cv_run(encoder_row, classifier)

            else:
                results = self._nocv_run(encoder_row, classifier)

            self._pipe_results[row] = results

    def encoder_score(self, scoring):
        keys = iter(self._encoder_results)
        keys = pd.DataFrame(keys, dtype=object)
        keys = keys.sort_index(axis=1)

        scores = [scoring(res) for res in self._encoder_results.values()]
        scores = pd.DataFrame(scores, dtype=object)
        scores = pd.concat([keys, scores], axis=1)

        return scores

    def pipe_score(self, scoring):
        encoder_keys = [row for row, _ in self._pipe_results]
        encoder_keys = pd.DataFrame(encoder_keys, dtype=object)
        encoder_keys = encoder_keys.sort_index(axis=1)

        classifier_keys = [row for _, row in self._pipe_results]
        classifier_keys = pd.DataFrame(classifier_keys, dtype=object)
        classifier_keys = classifier_keys.sort_index(axis=1)

        scores = [scoring(res) for res in self._pipe_results.values()]
        scores = pd.DataFrame(scores, dtype=object)
        scores = pd.concat([encoder_keys, classifier_keys, scores], axis=1)

        return scores

    def load(self, backup_file):
        with open(backup_file, "rb") as fp:
            backup = pickle.load(fp)

        self._transform = backup["transform"]
        self._encoder_results = backup["encoder_results"]
        self._pipe_results = backup["pipe_results"]

    def dump(self, backup_file):
        backup = {
            "transform": self._transform,
            "encoder_results": self._encoder_results,
            "pipe_results": self._pipe_results,
        }

        with open(backup_file, "wb") as fp:
            pickle.dump(backup, fp)
