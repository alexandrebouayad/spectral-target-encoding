import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

with open("datasets.yml", "r") as fp:
    DATASETS = yaml.safe_load(fp)

from config import CLASSIFIERS, DATASETS, ENCODERS, VERBOSE

ml_pipeline = Pipeline(
    [("transformer", None), ("classifier", None)],
    memory=CACHE_DIR,
)


def ml_task(*, encoder, classifier, dataset, entropy):
    random_state = entropy
    encoder, cv_params = ENCODERS[encoder]
    encoder = clone(encoder)
    for param_name in encoder.get_params():
        if param_name.endswith("random_state"):
            encoder.set_params({param_name: random_state})
    classifier = CLASSIFIERS[classifier]
    classifier = clone(classifier)
    for param_name in classifier.get_params():
        if param_name.endswith("random_state"):
            classifier.set_params({param_name: random_state})
    sampler, categorical, numerical = DATASETS[dataset]
    train, test = sampler(random_state)

    num_preproc = make_pipeline(SimpleImputer(), StandardScaler())
    enc_transformer = ("categorical", encoder, categorical)
    num_transformer = ("numerical", num_preproc, numerical)
    transformer = ColumnTransformer([enc_transformer, num_transformer])
    ml_pipeline.set_params(
        transformer=transformer,
        classifier=classifier,
        verbose=VERBOSE,
    )
    if cv_params is None:
        estimator = ml_pipeline
    else:
        param_grid = {
            f"transformer__encoder__{key}": values
            for key, values in cv_params["param_grid"].items()
        }
        estimator = GridSearchCV(
            estimator=ml_pipeline,
            param_grid=param_grid,
            scoring=cv_params["scoring"],
            cv=cv_params["n_folds"],
            refit=True,
        )
    start_time = time()
    estimator.fit(*train)
    fit_time = time() - start_time
    scoring = {
        name: make_scorer(metric) for name, metric in ML_METRICS.items()
    }
    scores = {name: score(estimator, *test) for name, score in scoring}
    if classifier is None:
        scores["fit_time"] = fit_time
    return scores


class MLTask:
    def init(self, encoder, classifier, dataset, *, memory=None, verbose=0):
        self.encoder = encoder
        self.classifier = classifier
        self.dataset = dataset
        self.memory = memory
        self.verbose = verbose
        self._pipe = Pipeline([("transformer", None), ("classifier", None)])

    def setup(self, seed):
        encoder, cv_params = ENCODERS[self.encoder]
        encoder = clone(encoder)
        for param_name in encoder.get_params():
            if param_name.endswith("random_state"):
                encoder.set_params({param_name: seed})
        classifier = CLASSIFIERS[self.classifier]
        classifier = clone(classifier)
        for param_name in classifier.get_params():
            if param_name.endswith("random_state"):
                classifier.set_params({param_name: seed})
        sampler, categorical, numerical = DATASETS[self.dastaset]
        self._train, self._test = self._sample(sampler, seed)

        num_preproc = make_pipeline(SimpleImputer(), StandardScaler())
        enc_transformer = ("categorical", encoder, categorical)
        num_transformer = ("numerical", num_preproc, numerical)
        transformer = ColumnTransformer([enc_transformer, num_transformer])
        self._pipe.set_params(
            transformer=transformer,
            classifier=classifier,
            memory=self.memory,
            verbose=self.verbose,
        )
        if cv_params is None:
            self._estimator = self._pipe
        else:
            param_grid = {
                f"transformer__encoder__{key}": values
                for key, values in cv_params["param_grid"].items()
            }
            self._estimator = GridSearchCV(
                estimator=self._pipe,
                param_grid=param_grid,
                scoring=cv_params["scoring"],
                cv=cv_params["n_folds"],
                refit=True,
            )

    def run(self):
        self._estimator.fit(*self._train)
        self.estimator_ = clone(self._estimator)


class RealDatasetSampler:
    def __init__(
        self,
        csvfile,
        *,
        drop=None,
        numerical=None,
        fill="na",
        target="target",
        predict="1",
        test_size=0.2,
        random_state=None,
    ):
        drop = drop or []
        numerical = numerical or []
        usecols = lambda col: col not in drop
        header = pd.read_csv(csvfile, usecols=usecols, nrows=0)
        columns = header.columns
        X_columns = [col for col in columns if col != target]
        categorical = [col for col in X_columns if col not in numerical]

        dtype = {
            **{col: float for col in numerical},
            **{col: str for col in categorical},
            target: str,
        }
        df = pd.read_csv(csvfile, usecols=columns, dtype=dtype)
        df = df.dropna(subset=[target])
        df[categorical] = df[categorical].fillna(fill)
        X = df[X_columns]
        y = df[target].map(lambda y: y == predict)
        y = y.astype(int)

        self.categorical = [df.columns.get_loc(name) for name in categorical]
        self.numerical = [df.columns.get_loc(name) for name in numerical]
        self.X = np.array(X)
        self.y = np.array(y)
        self.test_size = test_size
        self.random_state = random_state

    def sample(self):
        return train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )


class ToyDatasetSampler:
    def __init__(self):
        pass


# class NoneEstimator(BaseEstimator):
#     def fit(self, X, y):
#         return self


# class DropTransformer(NoneEstimator, TransformerMixin):
#     def transform(self, X):
#         n_features = X.shape[0]
#         return np.empty((n_features, 0))


# class MLBenchmark:
#     def __init__(
#         self,
#         encoders,
#         classifiers,
#         *,
#         csvfile,
#         drop=None,
#         numerical=None,
#         fill="na",
#         target="target",
#         predict="1",
#         test_size=0.2,
#         scoring="roc_auc",
#         seed=None,
#         verbose=0,
#     ):
#         self._init_df(
#             csvfile=csvfile,
#             drop=drop,
#             numerical=numerical,
#             fill=fill,
#             target=target,
#             predict=predict,
#         )
#         self._test_size = test_size
#         self._scoring = (
#             {"roc_auc": make_scorer(roc_auc_score)}
#             if scoring == "roc_auc"
#             else scoring
#         )
#         self._init_mappings(encoders, classifiers)
#         self._init_random(seed)
#         self.verbose = verbose

#     def _init_df(
#         self,
#         *,
#         csvfile,
#         drop,
#         numerical,
#         fill,
#         target,
#         predict,
#     ):
#         drop = drop or []
#         numerical = numerical or []
#         usecols = lambda col: col not in drop
#         header = pd.read_csv(csvfile, usecols=usecols, nrows=0)
#         columns = header.columns
#         X_columns = [col for col in columns if col != target]
#         categorical = [col for col in X_columns if col not in numerical]

#         dtype = {
#             **{col: float for col in numerical},
#             **{col: str for col in categorical},
#             target: str,
#         }
#         df = pd.read_csv(csvfile, usecols=columns, dtype=dtype)
#         df = df.dropna(subset=[target])
#         df[categorical] = df[categorical].fillna(fill)
#         X = df[X_columns]
#         y = df[target].map(lambda y: y == predict)
#         y = y.astype(int)

#         self._categorical = [df.columns.get_loc(name) for name in categorical]
#         self._numerical = [df.columns.get_loc(name) for name in numerical]
#         self._X = np.array(X)
#         self._y = np.array(y)

#     def _init_mappings(self, encoders, classifiers):
#         self._encoders = {
#             enc_name: (clone(enc_spec["object"]), enc_spec["cv"])
#             for enc_name, enc_spec in encoders.items()
#         }
#         if self._numerical:
#             self._encoders["drop"] = (DropTransformer(), None)

#         self._classifiers = {
#             clf_name: clone(clf_obj)
#             for clf_name, clf_obj in classifiers.items()
#         }
#         self._classifiers["none"] = NoneEstimator()

#         self._transforms = defaultdict(list)
#         self._results = defaultdict(list)

#     def _init_random(self, seed):
#         seq = SeedSequence(seed)
#         bg = PCG64DXSM(seq)
#         rns = RandomState(bg)
#         self._rns = rns

#         random_encoders = [
#             encoder
#             for encoder, _ in self._encoders.values()
#             if "random_state" in encoder.get_params()
#         ]
#         random_classifiers = [
#             classifier
#             for classifier in self._classifiers.values()
#             if "random_state" in classifier.get_params()
#         ]
#         for object in random_encoders + random_classifiers:
#             object.set_params(random_state=rns)

#     def _sample(self):
#         shuffle_split = ShuffleSplit(
#             n_splits=1,
#             test_size=self._test_size,
#             random_state=self._rns,
#         )
#         [(train, test)] = shuffle_split.split(self._X)
#         self._X_train = self._X[train].copy()
#         self._X_test = self._X[test].copy()
#         self._y_train = self._y[train].copy()
#         self._y_test = self._y[test].copy()

#     def _run(self, res_key, run_idx):
#         enc_key, clf_key = res_key
#         encoder, cv_params = self._encoders[enc_key]
#         classifier = self._classifiers[clf_key]
#         encoding = clf_key == "none"
#         cross_validating = bool(cv_params)

#         num_preproc = make_pipeline(SimpleImputer(), StandardScaler())
#         enc_trf = ("enc", encoder, self._categorical)
#         num_trf = ("num", num_preproc, self._numerical)
#         trf = ColumnTransformer([enc_trf, num_trf])
#         steps = [("trf", trf), ("clf", classifier)]
#         estimator = Pipeline(steps)

#         if cross_validating:
#             param_grid = {
#                 f"trf__enc__{key}": value
#                 for key, value in cv_params["grid"].items()
#             }
#             estimator = GridSearchCV(
#                 estimator=estimator,
#                 param_grid=param_grid,
#                 scoring=cv_params["scoring"],
#                 cv=cv_params["fold"],
#                 refit=True,
#             )

#         elif not encoding:
#             transform = self._transforms[enc_key][run_idx]
#             cached_encoder = FunctionTransformer(transform)
#             estimator.set_params(trf__enc=cached_encoder)

#         with warnings.catch_warnings():
#             if not self.verbose:
#                 warnings.simplefilter("ignore")
#             start_time = time()
#             estimator.fit(self._X_train, self._y_train)
#             runtime = time() - start_time

#         fitted_pipe = (
#             estimator if not cross_validating else estimator.best_estimator_
#         )
#         fitted_encoder = fitted_pipe.named_steps.trf.named_transformers_.enc

#         encoder_n_iter = np.mean(getattr(fitted_encoder, "n_iter_", np.nan))
#         fit_time = (
#             runtime
#             if encoding or cross_validating
#             else runtime + self._results[enc_key, "none"][run_idx]["fit_time"]
#         )
#         encoder_params = (
#             {
#                 key.removeprefix("trf__enc__"): val
#                 for key, val in estimator.best_params_.items()
#             }
#             if cross_validating
#             else np.nan
#         )
#         results = {
#             key: scorer(estimator, self._X_test, self._y_test)
#             if not encoding
#             else np.nan
#             for key, scorer in self._scoring.items()
#         }
#         results |= {
#             "encoder_n_iter": encoder_n_iter,
#             "fit_time": fit_time,
#             "encoder_params": encoder_params,
#         }
#         results = {key: [val] for key, val in results.items()}

#         if encoding:
#             self._transforms[enc_key] += [fitted_encoder.transform]
#         dtype = {key: float for key in results}
#         dtype["encoder_params"] = dict
#         dtype = list(dtype.items())
#         self._results[res_key] = structured_append(
#             self._results[res_key], results, dtype=dtype
#         )
#         return runtime

#     def run(self, repeat=1, *, cache_file=None, export_file=None):
#         original_sigint_handler = signal.getsignal(signal.SIGINT)
#         signal.signal(signal.SIGINT, sigint_handler)

#         key_pairs = [
#             (enc_key, "none")
#             for enc_key, (_, cv_params) in self._encoders.items()
#             if not cv_params
#         ]
#         key_pairs += [
#             (enc_key, clf_key)
#             for clf_key, enc_key in product(self._classifiers, self._encoders)
#             if clf_key != "none"
#         ]
#         for run_idx in range(repeat):
#             if any(run_idx >= len(self._results[key]) for key in key_pairs):
#                 self._sample()

#             for res_idx, res_key in enumerate(key_pairs):
#                 enc_key, clf_key = res_key
#                 head_msg = "run " f"{run_idx + 1}" "/" f"{repeat}" "; "
#                 head_msg += "fit " f"{res_idx + 1}" "/" f"{len(key_pairs)}"
#                 body_msgs = ["encoder: " + enc_key, "classifier: " + clf_key]

#                 if run_idx < len(self._results[res_key]):
#                     self._log("CACHED", head_msg, body_msgs)
#                     continue

#                 self._log("START", head_msg, body_msgs)
#                 runtime = self._run(res_key, run_idx)
#                 if cache_file:
#                     self.dump(cache_file)
#                 if export_file:
#                     self.to_csv(export_file)
#                 runtime_msgs = human_format(runtime)
#                 self._log("END", head_msg, runtime_msgs, sep=", ")

#         signal.signal(signal.SIGINT, original_sigint_handler)

#     @property
#     def results(self):
#         return {
#             res_key: array
#             for res_key, array in self._results.items()
#             if res_key != ("drop", "none") and len(array) > 0
#         }

#     def to_csv(self, csvfile):
#         data = []
#         for res_key, array in self.results.items():
#             enc_key, clf_key = res_key
#             record = {
#                 "encoder": enc_key,
#                 "classifier": clf_key,
#                 "n_runs": len(array),
#             }
#             for name in array.dtype.names:
#                 record |= {
#                     f"{name}_{i}": value
#                     for i, value in enumerate(array[name], start=1)
#                 }
#                 if not np.issubdtype(array.dtype[name], np.number):
#                     continue
#                 series = pd.Series(array[name])
#                 percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
#                 stats = series.describe(percentiles)
#                 stats = stats.drop("count")
#                 stats = stats.add_prefix(name + "_")
#                 record |= stats.to_dict()
#             data.append(record)

#         df = pd.DataFrame(data)
#         df.to_csv(csvfile, index=False)

#     def dump(self, file):
#         with open(file, "wb") as fp:
#             pickle.dump(self, fp)

#     @classmethod
#     def load(cls, backup_file):
#         with open(backup_file, "rb") as fp:
#             return pickle.load(fp)
