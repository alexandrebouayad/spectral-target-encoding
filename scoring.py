from ast import Or
from collections import OrderedDict

import numpy as np


def stat_scoring(results, sample_row):
    a = sample_row["a"]
    b = sample_row["b"]

    nu = a + b
    mu = a / nu
    mu2p = mu * (a + 1) / (nu + 1)

    results = dict(results)
    results.pop("ps_")
    a_ = results.pop("a_")
    b_ = results.pop("b_")

    nu_ = a_ + b_
    mu_ = a_ / nu_
    mu2p_ = mu_ * (a_ + 1) / (nu_ + 1)

    params = OrderedDict(a=a, b=b, mu=mu, mu2p=mu2p, nu=nu)
    params_ = OrderedDict(a=a_, b=b_, mu=mu_, mu2p=mu2p_, nu=nu_)

    errors = [(key, np.abs(params_[key] - params[key])) for key in params]
    errors = OrderedDict(errors)
    errors["a-b"] = np.sqrt(errors["a"] ** 2 + errors["b"] ** 2)
    errors["mu-mu2p"] = np.sqrt(errors["mu"] ** 2 + errors["mu2p"])

    params_ = [(f"{key}_", vals) for key, vals in params_.items()]
    errors = [(f"error_{key}", vals) for key, vals in errors.items()]
    results = OrderedDict(params_) | OrderedDict(errors) | results

    scores = OrderedDict()

    for key, array in results.items():
        scores[f"mean_{key}"] = array.mean()
        scores[f"std_{key}"] = array.std()

    return scores


def encoder_scoring(results):
    return OrderedDict(
        mean_n_iter=results["n_iter"].mean(),
        std_n_iter=results["n_iter"].std(),
        mean_fit_time=results["fit_time"].mean(),
        std_fit_time=results["fit_time"].std(),
    )


def pipe_scoring(results):
    score_items = [
        (key.removeprefix("test_"), values)
        for key, values in results.items()
        if key.startswith("test_")
    ]

    score_items += [("fit_time", results["fit_time"])]

    scores = OrderedDict()

    for key, values in score_items:
        scores["mean_" + key] = np.mean(values)
        scores["std_" + key] = np.std(values)

    return scores
