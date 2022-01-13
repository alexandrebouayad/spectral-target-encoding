from itertools import combinations_with_replacement as combinations_wr
from itertools import product
from pathlib import Path

from category_encoders import TargetEncoder
from category_encoders.glmm import GLMMEncoder
from category_encoders.james_stein import JamesSteinEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import TargetRegressorEncoder

from spcode import HBBM, HBBMEncoder

SEED = 30306210405750927982013772147887179378
VERBOSE = True

RESULT_DIR = Path("cache")
DATA_DIR = Path("data")
RESULT_DIR = Path("results")

ENCODERS = {
    "spectral": {
        "encoder": HBBMEncoder(),
        "cv": None,
    },
    #
    "hbbm_mle": {
        "encoder": HBBMEncoder(method="mle"),
        "cv": None,
    },
    #
    "glmm": {
        "encoder": GLMMEncoder(),
        "cv": None,
    },
    #
    "target": {
        "encoder": TargetEncoder(),
        "cv": None,
    },
    #
    "target_regressor": {
        "encoder": TargetRegressorEncoder(),
        "cv": None,
    },
    #
    "james_stein": {
        "encoder": JamesSteinEncoder(),
        "cv": None,
    },
    #
    "spectral_cv": {
        "encoder": HBBMEncoder(),
        "cv": {
            "grid": {
                "mu_prior": [0.1, 0.5, 0.9],
                "nu_prior": [0.0001, 1.0, 10.0, 100.0],
            },
            "fold": 5,
            "scoring": "roc_auc",
        },
    },
    #
    "target_cv": {
        "encoder": TargetEncoder(),
        "cv": {
            "grid": {
                "smoothing": [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    5.0,
                    10.0,
                    20.0,
                    30.0,
                    50.0,
                    100.0,
                    200.0,
                    300.0,
                ]
            },
            "fold": 5,
            "scoring": "roc_auc",
        },
    },
}

CLASSIFIERS = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "multilayer_perceptron": MLPClassifier(),
}

SCORING = {
    "roc_auc": make_scorer(roc_auc_score),
}

TEST_SIZE = 0.2

HBB_MODELS = {
    "spectral": HBBM(),
    "mle": HBBM(method="mle"),
}

HBB_PARAMS = [
    {"a": a, "b": b, "n_cat": n_cat, "n_obs": n_obs}
    for a, b in combinations_wr([0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 2)
    for n_cat, n_obs in product([[1, 10, 100, 1000]], repeat=2)
]
