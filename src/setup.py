from category_encoders import TargetEncoder
from category_encoders.glmm import GLMMEncoder
from category_encoders.james_stein import JamesSteinEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import TargetRegressorEncoder

from .hbbm import HBBM, HBBMEncoder

MODELS = {
    "HBBM": HBBM(),
}

ENCODERS = {
    "drop": "drop",
    "SpectralTargetEncoder": HBBMEncoder(),
    "MLE-HBBMEncoder": HBBMEncoder(method="mle"),
    "GLMMEncoder": GLMMEncoder(),
    "JamesSteinEncoder": JamesSteinEncoder(),
    "TargetEncoder": TargetEncoder(),
    "TargetRegressorEncoder": TargetRegressorEncoder(),
}

CLASSIFIERS = {
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
}

ste_grid = {
    "encoder__mu_prior": [0.1, 0.5, 0.9],
    "encoder__nu_prior": [0.0, 1.0, 10.0, 100.0],
}

smoothing_grid = {
    "encoder__smoothing": [
        0.0,
        5.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
    ]
}

GRIDS = {
    "STEGrid": ste_grid,
    "SmoothingGrid": smoothing_grid,
}
