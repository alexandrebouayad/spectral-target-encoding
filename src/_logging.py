import warnings
from time import time

import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.metaestimators import if_delegate_has_method


def human_readable(delta):
    delta_units = [
        "years",
        "months",
        "days",
        "hours",
        "minutes",
        "seconds",
        "microseconds",
    ]

    return [
        "{} {}".format(
            int(getattr(delta, unit)),
            unit if getattr(delta, unit) > 1 else unit[:-1],
        )
        for unit in delta_units
        if getattr(delta, unit)
    ]


class LogCounter:
    def __init__(self, _count=None):
        if _count is None:
            self._count = 0

        elif isinstance(_count, LogCounter):
            if isinstance(_count._count, LogCounter):
                self._count = LogCounter(_count._count)

            else:
                self._count = _count

        else:
            raise TypeError

    def __deepcopy__(self, memo=None):
        return LogCounter(self)

    def increment(self):
        if isinstance(self._count, int):
            self._count += 1

        else:
            self._count.increment()

    def get_value(self):
        if isinstance(self._count, int):
            return self._count

        return self._count.get_value()

    def reset(self):
        if isinstance(self._count, int):
            self._count = 0

        else:
            self._count.reset()


class Logger:
    def __init__(self):
        self.count = LogCounter()
        self.head_msg = ""
        self.body_msgs = []


class LoggedEstimator(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, logger):
        self.estimator = estimator
        self.logger = logger

    def fit(self, X, y=None):
        self.logger.count.increment()

        count = self.logger.count.get_value()
        head_msg = self.logger.head_msg.format(count)
        body_msgs = self.logger.body_msgs

        print(head_msg, "START", sep="  ", end="  ")
        print(*body_msgs, sep=", ")

        self.estimator_ = clone(self.estimator)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_fit_time = time()
            self.estimator_.fit(X, y)
            end_fit_time = time()
            fit_time = end_fit_time - start_fit_time

        self.fit_time_ = fit_time

        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_

        delta = relativedelta(seconds=fit_time)
        delta = human_readable(delta)

        print(head_msg, "END  ", "fit_time=", sep="  ", end=" ")
        print(*delta, sep=", ")

        return self

    @if_delegate_has_method("estimator")
    def transform(self, X):
        return self.estimator_.transform(X)

    @if_delegate_has_method("estimator")
    def predict(self, X):
        return self.estimator_.predict(X)

    @if_delegate_has_method("estimator")
    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    @if_delegate_has_method("estimator")
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @if_delegate_has_method("estimator")
    def score(self, X=None, y=None):
        return self.estimator_.score(X, y)
