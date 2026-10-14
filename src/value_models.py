"""Model wrappers for predicting 90-day revenue.

LogTargetRegressor: revenue is heavily skewed, so the model learns log(1 + revenue).
    Converting back with exp() predicts the median, not the mean, so totals come out too low.
    Duan's smearing factor rescales predictions so the totals are unbiased.

HurdleValueModel: splits the question in two.
    Part 1: will the customer buy at all? (classifier)
    Part 2: if they buy, how much? (regressor trained only on buyers)
    Expected value = P(buy) x spend-if-buy.
"""
import numpy as np
from sklearn.base import BaseEstimator, clone


class LogTargetRegressor(BaseEstimator):
    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        self.model_ = clone(self.base).fit(X, np.log1p(y))
        residuals = np.log1p(y) - self.model_.predict(X)
        self.smearing_ = float(np.mean(np.exp(residuals)))
        return self

    def predict(self, X):
        return np.clip(np.exp(self.model_.predict(X)) * self.smearing_ - 1, 0, None)


class HurdleValueModel(BaseEstimator):
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y, churned):
        buyers = np.asarray(churned) == 0
        self.classifier_ = clone(self.classifier).fit(X, churned)
        self.regressor_ = LogTargetRegressor(self.regressor).fit(X[buyers], np.asarray(y)[buyers])
        return self

    def predict(self, X):
        p_buy = 1 - self.classifier_.predict_proba(X)[:, 1]
        return p_buy * self.regressor_.predict(X)
