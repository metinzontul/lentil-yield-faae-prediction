# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
import numpy as np

import config

FAAE_NAME = "FAAE (Foundation-Augmented Averaging Ensemble)"


def build_faae_members():
    
    S = config.SEED
    return [
        TabPFNRegressor(device="cpu"),
        ExtraTreesRegressor(n_estimators=400, random_state=S, n_jobs=-1),
        CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4,
                          random_seed=S, verbose=0),
        XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                     subsample=0.9, colsample_bytree=0.9,
                     random_state=S, n_jobs=-1, verbosity=0),
    ]


class AveragingRegressor(BaseEstimator, RegressorMixin):


    def __init__(self, members=None):
        self.members = members

    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for m in self.members]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.fitted_], axis=0)


def build_faae(tuned=False):

    if not tuned:
        return AveragingRegressor(members=build_faae_members())

    from sklearn.model_selection import KFold, GridSearchCV
    inner = KFold(n_splits=3, shuffle=True, random_state=config.SEED)
    et = GridSearchCV(
        ExtraTreesRegressor(random_state=config.SEED, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [None, 16]},
        scoring="r2", cv=inner, n_jobs=1)
    cat = GridSearchCV(
        CatBoostRegressor(random_seed=config.SEED, verbose=0),
        {"iterations": [400], "depth": [4, 6], "learning_rate": [0.03, 0.1]},
        scoring="r2", cv=inner, n_jobs=1)
    xgb = GridSearchCV(
        XGBRegressor(random_state=config.SEED, n_jobs=-1, verbosity=0),
        {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.03, 0.1]},
        scoring="r2", cv=inner, n_jobs=1)
    tab = TabPFNRegressor(device="cpu")
    return AveragingRegressor(members=[tab, et, cat, xgb])
