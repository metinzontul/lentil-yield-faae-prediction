# -*- coding: utf-8 -*-


from sklearn.linear_model import Ridge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor

import config

FIXED_FAAE_MEMBERS = ["TabPFN", "Extra Trees", "CatBoost", "XGBoost"]

# Members selected per fold under nested member selection
TOP_K = 4


def candidate_specs():

    S = config.SEED
    return {
        "Ridge Regression": (
            Ridge(),
            {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        "Random Forest": (
            RandomForestRegressor(random_state=S, n_jobs=1),
            {"n_estimators": [200, 400], "max_depth": [None, 8, 16]}),
        "Extra Trees": (
            ExtraTreesRegressor(random_state=S, n_jobs=1),
            {"n_estimators": [200, 400], "max_depth": [None, 16]}),
        "XGBoost": (
            XGBRegressor(random_state=S, n_jobs=1, verbosity=0),
            {"n_estimators": [200, 400], "max_depth": [3, 5],
             "learning_rate": [0.03, 0.1]}),
        "LightGBM": (
            LGBMRegressor(random_state=S, n_jobs=1, verbose=-1),
            {"n_estimators": [300, 500], "max_depth": [3, 5],
             "learning_rate": [0.03, 0.1]}),
        "CatBoost": (
            CatBoostRegressor(random_seed=S, verbose=0),
            {"iterations": [400], "depth": [4, 6],
             "learning_rate": [0.03, 0.1]}),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=S),
            {"n_estimators": [300], "max_depth": [2, 3],
             "learning_rate": [0.03, 0.1]}),
        "TabPFN": (
            TabPFNRegressor(device="cpu"), {}),
    }


def make_pipe(est):
    """Wrap an estimator with StandardScaler, as in nested_cv.py."""
    return Pipeline([("scaler", StandardScaler()), ("model", est)])
