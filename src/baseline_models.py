# -*- coding: utf-8 -*-


import os
import sys
import csv
import json
import warnings
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
from data_utils import set_seed, load_dataset
from nested_cv import nested_cv_estimator

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, StackingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor


def fmt(m, s):
    return f"{m:.4f} +/- {s:.4f}"


def make_hybrid():
    """TabPFN + Extra Trees -> Ridge meta-learner (out-of-fold stacking). No inner tuning."""
    base = [
        ("tabpfn", TabPFNRegressor(device="cpu")),
        ("extratrees", ExtraTreesRegressor(n_estimators=400,
                                           random_state=config.SEED, n_jobs=-1)),
    ]
    from sklearn.model_selection import KFold
    return StackingRegressor(
        estimators=base, final_estimator=Ridge(alpha=1.0),
        cv=KFold(n_splits=5, shuffle=True, random_state=config.SEED),
        n_jobs=1, passthrough=False,
    )


def save(results):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    base = os.path.join(config.RESULTS_DIR, "nested_cv")

    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "R2_mean", "R2_std", "RMSE_mean", "RMSE_std",
                    "MAE_mean", "MAE_std", "outer_splits"])
        for r in results:
            w.writerow([r["name"], f"{r['R2_mean']:.4f}", f"{r['R2_std']:.4f}",
                        f"{r['RMSE_mean']:.4f}", f"{r['RMSE_std']:.4f}",
                        f"{r['MAE_mean']:.4f}", f"{r['MAE_std']:.4f}",
                        r.get("outer_splits")])

    for r in sorted(results, key=lambda r: r["R2_mean"], reverse=True):
        tag = "" if r.get("tuned", True) else " (no tuning)"
    print(f"  --> Saved: {base}.json / .csv", flush=True)


def run_model(idx, total, label, fn, results):
    print(f"\n[{idx}/{total}] {label} starting...", flush=True)
    results.append(fn())
    r = results[-1]
    print(f"  R2   : {fmt(r['R2_mean'], r['R2_std'])}", flush=True)
    print(f"  RMSE : {fmt(r['RMSE_mean'], r['RMSE_std'])}", flush=True)
    print(f"  MAE  : {fmt(r['MAE_mean'], r['MAE_std'])}", flush=True)
    save(results)


def main():
    set_seed()

    print("Loading data...", flush=True)
    X, y, feats, target, _ = load_dataset(drop_features=config.DROP_FEATURES)
    print(f"Target: {target} | N={len(X)} | Features={len(feats)} (leakage-free) "
          f"| dropped: {', '.join(config.DROP_FEATURES)}", flush=True)

    # (label, estimator, inner-tuning grid). Empty grid ({}) -> no tuning.
    specs = [
        ("Ridge Regression",
         lambda: nested_cv_estimator(
             "Ridge Regression", Ridge(),
             {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, X, y)),
        ("Random Forest",
         lambda: nested_cv_estimator(
             "Random Forest", RandomForestRegressor(random_state=config.SEED, n_jobs=1),
             {"n_estimators": [200, 400], "max_depth": [None, 8, 16]}, X, y)),
        ("Extra Trees",
         lambda: nested_cv_estimator(
             "Extra Trees", ExtraTreesRegressor(random_state=config.SEED, n_jobs=1),
             {"n_estimators": [200, 400], "max_depth": [None, 16]}, X, y)),
        ("XGBoost",
         lambda: nested_cv_estimator(
             "XGBoost", XGBRegressor(random_state=config.SEED, n_jobs=1, verbosity=0),
             {"n_estimators": [200, 400], "max_depth": [3, 5],
              "learning_rate": [0.03, 0.1]}, X, y)),
        ("LightGBM",
         lambda: nested_cv_estimator(
             "LightGBM", LGBMRegressor(random_state=config.SEED, n_jobs=1, verbose=-1),
             {"n_estimators": [300, 500], "max_depth": [3, 5],
              "learning_rate": [0.03, 0.1]}, X, y)),
        ("CatBoost",
         lambda: nested_cv_estimator(
             "CatBoost", CatBoostRegressor(random_seed=config.SEED, verbose=0),
             {"iterations": [400], "depth": [4, 6],
              "learning_rate": [0.03, 0.1]}, X, y)),
        ("Gradient Boosting",
         lambda: nested_cv_estimator(
             "Gradient Boosting", GradientBoostingRegressor(random_state=config.SEED),
             {"n_estimators": [300], "max_depth": [2, 3],
              "learning_rate": [0.03, 0.1]}, X, y)),
        ("TabPFN",
         lambda: nested_cv_estimator(
             "TabPFN", TabPFNRegressor(device="cpu"), {}, X, y)),
        ("Hybrid (TabPFN+ExtraTrees)",
         lambda: nested_cv_estimator(
             "Hybrid (TabPFN+ExtraTrees)", make_hybrid(), {}, X, y)),
    ]

    total = len(specs)
    print(f"Running {total} models.\n", flush=True)

    results = []
    for i, (label, fn) in enumerate(specs, 1):
        run_model(i, total, label, fn, results)

    results.sort(key=lambda r: r["R2_mean"], reverse=True)
    print("\n=== DONE (sorted by R2) ===", flush=True)
    print(f"{'Model':<28} {'R2':>16} {'RMSE':>16} {'MAE':>16}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        print(f"{r['name']:<28} {fmt(r['R2_mean'], r['R2_std']):>16} "
              f"{fmt(r['RMSE_mean'], r['RMSE_std']):>16} "
              f"{fmt(r['MAE_mean'], r['MAE_std']):>16}", flush=True)
    save(results)


if __name__ == "__main__":
    main()
