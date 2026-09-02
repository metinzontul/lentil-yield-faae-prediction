# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore")

import os
import sys
import csv
import json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
from data_utils import set_seed, load_dataset
from nested_cv import nested_cv_estimator
from faae_model import build_faae, FAAE_NAME


def fmt(m, s):
    return f"{m:.4f} +/- {s:.4f}"


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
    print(f"Saved: {base}.json / .csv", flush=True)


def main():
    set_seed()
    X, y, feats, target, _ = load_dataset(drop_features=config.DROP_FEATURES)
    print(f"FAAE (nested CV, {config.N_SPLITS} outer folds) | "
          f"{len(feats)} features (leakage-free) | target: {target}\n", flush=True)

    res = nested_cv_estimator(FAAE_NAME, build_faae(tuned=True), {}, X, y)

    print("\n=== FAAE (nested, 5 outer folds) ===")
    print(f"  R2   : {fmt(res['R2_mean'], res['R2_std'])}")
    print(f"  RMSE : {fmt(res['RMSE_mean'], res['RMSE_std'])}")
    print(f"  MAE  : {fmt(res['MAE_mean'], res['MAE_std'])}")

    base = os.path.join(config.RESULTS_DIR, "nested_cv")
    if os.path.exists(base + ".json"):
        with open(base + ".json", encoding="utf-8") as f:
            results = json.load(f)
    else:
        print("\nWarning: results/nested_cv.json not found; run baseline_models.py "
              "first for the full comparison table. Saving FAAE alone for now.")
        results = []

    results = [r for r in results if r["name"] != FAAE_NAME]
    results.append(res)
    results.sort(key=lambda r: r["R2_mean"], reverse=True)
    save(results)

    print("\nRanking (by R2):")
    for i, r in enumerate(results, 1):
        print(f"  {i:>2}. {r['name']:<45} R2={fmt(r['R2_mean'], r['R2_std'])}")


if __name__ == "__main__":
    main()
