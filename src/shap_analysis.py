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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import config
from data_utils import set_seed, load_dataset
from sklearn.preprocessing import StandardScaler
from faae_model import build_faae_members

N_BACKGROUND = 15
N_EXPLAIN = 40
N_SAMPLES = 120


def main():
    set_seed()
    X, y, feats, target, df = load_dataset(drop_features=config.DROP_FEATURES)
    y = y.ravel()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"SHAP + FAAE | {len(feats)} features (leakage-free) | target: {target}", flush=True)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    print("Fitting FAAE members on the full dataset (TabPFN+ET+Cat+XGB)...", flush=True)
    members = build_faae_members()
    for m in members:
        m.fit(Xs, y)

    def predict_fn(data):
        return np.mean([m.predict(data) for m in members], axis=0)

    bg = shap.kmeans(Xs, min(N_BACKGROUND, len(Xs)))
    X_explain = Xs[: min(N_EXPLAIN, len(Xs))]
    print(f"KernelExplainer ({N_BACKGROUND} kmeans, {len(X_explain)} samples, "
          f"nsamples={N_SAMPLES})...", flush=True)
    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = np.asarray(explainer.shap_values(X_explain, nsamples=N_SAMPLES))

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    total = float(mean_abs.sum()) or 1.0
    rows, cum = [], 0.0
    for rank, idx in enumerate(order, 1):
        rel = float(mean_abs[idx]) / total * 100.0
        cum += rel
        rows.append({"rank": rank, "feature": feats[idx],
                     "mean_abs_shap": float(mean_abs[idx]),
                     "relative_pct": rel, "cumulative_pct": cum})

    print("\n=== FAAE GLOBAL SHAP IMPORTANCE ===")
    print(f"{'#':<3}{'Feature':<28}{'mean|SHAP|':>12}{'Rel %':>9}{'Cum %':>9}")
    print("-" * 61)
    for r in rows:
        print(f"{r['rank']:<3}{r['feature']:<28}{r['mean_abs_shap']:>12.4f}"
              f"{r['relative_pct']:>9.2f}{r['cumulative_pct']:>9.2f}")

    base = os.path.join(config.RESULTS_DIR, "shap_faae_importance")
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["rank", "feature", "mean_abs_shap", "relative_pct", "cumulative_pct"])
        for r in rows:
            w.writerow([r["rank"], r["feature"], f"{r['mean_abs_shap']:.4f}",
                        f"{r['relative_pct']:.2f}", f"{r['cumulative_pct']:.2f}"])
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"model": "FAAE", "target": target, "rows": rows}, f,
                  indent=2, ensure_ascii=False)

    plt.figure()
    shap.summary_plot(shap_values, X_explain, feature_names=feats, show=False)
    plt.title("SHAP summary plot — FAAE model", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "shap_faae_summary.png"),
                dpi=300, bbox_inches="tight"); plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_explain, feature_names=feats,
                      plot_type="bar", show=False)
    plt.title("Mean absolute SHAP importance — FAAE model", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "shap_faae_bar.png"),
                dpi=300, bbox_inches="tight"); plt.close()

    print("\nSaved: shap_faae_importance.{csv,md,json}, "
          "shap_faae_summary.png, shap_faae_bar.png")


if __name__ == "__main__":
    main()
