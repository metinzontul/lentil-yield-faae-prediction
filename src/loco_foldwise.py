# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore")

import os
import sys
import csv
import json
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import config
from data_utils import set_seed, load_dataset
from faae_model import build_faae_members

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def faae_fold_r2(Xin, y, splits):
    """Per-fold R2 for FAAE (no averaging)."""
    scores = []
    for tr, te in splits:
        sx = StandardScaler().fit(Xin[tr])
        Xtr, Xte = sx.transform(Xin[tr]), sx.transform(Xin[te])
        preds = []
        for m in build_faae_members():
            mm = clone(m).fit(Xtr, y[tr])
            preds.append(mm.predict(Xte))
        scores.append(float(r2_score(y[te], np.mean(preds, axis=0))))
    return np.array(scores)


def main():
    set_seed()
    X, y, feats, target, _ = load_dataset(drop_features=config.DROP_FEATURES)
    y = y.ravel()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    splits = list(KFold(n_splits=config.N_SPLITS, shuffle=True,
                        random_state=config.SEED).split(X))
    n_folds = len(splits)

    print(f"LOCO fold-wise | FAAE | {len(feats)} features (leakage-free) | "
          f"target: {target} | {n_folds} folds", flush=True)
    print("Fixed hyperparameters, as in loco_analysis.py\n", flush=True)

    t0 = time.time()
    print("Baseline FAAE (all features)...", flush=True)
    base_folds = faae_fold_r2(X, y, splits)
    print("  per-fold R2: " + ", ".join(f"{v:.4f}" for v in base_folds), flush=True)
    print(f"  mean = {base_folds.mean():.4f} +/- {base_folds.std(ddof=1):.4f}\n",
          flush=True)

    rows = []
    for j, fname in enumerate(feats):
        keep = [k for k in range(len(feats)) if k != j]
        red = faae_fold_r2(X[:, keep], y, splits)
        d = base_folds - red                       # paired within fold
        rows.append({
            "feature": fname,
            "r2_without_per_fold": red.tolist(),
            "delta_r2_per_fold": d.tolist(),
            "delta_r2_mean": float(d.mean()),
            "delta_r2_sd": float(d.std(ddof=1)),
            "delta_r2_min": float(d.min()),
            "delta_r2_max": float(d.max()),
            "n_folds_positive": int((d > 0).sum()),
        })
        print(f"  {fname:<28} dR2 = {d.mean():+.4f} +/- {d.std(ddof=1):.4f}  "
              "(folds: " + ", ".join(f"{v:+.4f}" for v in d) + ")", flush=True)

    elapsed = time.time() - t0
    rows.sort(key=lambda r: -r["delta_r2_mean"])

    out = {
        "analysis": "LOCO fold-wise variability (FAAE)",
        "delta_definition": ("delta_R2(f, fold k) = R2_full(fold k) - "
                             "R2_without_f(fold k); paired within each fold"),
        "protocol": {
            "folds": f"KFold({config.N_SPLITS}, shuffle=True, random_state={config.SEED})",
            "scaling": "StandardScaler fitted on the training fold only",
            "hyperparameters": "fixed (same as loco_analysis.py)",
            "seed": config.SEED,
        },
        "target": target, "n_samples": int(len(X)), "n_features": len(feats),
        "features": feats,
        "baseline_r2_per_fold": base_folds.tolist(),
        "baseline_r2_mean": float(base_folds.mean()),
        "baseline_r2_sd": float(base_folds.std(ddof=1)),
        "per_feature": rows,
        "runtime_seconds": round(elapsed, 1),
        "interpretation_caveat": ("Small differences among lower-ranked features "
                                  "fall within the fold-to-fold spread and do not "
                                  "establish a rank ordering."),
    }

    base = os.path.join(config.RESULTS_DIR, "loco_foldwise")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Feature"] + [f"dR2_fold{k}" for k in range(1, n_folds + 1)] +
                   ["dR2_mean", "dR2_SD", "dR2_min", "dR2_max", "folds_positive"])
        for r in rows:
            w.writerow([r["feature"]] +
                       [f"{v:.4f}" for v in r["delta_r2_per_fold"]] +
                       [f"{r['delta_r2_mean']:.4f}", f"{r['delta_r2_sd']:.4f}",
                        f"{r['delta_r2_min']:.4f}", f"{r['delta_r2_max']:.4f}",
                        f"{r['n_folds_positive']}/{n_folds}"])
        w.writerow([])
        w.writerow(["BASELINE_R2"] + [f"{v:.4f}" for v in base_folds] +
                   [f"{base_folds.mean():.4f}", f"{base_folds.std(ddof=1):.4f}",
                    "", "", ""])


    print(f"\nRuntime: {elapsed/60:.1f} min")
    print(f"  --> Saved: {base}.json / .csv")


if __name__ == "__main__":
    main()
