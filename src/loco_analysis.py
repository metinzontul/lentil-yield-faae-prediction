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

import config
from data_utils import set_seed, load_dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.base import clone
from faae_model import build_faae_members


def faae_cv_r2(Xin, y, splits):
    scores = []
    for tr, te in splits:
        sx = StandardScaler().fit(Xin[tr])
        Xtr, Xte = sx.transform(Xin[tr]), sx.transform(Xin[te])
        preds = []
        for m in build_faae_members():
            mm = clone(m); mm.fit(Xtr, y[tr]); preds.append(mm.predict(Xte))
        scores.append(r2_score(y[te], np.mean(preds, axis=0)))
    return np.array(scores)


def main():
    set_seed()
    X, y, feats, target, df = load_dataset(drop_features=config.DROP_FEATURES)
    y = y.ravel()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Single outer 5-fold pass for speed
    splits = list(KFold(n_splits=config.N_SPLITS, shuffle=True,
                        random_state=config.SEED).split(X))
    print(f"LOCO + FAAE | {len(feats)} features (leakage-free) | target: {target} | "
          f"{len(splits)} folds\n", flush=True)

    from tqdm import tqdm
    print("Baseline FAAE (all features)...", flush=True)
    base_r2 = float(faae_cv_r2(X, y, splits).mean())
    print(f"  Baseline R2 = {base_r2:.4f}\n")

    rows = []
    pbar = tqdm(total=len(feats), desc="  LOCO-FAAE", unit="feat",
                file=sys.stdout, dynamic_ncols=True, ascii=True)
    for j, fname in enumerate(feats):
        keep = [k for k in range(len(feats)) if k != j]
        red_r2 = float(faae_cv_r2(X[:, keep], y, splits).mean())
        drop = base_r2 - red_r2
        rows.append({"feature": fname, "r2_without": red_r2, "loco_drop": drop})
        pbar.set_postfix_str(f"{fname[:16]} dR2={drop:+.4f}")
        pbar.update(1)
    pbar.close()

    rows.sort(key=lambda r: r["loco_drop"], reverse=True)
    total_drop = sum(max(0.0, r["loco_drop"]) for r in rows) or 1.0
    for r in rows:
        r["relative_pct"] = max(0.0, r["loco_drop"]) / total_drop * 100.0

    print("\n=== LOCO FEATURE CONTRIBUTION (FAAE) ===")
    print(f"Baseline R2 = {base_r2:.4f}")
    print(f"{'#':<3}{'Feature':<28}{'R2 without':>12}{'dR2':>11}{'Rel %':>9}")
    print("-" * 63)
    for i, r in enumerate(rows, 1):
        print(f"{i:<3}{r['feature']:<28}{r['r2_without']:>12.4f}"
              f"{r['loco_drop']:>+11.4f}{r['relative_pct']:>9.2f}")

    base = os.path.join(config.RESULTS_DIR, "loco_faae")
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["rank", "feature", "r2_without", "loco_drop", "relative_pct"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["feature"], f"{r['r2_without']:.4f}",
                        f"{r['loco_drop']:.4f}", f"{r['relative_pct']:.2f}"])
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"model": "FAAE", "target": target, "base_r2": base_r2,
                   "rows": rows}, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(9, 5.5))
    names = [r["feature"] for r in rows][::-1]
    vals = [r["loco_drop"] for r in rows][::-1]
    colors = ["tab:green" if v >= 0 else "tab:red" for v in vals]
    plt.barh(names, vals, color=colors, alpha=0.85)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("ΔR² (LOCO importance = R²_full − R²_without)")
    plt.title("LOCO feature contribution — FAAE (leakage-free, 7 features)")
    plt.grid(True, axis="x", alpha=0.3); plt.tight_layout()
    plt.savefig(base + ".png", dpi=300); plt.close()

    print("\nSaved: loco_faae.{csv,md,json}, loco_faae.png")


if __name__ == "__main__":
    main()
