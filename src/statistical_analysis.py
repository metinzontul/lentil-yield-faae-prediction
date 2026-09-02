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
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp

import config
from data_utils import set_seed, load_dataset
from faae_model import build_faae, FAAE_NAME
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor


def make_models():
    """Baseline models (Hybrid excluded — dominated by TabPFN alone) plus FAAE."""
    return [
        ("TabPFN", lambda: TabPFNRegressor(device="cpu")),
        ("Extra Trees", lambda: ExtraTreesRegressor(n_estimators=400, random_state=config.SEED, n_jobs=-1)),
        ("CatBoost", lambda: CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4,
                                               random_seed=config.SEED, verbose=0)),
        ("XGBoost", lambda: XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                                         subsample=0.9, colsample_bytree=0.9,
                                         random_state=config.SEED, n_jobs=-1, verbosity=0)),
        ("Gradient Boosting", lambda: GradientBoostingRegressor(n_estimators=400, learning_rate=0.03,
                                                                max_depth=3, subsample=0.9,
                                                                random_state=config.SEED)),
        ("Random Forest", lambda: RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)),
        ("LightGBM", lambda: LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=4,
                                           subsample=0.9, colsample_bytree=0.9,
                                           random_state=config.SEED, n_jobs=-1, verbose=-1)),
        ("Ridge Regression", lambda: Ridge(alpha=1.0)),
        (FAAE_NAME, lambda: build_faae(tuned=False)),
    ]


def main():
    set_seed()
    X, y, feats, target, df = load_dataset(drop_features=config.DROP_FEATURES)
    y = y.ravel()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    models = make_models()
    names = [n for n, _ in models]

    rkf = RepeatedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS,
                        random_state=config.SEED)
    splits = list(rkf.split(X))
    n_folds = len(splits)
    print(f"Leakage-free | {len(feats)} features | {len(names)} models | "
          f"{n_folds} folds ({config.N_SPLITS}x{config.N_REPEATS})\n")

    from tqdm import tqdm
    # R2 matrix: [n_folds x n_models]
    R2 = np.zeros((n_folds, len(names)))
    for j, (name, build) in enumerate(models):
        pbar = tqdm(total=n_folds, desc=f"  {name}", unit="fold",
                    file=sys.stdout, dynamic_ncols=True, ascii=True)
        for i, (tr, te) in enumerate(splits):
            sx = StandardScaler().fit(X[tr])
            Xtr, Xte = sx.transform(X[tr]), sx.transform(X[te])
            m = build()
            m.fit(Xtr, y[tr])
            R2[i, j] = r2_score(y[te], m.predict(Xte))
            pbar.update(1)
        pbar.close()

    # ---- Friedman test ----
    stat, p_fried = friedmanchisquare(*[R2[:, j] for j in range(len(names))])
    # Mean rank (higher R2 = better -> reverse ordering)
    ranks = np.zeros_like(R2)
    for i in range(n_folds):
        order = (-R2[i]).argsort()
        r = np.empty(len(names))
        r[order] = np.arange(1, len(names) + 1)
        ranks[i] = r
    mean_rank = ranks.mean(axis=0)

    # ---- Post-hoc Nemenyi ----
    nem = sp.posthoc_nemenyi_friedman(R2)   # DataFrame, rows/cols = model index
    nem.index = names; nem.columns = names

    # ---- Wilcoxon vs best (Holm-corrected) ----
    best = int(np.argmin(mean_rank))
    best_name = names[best]
    wilcox = []
    for j in range(len(names)):
        if j == best:
            continue
        try:
            w, pw = wilcoxon(R2[:, best], R2[:, j])
        except Exception:
            pw = 1.0
        wilcox.append((names[j], float(pw)))
    wilcox.sort(key=lambda t: t[1])
    m = len(wilcox)
    holm = []
    for k, (nm, pw) in enumerate(wilcox):
        p_adj = min(1.0, pw * (m - k))
        holm.append((nm, pw, p_adj))

    # ---- Print summary ----
    print("\n=== STATISTICAL SIGNIFICANCE ===")
    print(f"Friedman chi2 = {stat:.3f}, p = {p_fried:.3e} "
          f"({'significant difference' if p_fried < 0.05 else 'no significant difference'})")
    print(f"\n{'Model':<32}{'R2 mean':>10}{'R2 std':>9}{'mean rank':>11}")
    print("-" * 62)
    order = np.argsort(mean_rank)
    for j in order:
        print(f"{names[j]:<32}{R2[:, j].mean():>10.4f}{R2[:, j].std():>9.4f}{mean_rank[j]:>11.2f}")

    print(f"\nBest (lowest rank): {best_name}")
    print(f"{'vs Model':<32}{'Wilcoxon p':>12}{'Holm p':>10}{'':>6}")
    print("-" * 60)
    for nm, pw, pa in holm:
        sig = "*" if pa < 0.05 else "n.s."
        print(f"{nm:<32}{pw:>12.4f}{pa:>10.4f}{sig:>6}")
    print("\n(* = significantly different from the best model; n.s. = not significant / equivalent)")

    # ---- Save: per-fold matrix ----
    with open(os.path.join(config.RESULTS_DIR, "stats_perfold_r2.csv"),
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["fold"] + names)
        for i in range(n_folds):
            w.writerow([i + 1] + [f"{R2[i, j]:.4f}" for j in range(len(names))])

    # ---- Save: model comparison + tests ----
    comp = {
        "friedman": {"chi2": float(stat), "p_value": float(p_fried)},
        "best_model": best_name,
        "models": [{"name": names[j], "R2_mean": float(R2[:, j].mean()),
                    "R2_std": float(R2[:, j].std()), "mean_rank": float(mean_rank[j])}
                   for j in order],
        "wilcoxon_vs_best_holm": [{"model": nm, "p_raw": pw, "p_holm": pa} for nm, pw, pa in holm],
        "n_folds": n_folds, "n_features": len(feats),
    }
    with open(os.path.join(config.RESULTS_DIR, "stats_model_comparison.json"),
              "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2, ensure_ascii=False)

    with open(os.path.join(config.RESULTS_DIR, "stats_model_comparison.csv"),
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["model", "R2_mean", "R2_std", "mean_rank",
                    "wilcoxon_p_vs_best", "holm_p_vs_best"])
        wd = {nm: (pw, pa) for nm, pw, pa in holm}
        for j in order:
            nm = names[j]
            pw, pa = wd.get(nm, ("", ""))
            w.writerow([nm, f"{R2[:, j].mean():.4f}", f"{R2[:, j].std():.4f}",
                        f"{mean_rank[j]:.2f}",
                        f"{pw:.4f}" if pw != "" else "-",
                        f"{pa:.4f}" if pa != "" else "-"])

    nem.to_csv(os.path.join(config.RESULTS_DIR, "stats_nemenyi_pvalues.csv"),
               sep=";", encoding="utf-8-sig")

    wd = {nm: (pw, pa) for nm, pw, pa in holm}
    for j in order:
        nm = names[j]
        pw, pa = wd.get(nm, ("-", "-"))
        pw_s = f"{pw:.4f}" if pw != "-" else "— (best)"
        pa_s = f"{pa:.4f}" if pa != "-" else "—"

    # ---- Figure 1: boxplot (per-fold R2) ----
    plt.figure(figsize=(11, 6))
    data = [R2[:, j] for j in order]
    plt.boxplot(data, labels=[names[j] for j in order], showmeans=True)
    plt.ylabel("Fold-wise R²")
    plt.title("Per-fold R² distribution across models (leakage-free, 7 features)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "stats_boxplot.png"), dpi=300)
    plt.close()

    # ---- Figure 2: mean rank bar ----
    plt.figure(figsize=(9, 6))
    plt.barh([names[j] for j in order[::-1]], [mean_rank[j] for j in order[::-1]],
             color="tab:blue", alpha=0.8)
    plt.xlabel("Mean rank (lower = better)")
    plt.title("Model mean rank (Friedman)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "stats_mean_rank.png"), dpi=300)
    plt.close()

    # ---- Figure 3: Nemenyi p-value heatmap ----
    plt.figure(figsize=(8.5, 7))
    order_names = [names[j] for j in order]
    M = nem.loc[order_names, order_names].values.astype(float)
    im = plt.imshow(M, cmap="viridis_r", vmin=0, vmax=1)
    plt.colorbar(im, label="Nemenyi p-value")
    plt.xticks(range(len(order_names)), order_names, rotation=45, ha="right")
    plt.yticks(range(len(order_names)), order_names)
    for a in range(len(order_names)):
        for b in range(len(order_names)):
            plt.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center",
                     color="white" if M[a, b] < 0.5 else "black", fontsize=7)
    plt.title("Post-hoc Nemenyi pairwise p-values")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "stats_nemenyi_heatmap.png"), dpi=300)
    plt.close()

    print("\nSaved: stats_perfold_r2.csv, stats_model_comparison.{csv,json,md}, "
          "stats_nemenyi_pvalues.csv, stats_boxplot.png, stats_mean_rank.png, "
          "stats_nemenyi_heatmap.png")


if __name__ == "__main__":
    main()
