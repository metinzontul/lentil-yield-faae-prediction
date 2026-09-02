# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore")

import os
import sys
import csv
import json
import time
import itertools

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import config
from data_utils import set_seed, load_dataset
from candidates import candidate_specs, make_pipe, TOP_K, FIXED_FAAE_MEMBERS

from sklearn.base import clone
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CLASSICAL = ["Extra Trees", "CatBoost", "XGBoost"]
FAAE_FIXED_NAME = "FAAE (fixed post-hoc composition)"
FAAE_NESTED_NAME = "FAAE (fully nested member selection)"


def metrics(y_true, y_pred):
    return {"R2": float(r2_score(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred))}


def main():
    set_seed()
    X, y, feats, target, _ = load_dataset(drop_features=config.DROP_FEATURES)
    y = y.ravel()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    specs = candidate_specs()
    names = list(specs)

    outer = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    inner = KFold(n_splits=3, shuffle=True, random_state=config.SEED)

    print(f"Ablation + pooled OOF | {len(feats)} features (leakage-free) | "
          f"target: {target}", flush=True)
    print(f"N={len(X)} | outer {config.N_SPLITS}-fold + inner 3-fold "
          f"GridSearchCV | seed={config.SEED}\n", flush=True)

    configs = {
        "TabPFN alone": ["TabPFN"],
        "Classical ensemble (ET+Cat+XGB)": CLASSICAL,
        "FAAE (TabPFN+ET+Cat+XGB)": FIXED_FAAE_MEMBERS,
        f"All-model ensemble ({len(names)} models)": names,
    }

    per_fold = {c: {"R2": [], "RMSE": [], "MAE": []} for c in configs}
    fold_rows, nested_sel = [], []
    oof = {n: np.full(len(y), np.nan) for n in names}
    oof[FAAE_FIXED_NAME] = np.full(len(y), np.nan)
    oof[FAAE_NESTED_NAME] = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    fold_r2 = {k: [] for k in list(names) + [FAAE_FIXED_NAME, FAAE_NESTED_NAME]}

    t0 = time.time()
    for k, (tr, te) in enumerate(outer.split(X), 1):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        fold_id[te] = k
        print(f"[Fold {k}/{config.N_SPLITS}] tuning "
              f"(n_train={len(tr)}, n_test={len(te)})...", flush=True)

        preds, inner_r2 = {}, {}
        for n in names:
            est, grid = specs[n]
            pipe = make_pipe(clone(est))
            if grid:
                gs = GridSearchCV(pipe, {f"model__{p}": v for p, v in grid.items()},
                                  scoring="r2", cv=inner, n_jobs=1)
                gs.fit(Xtr, ytr)
                preds[n] = gs.best_estimator_.predict(Xte)
                inner_r2[n] = float(gs.best_score_)
            else:
                m = clone(pipe).fit(Xtr, ytr)
                preds[n] = m.predict(Xte)
                inner_r2[n] = float(np.mean(
                    cross_val_score(pipe, Xtr, ytr, scoring="r2", cv=inner, n_jobs=1)))
            oof[n][te] = preds[n]
            fold_r2[n].append(float(r2_score(yte, preds[n])))

        # (A) ablation configurations
        row = {"fold": k}
        for cname, members in configs.items():
            avg = np.mean([preds[m] for m in members], axis=0)
            mm = metrics(yte, avg)
            for key in ("R2", "RMSE", "MAE"):
                per_fold[cname][key].append(mm[key])
            row[cname] = mm
            print(f"    {cname:<34} R2={mm['R2']:.4f}", flush=True)
        fold_rows.append(row)

        # (C) FAAE variants for pooled OOF
        avg_fixed = np.mean([preds[m] for m in FIXED_FAAE_MEMBERS], axis=0)
        oof[FAAE_FIXED_NAME][te] = avg_fixed
        fold_r2[FAAE_FIXED_NAME].append(float(r2_score(yte, avg_fixed)))

        top = sorted(names, key=lambda n: inner_r2[n], reverse=True)[:TOP_K]
        nested_sel.append({"fold": k, "selected_members": top})
        avg_nested = np.mean([preds[m] for m in top], axis=0)
        oof[FAAE_NESTED_NAME][te] = avg_nested
        fold_r2[FAAE_NESTED_NAME].append(float(r2_score(yte, avg_nested)))
        print(f"    nested selection: {', '.join(top)}\n", flush=True)

    elapsed = time.time() - t0
    for n, v in oof.items():
        assert not np.isnan(v).any(), f"incomplete OOF: {n}"

    # ---------------- (A) ablation summary ----------------
    summary = {}
    for cname in configs:
        d = per_fold[cname]
        summary[cname] = {
            "members": configs[cname], "n_members": len(configs[cname]),
            "R2_mean": float(np.mean(d["R2"])), "R2_std": float(np.std(d["R2"])),
            "RMSE_mean": float(np.mean(d["RMSE"])), "RMSE_std": float(np.std(d["RMSE"])),
            "MAE_mean": float(np.mean(d["MAE"])), "MAE_std": float(np.std(d["MAE"])),
            "per_fold_R2": d["R2"],
        }

    # ---------------- (B) OOF correlations ----------------
    resid = {n: y - oof[n] for n in FIXED_FAAE_MEMBERS}

    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    resid_corr, pred_corr = {}, {}
    for a, b in itertools.combinations(FIXED_FAAE_MEMBERS, 2):
        resid_corr[f"{a} vs {b}"] = pearson(resid[a], resid[b])
        pred_corr[f"{a} vs {b}"] = pearson(oof[a], oof[b])

    def matrix(vmap):
        return {a: {b: (1.0 if a == b else pearson(vmap[a], vmap[b]))
                    for b in FIXED_FAAE_MEMBERS} for a in FIXED_FAAE_MEMBERS}

    ablation = {
        "analysis": "FAAE ablation + OOF residual correlations",
        "protocol_note": ("All configurations use the same tuned models and the "
                          "same folds. FAAE here is the fixed composition; "
                          "nested member selection is reported separately."),
        "protocol": {
            "outer": f"KFold({config.N_SPLITS}, shuffle=True, random_state={config.SEED})",
            "inner": f"KFold(3, shuffle=True, random_state={config.SEED}) GridSearchCV",
            "scaling": "StandardScaler inside Pipeline", "seed": config.SEED,
        },
        "target": target, "n_samples": int(len(X)), "n_features": len(feats),
        "features": feats, "candidates": names,
        "configurations": summary,
        "residual_correlations_pairwise": resid_corr,
        "prediction_correlations_pairwise": pred_corr,
        "residual_correlation_matrix": matrix(resid),
        "prediction_correlation_matrix": matrix({n: oof[n] for n in FIXED_FAAE_MEMBERS}),
        "per_fold": fold_rows,
        "runtime_seconds": round(elapsed, 1),
    }

    base_a = os.path.join(config.RESULTS_DIR, "faae_ablation")
    with open(base_a + ".json", "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2, ensure_ascii=False)
    with open(base_a + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Configuration", "n_members", "R2_mean", "R2_std",
                    "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std"])
        for c, s in summary.items():
            w.writerow([c, s["n_members"], f"{s['R2_mean']:.4f}", f"{s['R2_std']:.4f}",
                        f"{s['RMSE_mean']:.4f}", f"{s['RMSE_std']:.4f}",
                        f"{s['MAE_mean']:.4f}", f"{s['MAE_std']:.4f}"])
        w.writerow([])
        w.writerow(["Pair", "residual_pearson_r", "prediction_pearson_r",
                    "", "", "", "", ""])
        for kk in resid_corr:
            w.writerow([kk, f"{resid_corr[kk]:.4f}", f"{pred_corr[kk]:.4f}",
                        "", "", "", "", ""])

    # ---------------- (C) pooled OOF ----------------
    report_order = ([FAAE_FIXED_NAME, FAAE_NESTED_NAME, "TabPFN", "Extra Trees",
                     "CatBoost", "XGBoost", "Gradient Boosting", "Random Forest",
                     "LightGBM", "Ridge Regression"])
    pooled_rows = []
    for n in report_order:
        p = metrics(y, oof[n])
        fr = np.array(fold_r2[n])
        pooled_rows.append({"model": n, **p,
                            "fold_R2_mean": float(fr.mean()),
                            "fold_R2_std": float(fr.std()),
                            "per_fold_R2": [float(v) for v in fr]})

    pooled = {
        "analysis": "Pooled out-of-fold metrics",
        "pooled_definition": ("Outer-test predictions concatenated into a single "
                              "vector of length N, with each metric computed once "
                              "on it rather than averaged over folds."),
        "protocol": ablation["protocol"],
        "target": target, "n_samples": int(len(X)), "n_features": len(feats),
        "fixed_faae_members": FIXED_FAAE_MEMBERS,
        "nested_selection_per_fold": nested_sel,
        "pooled_metrics": pooled_rows,
        "runtime_seconds": round(elapsed, 1),
    }

    base_p = os.path.join(config.RESULTS_DIR, "pooled_oof")
    with open(base_p + ".json", "w", encoding="utf-8") as f:
        json.dump(pooled, f, indent=2, ensure_ascii=False)
    with open(base_p + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Model", "pooled_R2", "pooled_RMSE", "pooled_MAE",
                    "foldwise_R2_mean", "foldwise_R2_SD"])
        for r in pooled_rows:
            w.writerow([r["model"], f"{r['R2']:.4f}", f"{r['RMSE']:.4f}",
                        f"{r['MAE']:.4f}", f"{r['fold_R2_mean']:.4f}",
                        f"{r['fold_R2_std']:.4f}"])
    with open(base_p + "_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["index", "outer_fold", "y_true"] + report_order)
        for i in range(len(y)):
            w.writerow([i, int(fold_id[i]), f"{y[i]:.4f}"] +
                       [f"{oof[n][i]:.4f}" for n in report_order])

    # ---------------- console summary ----------------
    print("=== CONFIGURATIONS ===")
    hdr = f"{'Configuration':<36}{'R2':>17} {'RMSE':>17} {'MAE':>17}"
    print(hdr); print("-" * len(hdr))
    for c, s in summary.items():
        print(f"{c:<36}{s['R2_mean']:.4f} +/- {s['R2_std']:.4f}  "
              f"{s['RMSE_mean']:.4f} +/- {s['RMSE_std']:.4f}  "
              f"{s['MAE_mean']:.4f} +/- {s['MAE_std']:.4f}")

    print("\n=== PAIRWISE PEARSON CORRELATIONS (OOF) ===")
    print(f"{'Pair':<34}{'residual r':>12}{'prediction r':>14}")
    print("-" * 60)
    for kk in resid_corr:
        print(f"{kk:<34}{resid_corr[kk]:>12.4f}{pred_corr[kk]:>14.4f}")

    print(f"\n=== POOLED OOF METRICS (N={len(X)}, computed once) ===")
    hdr = f"{'Model':<38}{'R2':>9}{'RMSE':>10}{'MAE':>9}{'foldSD':>9}"
    print(hdr); print("-" * len(hdr))
    for r in pooled_rows:
        print(f"{r['model']:<38}{r['R2']:>9.4f}{r['RMSE']:>10.4f}"
              f"{r['MAE']:>9.4f}{r['fold_R2_std']:>9.4f}")

    print(f"\nRuntime: {elapsed/60:.1f} min")
    print(f"  --> Saved: {base_a}.* and {base_p}.*")


if __name__ == "__main__":
    main()
