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
from candidates import candidate_specs, make_pipe, TOP_K, FIXED_FAAE_MEMBERS

from sklearn.base import clone
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def nested_member_selection(X, y, verbose=True):
    """Return aggregate metrics and per-fold detail for nested selection."""
    specs = candidate_specs()
    names = list(specs)

    outer = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    inner = KFold(n_splits=3, shuffle=True, random_state=config.SEED)

    r2s, rmses, maes, folds = [], [], [], []

    for k, (tr, te) in enumerate(outer.split(X), 1):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        if verbose:
            print(f"[Fold {k}/{config.N_SPLITS}] inner tuning "
                  f"(n_train={len(tr)}, n_test={len(te)})...", flush=True)

        inner_r2, best_params, fitted = {}, {}, {}
        for n in names:
            est, grid = specs[n]
            pipe = make_pipe(clone(est))
            if grid:
                gs = GridSearchCV(pipe, {f"model__{p}": v for p, v in grid.items()},
                                  scoring="r2", cv=inner, n_jobs=1)
                gs.fit(Xtr, ytr)
                inner_r2[n] = float(gs.best_score_)
                best_params[n] = {p.replace("model__", ""): v
                                  for p, v in gs.best_params_.items()}
                fitted[n] = gs.best_estimator_
            else:
                inner_r2[n] = float(np.mean(
                    cross_val_score(pipe, Xtr, ytr, scoring="r2",
                                    cv=inner, n_jobs=1)))
                best_params[n] = {}
                fitted[n] = clone(pipe).fit(Xtr, ytr)

        ranked = sorted(names, key=lambda n: inner_r2[n], reverse=True)
        top = ranked[:TOP_K]

        avg = np.mean([fitted[n].predict(Xte) for n in top], axis=0)
        r2 = float(r2_score(yte, avg))
        rmse = float(np.sqrt(mean_squared_error(yte, avg)))
        mae = float(mean_absolute_error(yte, avg))
        r2s.append(r2); rmses.append(rmse); maes.append(mae)

        folds.append({
            "fold": k, "n_train": int(len(tr)), "n_test": int(len(te)),
            "selected_members": top,
            "matches_fixed_faae": set(top) == set(FIXED_FAAE_MEMBERS),
            "inner_cv_r2": inner_r2, "inner_cv_ranking": ranked,
            "selected_hyperparameters": {n: best_params[n] for n in top},
            "R2": r2, "RMSE": rmse, "MAE": mae,
        })
        if verbose:
            print(f"    selected: {', '.join(top)}")
            print(f"    outer-test R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f}\n",
                  flush=True)

    freq = {n: sum(n in f["selected_members"] for f in folds) for n in names}
    freq = dict(sorted(freq.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
        "folds_matching_fixed_faae":
            f"{sum(f['matches_fixed_faae'] for f in folds)}/{config.N_SPLITS}",
        "member_selection_frequency": freq,
        "per_fold": folds,
    }


def main():
    set_seed()
    X, y, feats, target, _ = load_dataset(drop_features=config.DROP_FEATURES)
    y = y.ravel()

    print(f"Nested member selection | {len(feats)} features | "
          f"target: {target}", flush=True)
    print(f"N={len(X)} | outer {config.N_SPLITS}-fold + inner 3-fold "
          f"GridSearchCV | candidates=8 | TOP_K={TOP_K} | seed={config.SEED}\n",
          flush=True)

    t0 = time.time()
    res = nested_member_selection(X, y)
    elapsed = time.time() - t0

    out = {
        "analysis": "FAAE with member selection inside the outer CV",
        "protocol": {
            "outer": f"KFold({config.N_SPLITS}, shuffle=True, random_state={config.SEED})",
            "inner": f"KFold(3, shuffle=True, random_state={config.SEED}) GridSearchCV",
            "ranking_signal": "inner-CV mean R2",
            "refit": "best params refit on the full outer-training fold",
            "scaling": "StandardScaler inside Pipeline",
            "seed": config.SEED,
        },
        "target": target, "n_samples": int(len(X)), "n_features": len(feats),
        "features": feats, "top_k": TOP_K,
        "fixed_faae_members": FIXED_FAAE_MEMBERS,
        "runtime_seconds": round(elapsed, 1),
        **res,
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    base = os.path.join(config.RESULTS_DIR, "nested_member_selection")

    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["fold", "R2", "RMSE", "MAE", "selected_members",
                    "matches_fixed_faae"])
        for fd in res["per_fold"]:
            w.writerow([fd["fold"], f"{fd['R2']:.4f}", f"{fd['RMSE']:.4f}",
                        f"{fd['MAE']:.4f}", " + ".join(fd["selected_members"]),
                        fd["matches_fixed_faae"]])
        w.writerow([])
        w.writerow(["MEAN", f"{res['R2_mean']:.4f}", f"{res['RMSE_mean']:.4f}",
                    f"{res['MAE_mean']:.4f}", "", ""])
        w.writerow(["SD", f"{res['R2_std']:.4f}", f"{res['RMSE_std']:.4f}",
                    f"{res['MAE_std']:.4f}", "", ""])
        w.writerow([])
        w.writerow(["member", "times_selected", "", "", "", ""])
        for n, c in res["member_selection_frequency"].items():
            w.writerow([n, f"{c}/{config.N_SPLITS}", "", "", "", ""])


    print("=== MEMBERS SELECTED PER FOLD ===")
    for fd in res["per_fold"]:
        print(f"  Fold {fd['fold']}: {', '.join(fd['selected_members'])}")
    print(f"\nFolds matching the fixed composition: "
          f"{res['folds_matching_fixed_faae']}")
    print("\n=== MEMBER SELECTION FREQUENCY ===")
    for n, c in res["member_selection_frequency"].items():
        if c:
            print(f"  {n:<20} {c}/{config.N_SPLITS}")
    print("\n=== NESTED-SELECTION FAAE ===")
    print(f"  R2   : {res['R2_mean']:.4f} +/- {res['R2_std']:.4f}")
    print(f"  RMSE : {res['RMSE_mean']:.4f} +/- {res['RMSE_std']:.4f}")
    print(f"  MAE  : {res['MAE_mean']:.4f} +/- {res['MAE_std']:.4f}")
    print(f"\nRuntime: {elapsed/60:.1f} min")
    print(f"  --> Saved: {base}.json / .csv")


if __name__ == "__main__":
    main()
