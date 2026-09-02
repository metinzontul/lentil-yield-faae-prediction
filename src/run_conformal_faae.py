# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore")

import os
import sys
import csv
import json
import math
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import config
from data_utils import set_seed, load_dataset
from faae_model import build_faae_members as make_members

COVERAGE_LEVELS = [0.80, 0.90, 0.95]
CALIB_FRACTION = 0.30
PLOT_COVERAGE = 0.90
CAL_GRID = np.linspace(0.50, 0.98, 25)


def faae_fit_predict(Xtr, ytr, Xpred):
    """FAAE = equal-weight average of member predictions."""
    preds = []
    for m in make_members():
        mm = clone(m); mm.fit(Xtr, ytr); preds.append(mm.predict(Xpred))
    return np.mean(preds, axis=0)


def cq(scores, coverage):
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def ms(a):
    return float(np.mean(a)), float(np.std(a))


def main():
    set_seed()
    X, y, feats, target, df = load_dataset(drop_features=config.DROP_FEATURES)
    y = y.ravel()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"Conformal + FAAE | {len(feats)} features (leakage-free) | target: {target}\n",
          flush=True)

    from tqdm import tqdm

    # ---------- Part A: repeated 5x3 coverage/width ----------
    rkf = RepeatedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS,
                        random_state=config.SEED)
    splits = list(rkf.split(X))
    r2s, rmses, maes = [], [], []
    cov = {"std": {c: [] for c in COVERAGE_LEVELS}, "norm": {c: [] for c in COVERAGE_LEVELS}}
    wid = {"std": {c: [] for c in COVERAGE_LEVELS}, "norm": {c: [] for c in COVERAGE_LEVELS}}

    pbar = tqdm(total=len(splits), desc="  Conformal+FAAE", unit="fold",
                file=sys.stdout, dynamic_ncols=True, ascii=True)
    for tr, te in splits:
        idx_pt, idx_cal = train_test_split(tr, test_size=CALIB_FRACTION,
                                           random_state=config.SEED)
        sx = StandardScaler().fit(X[idx_pt])
        Xpt, Xcal, Xte = sx.transform(X[idx_pt]), sx.transform(X[idx_cal]), sx.transform(X[te])

        # Fit FAAE members once (proper-train), reuse for all predictions
        fitted = [clone(m).fit(Xpt, y[idx_pt]) for m in make_members()]
        def faae(Z):
            return np.mean([m.predict(Z) for m in fitted], axis=0)

        pred_cal = faae(Xcal)
        abs_cal = np.abs(y[idx_cal] - pred_cal)
        # sigma model: learns FAAE's absolute residuals (local difficulty)
        resid_pt = np.abs(y[idx_pt] - faae(Xpt))
        sigma = RandomForestRegressor(n_estimators=200, random_state=config.SEED, n_jobs=-1)
        sigma.fit(Xpt, resid_pt)
        beta = 0.1 * float(np.mean(resid_pt)) + 1e-6
        sig_cal = sigma.predict(Xcal) + beta
        scores_norm = abs_cal / sig_cal

        pred_te = faae(Xte)
        r2s.append(r2_score(y[te], pred_te))
        rmses.append(float(np.sqrt(mean_squared_error(y[te], pred_te))))
        maes.append(mean_absolute_error(y[te], pred_te))
        sig_te = sigma.predict(Xte) + beta
        for c in COVERAGE_LEVELS:
            q = cq(abs_cal, c)
            cov["std"][c].append(float(np.mean((y[te] >= pred_te - q) & (y[te] <= pred_te + q))))
            wid["std"][c].append(float(2 * q))
            qn = cq(scores_norm, c); half = qn * sig_te
            cov["norm"][c].append(float(np.mean((y[te] >= pred_te - half) & (y[te] <= pred_te + half))))
            wid["norm"][c].append(float(np.mean(2 * half)))
        pbar.update(1)
    pbar.close()

    r2_m, r2_s = ms(r2s); rmse_m, rmse_s = ms(rmses); mae_m, mae_s = ms(maes)
    ystd = float(y.std())
    print("\n=== POINT ESTIMATE (FAAE, proper-train) ===")
    print(f"  R2={r2_m:.4f}+/-{r2_s:.4f}  RMSE={rmse_m:.4f}  MAE={mae_m:.4f}")

    rows = []
    for key, label in [("std", "standard"), ("norm", "normalized")]:
        print(f"\n{label}:")
        for c in COVERAGE_LEVELS:
            cm, cs = ms(cov[key][c]); wm, ws = ms(wid[key][c])
            print(f"  {int(c*100)}%: coverage={cm*100:.1f}+/-{cs*100:.1f}  width={wm:.2f}")
            rows.append({"method": label, "target_coverage": c,
                         "empirical_coverage_mean": cm, "empirical_coverage_std": cs,
                         "width_mean": wm, "width_std": ws, "width_over_ystd": wm / ystd})

    out = {"method": "Split Conformal + FAAE (standard + normalized)",
           "features": feats, "n_features": len(feats), "target": target,
           "dropped": config.DROP_FEATURES, "n_splits": config.N_SPLITS,
           "n_repeats": config.N_REPEATS, "calib_fraction": CALIB_FRACTION,
           "point": {"R2_mean": r2_m, "R2_std": r2_s, "RMSE_mean": rmse_m,
                     "RMSE_std": rmse_s, "MAE_mean": mae_m, "MAE_std": mae_s},
           "conformal": rows}
    base = os.path.join(config.RESULTS_DIR, "conformal_faae")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["method", "target_coverage", "empirical_coverage_mean",
                    "empirical_coverage_std", "width_mean", "width_std", "width_over_ystd"])
        for r in rows:
            w.writerow([r["method"], f"{r['target_coverage']:.2f}",
                        f"{r['empirical_coverage_mean']:.4f}", f"{r['empirical_coverage_std']:.4f}",
                        f"{r['width_mean']:.4f}", f"{r['width_std']:.4f}", f"{r['width_over_ystd']:.4f}"])

    # ---------- Part B: figures (single 5-fold pass) ----------
    print("\nGenerating figures (5-fold)...", flush=True)
    kf = list(KFold(n_splits=config.N_SPLITS, shuffle=True,
                    random_state=config.SEED).split(X))
    folds = []
    pb2 = tqdm(total=len(kf), desc="  Fig folds", unit="fold",
               file=sys.stdout, dynamic_ncols=True, ascii=True)
    for tr, te in kf:
        idx_pt, idx_cal = train_test_split(tr, test_size=CALIB_FRACTION,
                                           random_state=config.SEED)
        sx = StandardScaler().fit(X[idx_pt])
        Xpt, Xcal, Xte = sx.transform(X[idx_pt]), sx.transform(X[idx_cal]), sx.transform(X[te])
        fitted = [clone(m).fit(Xpt, y[idx_pt]) for m in make_members()]
        def faae(Z):
            return np.mean([m.predict(Z) for m in fitted], axis=0)
        abs_cal = np.abs(y[idx_cal] - faae(Xcal))
        resid_pt = np.abs(y[idx_pt] - faae(Xpt))
        sigma = RandomForestRegressor(n_estimators=200, random_state=config.SEED, n_jobs=-1)
        sigma.fit(Xpt, resid_pt)
        beta = 0.1 * float(np.mean(resid_pt)) + 1e-6
        folds.append({"y": y[te], "pred": faae(Xte), "sig_te": sigma.predict(Xte) + beta,
                      "abs_cal": abs_cal, "norm_cal": abs_cal / (sigma.predict(Xcal) + beta)})
        pb2.update(1)
    pb2.close()

    # Calibration curve
    emp_std, emp_norm = [], []
    for c in CAL_GRID:
        s, n, tot = 0, 0, 0
        for f in folds:
            q = cq(f["abs_cal"], c)
            s += np.sum((f["y"] >= f["pred"] - q) & (f["y"] <= f["pred"] + q))
            qn = cq(f["norm_cal"], c); half = qn * f["sig_te"]
            n += np.sum((f["y"] >= f["pred"] - half) & (f["y"] <= f["pred"] + half))
            tot += len(f["y"])
        emp_std.append(s / tot); emp_norm.append(n / tot)
    plt.figure(figsize=(6, 6))
    plt.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=1.2, label="Ideal (y = x)")
    plt.plot(CAL_GRID, emp_std, "o-", ms=4, label="Standard conformal")
    plt.plot(CAL_GRID, emp_norm, "s-", ms=4, label="Normalized (locally-adaptive)")
    plt.xlabel("Target coverage (1 - α)"); plt.ylabel("Empirical coverage")
    plt.title("Conformal Calibration — FAAE (leakage-free, 7 features)")
    plt.legend(loc="upper left"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(base + "_calibration.png", dpi=300); plt.close()

    # Interval band (90%)
    yt, pr, lo_s, hi_s, lo_n, hi_n = [], [], [], [], [], []
    for f in folds:
        q = cq(f["abs_cal"], PLOT_COVERAGE); qn = cq(f["norm_cal"], PLOT_COVERAGE)
        half = qn * f["sig_te"]
        yt.append(f["y"]); pr.append(f["pred"])
        lo_s.append(f["pred"] - q); hi_s.append(f["pred"] + q)
        lo_n.append(f["pred"] - half); hi_n.append(f["pred"] + half)
    yt = np.concatenate(yt); pr = np.concatenate(pr)
    lo_s, hi_s = np.concatenate(lo_s), np.concatenate(hi_s)
    lo_n, hi_n = np.concatenate(lo_n), np.concatenate(hi_n)
    order = np.argsort(yt); xr = np.arange(len(yt))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, lo, hi, ttl in [
        (axes[0], lo_s[order], hi_s[order], "Standard conformal (constant width)"),
        (axes[1], lo_n[order], hi_n[order], "Normalized conformal (adaptive width)"),
    ]:
        ax.fill_between(xr, lo, hi, alpha=0.25, color="tab:blue",
                        label=f"{int(PLOT_COVERAGE*100)}% prediction interval")
        ax.plot(xr, pr[order], color="tab:blue", lw=1.0, label="Predicted")
        ax.scatter(xr, yt[order], s=14, color="tab:red", zorder=3, label="True")
        c = np.mean((yt[order] >= lo) & (yt[order] <= hi))
        ax.set_title(f"{ttl}\nempirical coverage = {c*100:.1f}%")
        ax.set_xlabel("Test samples (sorted by true yield)")
        ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"{target}")
    plt.suptitle("Out-of-fold 90% Conformal Prediction Intervals — FAAE", y=1.02)
    plt.tight_layout()
    plt.savefig(base + "_intervals.png", dpi=300, bbox_inches="tight"); plt.close()

    print(f"\nSaved: {base}.json/.csv, {base}_calibration.png, {base}_intervals.png")


if __name__ == "__main__":
    main()
