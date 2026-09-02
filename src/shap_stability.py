# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore")

import os
import sys

# Thread caps must be set before numeric libraries are imported.
if len(sys.argv) > 2:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = sys.argv[2]

import csv
import json
import time
import itertools

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import shap
from scipy.stats import spearmanr

import config
from data_utils import set_seed, load_dataset
from faae_model import build_faae_members
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 42, 123]
N_BACKGROUND = 15
N_EXPLAIN = 40
N_SAMPLES = 120

INTERPRETATION_LIMIT = (
    "SHAP values come from the FAAE model fitted to the complete dataset and "
    "are not out-of-fold estimates of feature importance.")


def _seed_path(seed):
    return os.path.join(config.RESULTS_DIR, f"_shap_stability_seed{seed}.json")


def run_one_seed(seed):
    """Compute SHAP for a single seed and save the intermediate JSON."""
    set_seed()
    X, y, feats, target, _ = load_dataset(drop_features=config.DROP_FEATURES)
    y = y.ravel()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"[seed={seed}] {len(feats)} features | background={N_BACKGROUND} "
          f"| explained={N_EXPLAIN} | nsamples={N_SAMPLES}", flush=True)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    members = build_faae_members()
    for m in members:
        m.fit(Xs, y)

    def predict_fn(data):
        return np.mean([m.predict(data) for m in members], axis=0)

    bg = shap.kmeans(Xs, min(N_BACKGROUND, len(Xs)))
    explainer = shap.KernelExplainer(predict_fn, bg)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(Xs), size=min(N_EXPLAIN, len(Xs)),
                             replace=False))
    np.random.seed(seed)                 # KernelExplainer coalition sampling

    t0 = time.time()
    sv = np.asarray(explainer.shap_values(Xs[idx], nsamples=N_SAMPLES))
    elapsed = time.time() - t0

    mean_abs = np.abs(sv).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    rel = mean_abs / total * 100.0
    order = np.argsort(mean_abs)[::-1]
    rank = np.empty(len(feats), dtype=int)
    rank[order] = np.arange(1, len(feats) + 1)

    out = {"seed": seed, "features": feats, "target": target,
           "explained_indices": [int(i) for i in idx],
           "mean_abs_shap": mean_abs.tolist(),
           "relative_pct": rel.tolist(), "rank": rank.tolist(),
           "n_background": N_BACKGROUND, "n_explain": N_EXPLAIN,
           "nsamples": N_SAMPLES, "shap_version": shap.__version__,
           "runtime_seconds": round(elapsed, 1)}
    with open(_seed_path(seed), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[seed={seed}] done ({elapsed/60:.1f} min) -> {_seed_path(seed)}",
          flush=True)
    for i in order:
        print(f"    {feats[i]:<28} {rel[i]:>5.1f}%  rank={rank[i]}", flush=True)
    return out


def build_report():
    """Combine the three per-seed results into the final report."""
    runs = []
    for s in SEEDS:
        p = _seed_path(s)
        if not os.path.exists(p):
            print(f"MISSING: {p}\n-> run: python shap_stability.py {s}")
            sys.exit(1)
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))

    feats = runs[0]["features"]
    for r in runs:
        assert r["features"] == feats
        assert r["n_background"] == runs[0]["n_background"]
        assert r["nsamples"] == runs[0]["nsamples"]
        assert r["n_explain"] == runs[0]["n_explain"]

    MA = np.array([r["mean_abs_shap"] for r in runs])
    RL = np.array([r["relative_pct"] for r in runs])
    RK = np.array([r["rank"] for r in runs])

    per_feature = []
    for j, f in enumerate(feats):
        per_feature.append({
            "feature": f,
            "mean_abs_shap_per_seed": {str(s): float(MA[i, j])
                                       for i, s in enumerate(SEEDS)},
            "relative_pct_per_seed": {str(s): float(RL[i, j])
                                      for i, s in enumerate(SEEDS)},
            "relative_pct_mean": float(RL[:, j].mean()),
            "relative_pct_sd": float(RL[:, j].std(ddof=1)),
            "rank_per_seed": {str(s): int(RK[i, j]) for i, s in enumerate(SEEDS)},
            "rank_mean": float(RK[:, j].mean()),
            "rank_min": int(RK[:, j].min()), "rank_max": int(RK[:, j].max()),
            "rank_range": int(RK[:, j].max() - RK[:, j].min()),
        })
    per_feature.sort(key=lambda d: -d["relative_pct_mean"])

    pairs = []
    for a, b in itertools.combinations(range(len(SEEDS)), 2):
        rho, p = spearmanr(RK[a], RK[b])
        pairs.append({"seed_a": SEEDS[a], "seed_b": SEEDS[b],
                      "spearman_rho": float(rho), "p_value": float(p)})
    rhos = np.array([p["spearman_rho"] for p in pairs])

    out = {
        "analysis": "SHAP feature-ranking stability across three seeds",
        "design": {
            "seeds": SEEDS,
            "fitted_FAAE_model": "trained once on the complete dataset",
            "background_set": (f"shap.kmeans(Xs, {runs[0]['n_background']}); "
                               f"shap {runs[0]['shap_version']} hardcodes "
                               "KMeans(random_state=0), so the background is "
                               "identical in every run"),
            "n_explain": runs[0]["n_explain"], "nsamples": runs[0]["nsamples"],
            "varied": ("explained-sample subset and KernelExplainer coalition "
                       "sampling"),
            "interpretation_limit": INTERPRETATION_LIMIT,
        },
        "features": feats, "target": runs[0].get("target"),
        "shap_version": runs[0]["shap_version"],
        "per_feature": per_feature,
        "per_run": [{"seed": r["seed"], "runtime_seconds": r["runtime_seconds"],
                     "explained_indices": r["explained_indices"],
                     "relative_pct": r["relative_pct"], "rank": r["rank"],
                     "mean_abs_shap": r["mean_abs_shap"]} for r in runs],
        "spearman_pairwise": pairs,
        "spearman_mean": float(rhos.mean()),
        "spearman_min": float(rhos.min()),
        "spearman_max": float(rhos.max()),
    }

    base = os.path.join(config.RESULTS_DIR, "shap_stability")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Feature"] +
                   [f"meanAbsSHAP_seed{s}" for s in SEEDS] +
                   [f"relPct_seed{s}" for s in SEEDS] +
                   ["relPct_mean", "relPct_SD"] +
                   [f"rank_seed{s}" for s in SEEDS] +
                   ["rank_mean", "rank_min", "rank_max", "rank_range"])
        for d in per_feature:
            w.writerow([d["feature"]] +
                       [f"{d['mean_abs_shap_per_seed'][str(s)]:.4f}" for s in SEEDS] +
                       [f"{d['relative_pct_per_seed'][str(s)]:.1f}" for s in SEEDS] +
                       [f"{d['relative_pct_mean']:.1f}",
                        f"{d['relative_pct_sd']:.1f}"] +
                       [d["rank_per_seed"][str(s)] for s in SEEDS] +
                       [f"{d['rank_mean']:.2f}", d["rank_min"], d["rank_max"],
                        d["rank_range"]])
        w.writerow([])
        w.writerow(["Spearman pair", "rho", "p_value"] + [""] * 13)
        for p in pairs:
            w.writerow([f"seed{p['seed_a']} vs seed{p['seed_b']}",
                        f"{p['spearman_rho']:.4f}", f"{p['p_value']:.4f}"] + [""] * 13)
        w.writerow(["MEAN", f"{rhos.mean():.4f}", ""] + [""] * 13)
        w.writerow(["MIN", f"{rhos.min():.4f}", ""] + [""] * 13)


    print("=== SHAP RANKING STABILITY ===")
    hdr = (f"{'Feature':<28}" + "".join(f"{'s'+str(s):>9}" for s in SEEDS) +
           f"{'mean±SD':>16}{'rank':>7}{'range':>8}")
    print(hdr); print("-" * len(hdr))
    for d in per_feature:
        rl = "".join(f"{d['relative_pct_per_seed'][str(s)]:>9.1f}" for s in SEEDS)
        print(f"{d['feature']:<28}{rl}"
              f"{d['relative_pct_mean']:>9.1f}±{d['relative_pct_sd']:<6.1f}"
              f"{d['rank_mean']:>7.2f}"
              f"{str(d['rank_min'])+'-'+str(d['rank_max']):>8}")
    print("\n=== PAIRWISE SPEARMAN ===")
    for p in pairs:
        print(f"  seed{p['seed_a']:<4} vs seed{p['seed_b']:<4} "
              f"rho={p['spearman_rho']:>7.4f}  p={p['p_value']:.4f}")
    print(f"\n  mean = {rhos.mean():.4f}   min = {rhos.min():.4f}")
    print(f"\n  --> Saved: {base}.json / .csv")


def main():
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        if seed not in SEEDS:
            print(f"warning: seed {seed} is not in {SEEDS}; "
                  "the combined report expects those three.")
        run_one_seed(seed)
    else:
        missing = [s for s in SEEDS if not os.path.exists(_seed_path(s))]
        if missing:
            print(f"Computing missing seeds sequentially: {missing}")
            print("(faster: run each seed in its own process, see the docstring)\n")
            for s in missing:
                run_one_seed(s)
            print()
        build_report()


if __name__ == "__main__":
    main()
