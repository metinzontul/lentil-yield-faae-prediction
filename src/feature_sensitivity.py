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
from nested_cv import nested_cv_estimator
from faae_model import build_faae
from nested_member_selection import nested_member_selection

LEAKY = config.LEAKY_FEATURE
BIO_YIELD = "Biological Yield (g)"
N_PODS = "Number of Pods per Plant"

SCENARIOS = [
    ("Full model", "7 predictors", [LEAKY]),
    ("Sensitivity 1", "6 predictors, Biological Yield excluded",
     [LEAKY, BIO_YIELD]),
    ("Sensitivity 2",
     "5 predictors, Biological Yield and Number of Pods per Plant excluded",
     [LEAKY, BIO_YIELD, N_PODS]),
]


def main():
    rows = []
    t0 = time.time()

    print("Feature sensitivity | FAAE", flush=True)
    print(f"Outer {config.N_SPLITS}-fold + inner 3-fold GridSearchCV | "
          f"seed={config.SEED}\n", flush=True)

    for label, desc, drops in SCENARIOS:
        print(f"=== {label} ({desc}) ===", flush=True)

        # -------- Protocol A: fixed composition --------
        set_seed()
        X, y, feats, target, _ = load_dataset(drop_features=drops)
        print(f"  predictors ({len(feats)}): {', '.join(feats)}", flush=True)
        print("  [A] fixed composition ...", flush=True)
        res_a = nested_cv_estimator(f"FAAE-fixed - {label}",
                                    build_faae(tuned=True), {}, X, y)

        # -------- Protocol B: nested member selection --------
        set_seed()
        X, y, feats, target, _ = load_dataset(drop_features=drops)
        y = y.ravel()
        print("  [B] nested member selection ...", flush=True)
        res_b = nested_member_selection(X, y, verbose=False)

        for proto, r in (("A - fixed composition", res_a),
                         ("B - nested member selection", res_b)):
            print(f"    {proto[:1]}: R2={r['R2_mean']:.4f}+/-{r['R2_std']:.4f}  "
                  f"RMSE={r['RMSE_mean']:.4f}+/-{r['RMSE_std']:.4f}  "
                  f"MAE={r['MAE_mean']:.4f}+/-{r['MAE_std']:.4f}", flush=True)
            rows.append({
                "protocol": proto, "scenario": label, "predictors_used": desc,
                "n_predictors": len(feats), "predictors": feats,
                "excluded": drops, "target": target, "n_samples": int(len(X)),
                "R2_mean": r["R2_mean"], "R2_std": r["R2_std"],
                "RMSE_mean": r["RMSE_mean"], "RMSE_std": r["RMSE_std"],
                "MAE_mean": r["MAE_mean"], "MAE_std": r["MAE_std"],
                "member_selection_frequency": r.get("member_selection_frequency"),
            })
        print(flush=True)

    elapsed = time.time() - t0

    out = {
        "analysis": "Feature sensitivity (FAAE, 3 scenarios x 2 protocols)",
        "protocol_note": ("Only the predictor set differs across scenarios; "
                          "protocols A and B are reported separately."),
        "protocol": {
            "outer": f"KFold({config.N_SPLITS}, shuffle=True, random_state={config.SEED})",
            "inner": f"KFold(3, shuffle=True, random_state={config.SEED}) GridSearchCV",
            "scaling": "StandardScaler inside Pipeline", "seed": config.SEED,
        },
        "runtime_seconds": round(elapsed, 1),
        "scenarios": rows,
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    base = os.path.join(config.RESULTS_DIR, "feature_sensitivity")

    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Protocol", "Scenario", "Predictors Used", "n_predictors",
                    "R2_mean", "R2_std", "RMSE_mean", "RMSE_std",
                    "MAE_mean", "MAE_std"])
        for r in rows:
            w.writerow([r["protocol"], r["scenario"], r["predictors_used"],
                        r["n_predictors"],
                        f"{r['R2_mean']:.4f}", f"{r['R2_std']:.4f}",
                        f"{r['RMSE_mean']:.4f}", f"{r['RMSE_std']:.4f}",
                        f"{r['MAE_mean']:.4f}", f"{r['MAE_std']:.4f}"])

    print("=== SUMMARY ===")
    hdr = f"{'Protocol':<10}{'Scenario':<16}{'#':>3}  {'R2':>17} {'RMSE':>17} {'MAE':>17}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['protocol'][:1]:<10}{r['scenario']:<16}{r['n_predictors']:>3}  "
              f"{r['R2_mean']:.4f} +/- {r['R2_std']:.4f}  "
              f"{r['RMSE_mean']:.4f} +/- {r['RMSE_std']:.4f}  "
              f"{r['MAE_mean']:.4f} +/- {r['MAE_std']:.4f}")
    print(f"\nRuntime: {elapsed/60:.1f} min")
    print(f"  --> Saved: {base}.json / .csv")


if __name__ == "__main__":
    main()
