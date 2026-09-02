# Lentil Seed Yield Prediction — Code and Data Supplement

Code and data to reproduce the yield-prediction experiments: baseline
models, the proposed FAAE ensemble, and the statistical, SHAP, LOCO and
robustness analyses reported in the article.

## Dataset

`data/Lentil.csv` — 264 samples, 9 columns. Target: `Total Seed Yield (kg/da)`.

The file contains **no missing values**, so no records were removed and no
imputation was performed (`data_utils.load_dataset` applies `dropna()` as a
safeguard, which is a no-op here: 264 rows before and after).

`Seed Yield per Plant (g)` is excluded from all experiments (near-perfect
proxy of the target, r ≈ 0.99). Seven features are used.

The dataset contains only phenotypic measurements; there is **no pedigree,
family, or population-structure column**, so group-aware cross-validation
(e.g. `GroupKFold`) is not possible from these data.

## Requirements

```
pip install -r requirements.txt
```

Pinned versions reproduce the reported results (Python 3.13, `tabpfn` 7.1.1,
`shap` 0.51.0, `scikit-learn` 1.7.2). TabPFN needs PyTorch; the CPU build is
sufficient, since every script instantiates `TabPFNRegressor(device="cpu")`.

## Usage

Run from the `src/` directory.

### Main experiments

```
python baseline_models.py        # 9 baseline models (Ridge, RF, XGBoost, ...)
python run_faae.py               # proposed FAAE model
python statistical_analysis.py   # Friedman / Nemenyi / Wilcoxon significance tests
python run_conformal_faae.py     # uncertainty quantification for FAAE
python shap_analysis.py          # SHAP feature importance for FAAE
python loco_analysis.py          # leave-one-covariate-out analysis for FAAE
```

### Robustness analyses

```
python nested_member_selection.py   # member selection moved inside the outer CV
python ablation_and_pooled_oof.py   # ablation, residual correlations, pooled OOF
python feature_sensitivity.py       # 7 / 6 / 5 predictor scenarios, both protocols
python loco_foldwise.py             # per-fold LOCO delta R2 (mean +/- SD)
python shap_stability.py            # SHAP ranking stability across 3 seeds
```

`shap_stability.py` is slow (Kernel SHAP over TabPFN, ~1–2 min per explained
observation on CPU). Run one seed per process to parallelise:

```
python shap_stability.py 0 &
python shap_stability.py 42 &
python shap_stability.py 123 &
python shap_stability.py            # combine into the final report
```

Shared modules (not run directly): `config.py`, `data_utils.py`,
`nested_cv.py`, `faae_model.py` (the FAAE definition), `candidates.py` (the
eight-candidate pool and search grids, identical to `baseline_models.py`).

All randomness is controlled by a fixed seed (`config.SEED = 42`).

## Two FAAE evaluation protocols

FAAE is reported under two protocols, kept separate throughout:

- **A — fixed composition.** TabPFN + Extra Trees + CatBoost + XGBoost, the
  top four learners in the nested-CV comparison, with members tuned per outer
  fold. This is the proposed model (`run_faae.py`).
- **B — nested member selection.** Members are re-selected in each outer fold
  from the eight candidates by inner-CV R² (`nested_member_selection.py`),
  which measures the selection optimism in protocol A (ΔR² = −0.0068).

Analyses that are not performance estimates — conformal, SHAP, LOCO and the
statistical tests — use the fixed-hyperparameter members
(`build_faae_members()`), where inner tuning does not apply.

## Outputs

Results are not version-controlled; every script writes to `results/`, which
is created on first run. Each analysis produces a `.json` file with the full
record and a `.csv` file with the tabular summary.

| File | Analysis |
|---|---|
| `nested_member_selection.*` | Per-fold selected members, inner-CV rankings, selected hyperparameters |
| `faae_ablation.*` | TabPFN alone / classical / FAAE / all-model, plus OOF residual and prediction correlations |
| `pooled_oof.*` | Pooled out-of-fold metrics (metrics computed once on concatenated predictions) |
| `pooled_oof_predictions.csv` | Raw out-of-fold predictions for all 264 observations |
| `feature_sensitivity.*` | 7 / 6 / 5 predictor scenarios under both protocols |
| `loco_foldwise.*` | Per-fold LOCO ΔR², paired within folds |
| `shap_stability.*` | SHAP contribution %, rank and Spearman ρ across seeds 0, 42, 123 |

## Notes on interpretation

**SHAP.** SHAP values come from the FAAE model fitted to the complete
dataset and are not out-of-fold estimates of feature importance;
`loco_foldwise.py` provides a cross-validated measure.

**SHAP background set.** `shap.kmeans()` hardcodes `KMeans(random_state=0)`
and takes no seed argument, so the background is identical in every run of
`shap_stability.py`; the seed varies the explained-sample subset.

**LOCO.** ΔR² is paired within each fold, so the standard deviation across
folds is interpretable. Small differences among lower-ranked features fall
within that spread and do not establish a rank ordering.

**Pooled vs. fold-averaged metrics.** Pooled metrics concatenate the
outer-test predictions and compute each metric once. They run systematically
higher than fold-averaged metrics and the two are not interchangeable.

**Statistical tests.** `statistical_analysis.py` evaluates every model on
the same repeated 5×3 folds with fixed, untuned hyperparameters, giving a
paired and equally conditioned comparison. Those values are set in the script
and are independent of the fold-specific hyperparameters chosen under nested
CV.

## Citation

> *Citation will be added upon publication.*

## License

Released under the MIT License — see `LICENSE`.
