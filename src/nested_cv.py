# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import config


def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return (r2_score(y_true, y_pred),
            float(np.sqrt(mse)),
            mean_absolute_error(y_true, y_pred))


def _agg(r2s, rmses, maes, name, extra=None):
    out = {
        "name": name,
        "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
    }
    if extra:
        out.update(extra)
    return out


def nested_cv_estimator(name, estimator, param_grid, X, y,
                        outer_splits=None, inner_splits=3):

    import sys
    from tqdm import tqdm

    outer_splits = outer_splits or config.N_SPLITS
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=config.SEED)
    inner = KFold(n_splits=inner_splits, shuffle=True, random_state=config.SEED)
    do_tune = bool(param_grid)

    r2s, rmses, maes, chosen = [], [], [], []
    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    grid = {f"model__{k}": v for k, v in param_grid.items()}

    with tqdm(total=outer_splits, desc=f"  {name}",
              unit="fold", file=sys.stdout, dynamic_ncols=True, ascii=True) as pbar:
        for k, (tr, te) in enumerate(outer.split(X), 1):
            if do_tune:
                est = GridSearchCV(pipe, grid, scoring="r2", cv=inner, n_jobs=1)
                est.fit(X[tr], y[tr].ravel())
                pred = est.predict(X[te]).reshape(-1, 1)
                chosen.append(est.best_params_)
            else:
                from sklearn.base import clone
                est = clone(pipe)
                est.fit(X[tr], y[tr].ravel())
                pred = est.predict(X[te]).reshape(-1, 1)
                chosen.append({})
            r2, rmse, mae = _metrics(y[te], pred)
            r2s.append(r2); rmses.append(rmse); maes.append(mae)
            pbar.set_postfix_str(f"fold {k}/{outer_splits} R2={r2:.4f}")
            pbar.update(1)

    return _agg(r2s, rmses, maes, name,
                {"kind": "nested", "tuned": do_tune, "outer_splits": outer_splits,
                 "inner_splits": inner_splits, "chosen_params_per_fold": chosen})


nested_cv_sklearn = nested_cv_estimator
