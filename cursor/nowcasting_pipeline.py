"""
Unified nowcasting pipeline for German unemployment using Google Trends.

This module centralises all helper functions and configuration needed by the
refactored notebook. It keeps the notebook readable by moving most logic into
importable functions while remaining lightweight (pandas / numpy / statsmodels).
"""

from __future__ import annotations

import dataclasses
import itertools
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclasses.dataclass
class ModelConfig:
    # data + paths
    data_dir: str = "./new_data"
    unemployment_file: str = "./unemployment_rate_germany_DE_only.csv"
    output_dir: str = "./outputs"
    figure_dpi: int = 150

    # sample windows
    sample_start: str = "2011-05-01"
    sample_end: str = "2023-01-31"
    eval_start: str = "2019-01-31"
    eval_end: Optional[str] = None

    window_mode: str = "expanding"  # or "rolling"
    rolling_window_months: Optional[int] = None

    target_transform: str = "diff"  # "diff" or "level"
    seasonal_dummies: bool = True

    # AR lag selection
    ar_lag_selection: Dict[str, object] = dataclasses.field(
        default_factory=lambda: {"criterion": "AIC", "max_lags": 6}
    )

    # ARX lag selection (grid over p and q)
    arx_lag_selection: Dict[str, object] = dataclasses.field(
        default_factory=lambda: {
            "criterion": "AIC",
            "max_ar_lags": 4,
            "max_gt_lags": 3,
            "search": "grid",  # "grid" or "sequential"
        }
    )

    # MIDAS
    midas_k_candidates: Sequence[int] = (4, 6, 8, 10, 12)
    midas_ic: str = "AIC"
    include_ar_term: bool = True

    # Feature options
    keywords: Sequence[str] = ("indeed", "stepstone", "jobbörse", "arbeitsamt", "bewerbung")
    keyword_file_map: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "indeed": "Indeed",
            "stepstone": "Stepstone",
            "jobbörse": "Jobbörse",
            "arbeitsamt": "Arbeitsamt",
            "bewerbung": "Bewerbung",
        }
    )
    biweekly_schemes: Sequence[str] = ("W12", "W34prev")

    # Model toggles
    enabled_models: Sequence[str] = ("AR", "ARX", "MIDAS", "MIDAS_restricted")
    include_engle_granger: bool = False


def ensure_output_dirs(cfg: ModelConfig) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def set_seed(seed: int = 7) -> None:
    np.random.seed(seed)


def _prep_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna(subset=["Date", value_col]).set_index("Date").sort_index()


def _median_ratio(a: pd.Series, b: pd.Series) -> float:
    idx = a.index.intersection(b.index)
    x, y = a.loc[idx], b.loc[idx]
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return 1.0
    return np.median((x[mask] / y[mask]).values)


def stitch_three_windows(w1: pd.DataFrame, w2: pd.DataFrame, w3: pd.DataFrame, value_col: str) -> pd.DataFrame:
    W1 = _prep_series(w1, value_col)
    W2 = _prep_series(w2, value_col)
    W3 = _prep_series(w3, value_col)

    s21 = _median_ratio(W1[value_col], W2[value_col])
    W2r = W2 * s21
    s32 = _median_ratio(W2r[value_col], W3[value_col])
    W3r = W3 * s32

    out = pd.concat(
        [
            W1,
            W2r.loc[~W2r.index.isin(W1.index)],
            W3r.loc[~W3r.index.isin(W1.index.union(W2r.index))],
        ]
    ).sort_index()
    return out


def load_gt_weekly(cfg: ModelConfig) -> pd.DataFrame:
    """Read weekly Google Trends CSVs for each keyword and stitch windows."""
    weekly_series = []
    for kw in cfg.keywords:
        base = cfg.keyword_file_map.get(kw, kw)
        paths = [
            Path(cfg.data_dir) / f"{base}_w1.csv",
            Path(cfg.data_dir) / f"{base}_w2.csv",
            Path(cfg.data_dir) / f"{base}_w3.csv",
        ]
        if not all(p.exists() for p in paths):
            raise FileNotFoundError(f"Missing CSVs for keyword '{kw}' under {cfg.data_dir}")
        dfs = [pd.read_csv(p, sep=",", skiprows=1) for p in paths]
        stitched = stitch_three_windows(dfs[0], dfs[1], dfs[2], value_col=kw)
        weekly_series.append(stitched.rename(columns={kw: kw}))

    df_week = weekly_series[0]
    for series in weekly_series[1:]:
        df_week = df_week.join(series, how="inner")
    df_week = df_week.loc[(df_week.index >= cfg.sample_start) & (df_week.index <= cfg.sample_end)]
    df_week = df_week.sort_index()

    # enforce weekly Sunday grid if possible
    if not (df_week.index.to_series().diff().dropna().dt.days == 7).all():
        # allow irregularities but warn by raising to caller
        print("Warning: weekly Google Trends not perfectly 7-day spaced.")
    return df_week


def build_biweekly(df_week: pd.DataFrame) -> pd.DataFrame:
    """Create W12 (weeks 1-2) and W34prev (weeks 3-4 of previous month) aggregates."""
    wk = df_week.copy()
    wk.index = wk.index.tz_localize(None)
    wk["day"] = wk.index.day

    w12 = wk[wk["day"] <= 15].groupby(pd.Grouper(freq="M")).mean()
    w34 = wk[wk["day"] > 15].groupby(pd.Grouper(freq="M")).mean().shift(1, freq="M")

    w12.columns = [f"{c}_W12" for c in w12.columns]
    w34.columns = [f"{c}_W34prev" for c in w34.columns]
    out = pd.concat([w12, w34], axis=1).sort_index()
    return out


def load_unemployment(cfg: ModelConfig) -> pd.Series:
    path = Path(cfg.unemployment_file)
    if not path.exists():
        raise FileNotFoundError(f"Unemployment file not found at {path}")
    data = pd.read_csv(path, sep=",")
    date_col = "Date" if "Date" in data.columns else data.columns[0]
    val_col = [c for c in data.columns if c != date_col][0]
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.tz_localize(None)
    data = data.dropna(subset=[date_col, val_col]).set_index(date_col).sort_index()
    series = data[val_col]
    return series.loc[(series.index >= cfg.sample_start) & (series.index <= cfg.sample_end)]


def make_monthly_panel(cfg: ModelConfig) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    gt_week = load_gt_weekly(cfg)
    biweekly = build_biweekly(gt_week)
    unemp = load_unemployment(cfg)
    panel = biweekly.join(unemp.to_frame(name="Unemp"), how="inner")
    return panel, unemp, gt_week


def add_month_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    dummies = pd.get_dummies(index.month, prefix="m", drop_first=True)
    dummies.index = index
    return dummies


def adf_test(series: pd.Series) -> Dict[str, float]:
    s_clean = series.dropna()
    stat, pval, *_ = adfuller(s_clean)
    return {"stat": stat, "pval": pval}


def engle_granger(y: pd.Series, x: pd.Series) -> Dict[str, float]:
    res = coint(y.dropna(), x.dropna())
    stat, pval = res[0], res[1]
    return {"stat": stat, "pval": pval}


# ---------------------------------------------------------------------
# Lag selection helpers
# ---------------------------------------------------------------------


def _information_criterion(model_res, criterion: str) -> float:
    if criterion.upper() == "AIC":
        return model_res.aic
    if criterion.upper() == "BIC":
        return model_res.bic
    raise ValueError("criterion must be 'AIC' or 'BIC'")


def select_ar_lag(y: pd.Series, max_lag: int, criterion: str, add_const: bool, dummies: Optional[pd.DataFrame] = None) -> int:
    y = y.dropna()
    best_ic = np.inf
    best_p = 1
    for p in range(1, max_lag + 1):
        df = pd.concat([y, y.shift(1).rename("lag1")], axis=1)
        for j in range(2, p + 1):
            df[f"lag{j}"] = y.shift(j)
        if dummies is not None:
            df = df.join(dummies)
        df = df.dropna()
        if df.empty:
            continue
        y_reg = df[y.name]
        X_reg = df.drop(columns=[y.name])
        if add_const:
            X_reg = sm.add_constant(X_reg)
        res = sm.OLS(y_reg, X_reg).fit()
        ic = _information_criterion(res, criterion)
        if ic < best_ic:
            best_ic = ic
            best_p = p
    return best_p


def select_arx_lags(y: pd.Series, x: pd.Series, cfg: ModelConfig, dummies: Optional[pd.DataFrame]) -> Tuple[int, int]:
    crit = cfg.arx_lag_selection["criterion"]
    max_p = cfg.arx_lag_selection["max_ar_lags"]
    max_q = cfg.arx_lag_selection["max_gt_lags"]
    search = cfg.arx_lag_selection.get("search", "grid")
    best = (1, 0, np.inf)

    if search == "sequential":
        # first pick AR order, then GT order conditionally
        p_best = select_ar_lag(y, max_p, crit, add_const=True, dummies=dummies)
        for q in range(0, max_q + 1):
            ic = _fit_arx_ic(y, x, p_best, q, crit, dummies)
            if ic < best[2]:
                best = (p_best, q, ic)
    else:
        for p, q in itertools.product(range(1, max_p + 1), range(0, max_q + 1)):
            ic = _fit_arx_ic(y, x, p, q, crit, dummies)
            if ic < best[2]:
                best = (p, q, ic)
    return best[0], best[1]


def _fit_arx_ic(y: pd.Series, x: pd.Series, p: int, q: int, criterion: str, dummies: Optional[pd.DataFrame]) -> float:
    df = pd.concat([y, x], axis=1).rename(columns={y.name: "y", x.name: "x"})
    for j in range(1, p + 1):
        df[f"y_lag{j}"] = df["y"].shift(j)
    for k in range(0, q + 1):
        df[f"x_lag{k}"] = df["x"].shift(k)
    if dummies is not None:
        df = df.join(dummies)
    df = df.dropna()
    if df.empty:
        return np.inf
    y_reg = df["y"]
    X_reg = df.drop(columns=["y"])
    X_reg = sm.add_constant(X_reg, has_constant="add")
    res = sm.OLS(y_reg, X_reg).fit()
    return _information_criterion(res, criterion)


# ---------------------------------------------------------------------
# Model API
# ---------------------------------------------------------------------


@dataclasses.dataclass
class ModelResult:
    name: str
    predictions: pd.Series
    actuals: pd.Series
    errors: pd.Series
    metrics: Dict[str, float]
    coeff_history: Optional[Dict[pd.Timestamp, pd.Series]] = None


def compute_metrics(pred: pd.Series, act: pd.Series) -> Dict[str, float]:
    err = pred - act
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
    }


def _expanding_indices(dates: pd.DatetimeIndex, start_eval: pd.Timestamp) -> List[pd.Timestamp]:
    return [d for d in dates if d >= start_eval]


def _get_window_indices(all_dates: pd.DatetimeIndex, t: pd.Timestamp, cfg: ModelConfig) -> pd.DatetimeIndex:
    if cfg.window_mode == "rolling" and cfg.rolling_window_months:
        start = t - pd.DateOffset(months=cfg.rolling_window_months)
        return all_dates[(all_dates > start) & (all_dates < t)]
    return all_dates[all_dates < t]


def _transform_target(series: pd.Series, cfg: ModelConfig) -> pd.Series:
    if cfg.target_transform == "level":
        return series
    return series.diff()


def fit_predict_ar(panel: pd.DataFrame, cfg: ModelConfig) -> ModelResult:
    y_level = panel["Unemp"]
    y = _transform_target(y_level, cfg).rename("y")
    dummies = add_month_dummies(y.index) if cfg.seasonal_dummies else None
    best_p = select_ar_lag(y, cfg.ar_lag_selection["max_lags"], cfg.ar_lag_selection["criterion"], add_const=True, dummies=dummies)

    preds = []
    acts = []
    coeffs = {}
    eval_dates = _expanding_indices(y.index, pd.to_datetime(cfg.eval_start))
    for t in eval_dates:
        train_idx = _get_window_indices(y.index, t, cfg)
        y_train = y.loc[train_idx]
        X = pd.concat([y_train.shift(j) for j in range(1, best_p + 1)], axis=1)
        X.columns = [f"lag{j}" for j in range(1, best_p + 1)]
        if cfg.seasonal_dummies:
            X = X.join(dummies.loc[X.index])
        reg_df = pd.concat([y_train, X], axis=1).dropna()
        if reg_df.empty:
            continue
        y_reg = reg_df[y_train.name]
        X_reg = sm.add_constant(reg_df.drop(columns=[y_train.name]), has_constant="add")
        res = sm.OLS(y_reg, X_reg).fit()
        coeffs[t] = res.params

        X_t = pd.concat([y.shift(j).loc[[t]] for j in range(1, best_p + 1)], axis=1)
        X_t.columns = [f"lag{j}" for j in range(1, best_p + 1)]
        if cfg.seasonal_dummies:
            X_t = X_t.join(dummies.loc[[t]])
        X_t = sm.add_constant(X_t, has_constant="add")
        d_pred = float(res.predict(X_t))
        if cfg.target_transform == "diff":
            level_prev = y_level.loc[:t].iloc[-2]
            pred_level = level_prev + d_pred
        else:
            pred_level = d_pred
        preds.append(pred_level)
        acts.append(y_level.loc[t])

    pred_s = pd.Series(preds, index=eval_dates[: len(preds)], name="AR_nowcast")
    act_s = pd.Series(acts, index=pred_s.index, name="Unemp")
    err_s = pred_s - act_s
    return ModelResult(
        name="AR",
        predictions=pred_s,
        actuals=act_s,
        errors=err_s,
        metrics=compute_metrics(pred_s, act_s),
        coeff_history=coeffs,
    )


def fit_predict_arx(panel: pd.DataFrame, gt_col: str, cfg: ModelConfig) -> ModelResult:
    y_level = panel["Unemp"]
    x = panel[gt_col]
    y = _transform_target(y_level, cfg).rename("y")
    x_d = x.diff() if cfg.target_transform == "diff" else x
    dummies = add_month_dummies(y.index) if cfg.seasonal_dummies else None
    p_sel, q_sel = select_arx_lags(y, x_d, cfg, dummies)

    preds, acts = [], []
    coeffs = {}
    eval_dates = _expanding_indices(y.index, pd.to_datetime(cfg.eval_start))
    for t in eval_dates:
        train_idx = _get_window_indices(y.index, t, cfg)
        y_train = y.loc[train_idx]
        x_train = x_d.loc[train_idx]
        df = pd.concat([y_train, x_train], axis=1).rename(columns={x_train.name: "x"})
        for j in range(1, p_sel + 1):
            df[f"y_lag{j}"] = y_train.shift(j)
        for k in range(0, q_sel + 1):
            df[f"x_lag{k}"] = x_train.shift(k)
        if cfg.seasonal_dummies:
            df = df.join(dummies)
        reg_df = df.dropna()
        if reg_df.empty:
            continue
        y_reg = reg_df["y"]
        X_reg = sm.add_constant(reg_df.drop(columns=["y"]), has_constant="add")
        res = sm.OLS(y_reg, X_reg).fit()
        coeffs[t] = res.params

        row = {}
        for j in range(1, p_sel + 1):
            row[f"y_lag{j}"] = y.shift(j).loc[t]
        for k in range(0, q_sel + 1):
            row[f"x_lag{k}"] = x_d.shift(k).loc[t]
        if cfg.seasonal_dummies:
            for c in dummies.columns:
                row[c] = dummies.loc[t, c]
        X_t = sm.add_constant(pd.DataFrame([row]), has_constant="add")
        d_pred = float(res.predict(X_t))
        if cfg.target_transform == "diff":
            level_prev = y_level.loc[:t].iloc[-2]
            pred_level = level_prev + d_pred
        else:
            pred_level = d_pred
        preds.append(pred_level)
        acts.append(y_level.loc[t])

    pred_s = pd.Series(preds, index=eval_dates[: len(preds)], name=f"ARX_{gt_col}")
    act_s = pd.Series(acts, index=pred_s.index, name="Unemp")
    err_s = pred_s - act_s
    return ModelResult(
        name=f"ARX_{gt_col}",
        predictions=pred_s,
        actuals=act_s,
        errors=err_s,
        metrics=compute_metrics(pred_s, act_s),
        coeff_history=coeffs,
    )


def _weekly_lag_matrix(gt_week: pd.Series, month_ends: pd.DatetimeIndex, K: int) -> pd.DataFrame:
    """Build MIDAS-style weekly lags for each forecast month (0=last week in month)."""
    rows = []
    idx = []
    for m in month_ends:
        weekly_slice = gt_week.loc[gt_week.index <= m]
        weekly_slice = weekly_slice.tail(K)
        if len(weekly_slice) < K:
            continue
        row = {f"lag{k}": weekly_slice.iloc[-(k + 1)] for k in range(K)}
        rows.append(row)
        idx.append(m)
    return pd.DataFrame(rows, index=pd.to_datetime(idx))


def fit_predict_midas(gt_week: pd.Series, unemp: pd.Series, cfg: ModelConfig, restricted: bool = False) -> ModelResult:
    y_level = unemp
    y = _transform_target(y_level, cfg).rename("y")
    month_ends = y.index
    dummies = add_month_dummies(month_ends) if cfg.seasonal_dummies else None

    best_ic = np.inf
    best_K = None
    best_res = None

    for K in cfg.midas_k_candidates:
        X_all = _weekly_lag_matrix(gt_week, month_ends, K)
        if cfg.target_transform == "diff":
            X_all = X_all.diff()
        X_all = X_all.reindex(month_ends)
        if restricted:
            X_all = X_all.mean(axis=1).to_frame("avg_week")
        if cfg.include_ar_term:
            X_all["ar1"] = y.shift(1)
        if cfg.seasonal_dummies:
            X_all = X_all.join(dummies)
        df = pd.concat([y, X_all], axis=1).dropna()
        if df.empty:
            continue
        y_reg = df["y"]
        X_reg = sm.add_constant(df.drop(columns=["y"]), has_constant="add")
        res = sm.OLS(y_reg, X_reg).fit()
        ic = _information_criterion(res, cfg.midas_ic)
        if ic < best_ic:
            best_ic, best_K, best_res = ic, K, res

    if best_res is None or best_K is None:
        raise RuntimeError("MIDAS estimation failed")

    preds, acts = [], []
    coeffs = {}
    eval_dates = _expanding_indices(month_ends, pd.to_datetime(cfg.eval_start))
    for t in eval_dates:
        X_all = _weekly_lag_matrix(gt_week, month_ends[month_ends <= t], best_K)
        if cfg.target_transform == "diff":
            X_all = X_all.diff()
        X_all = X_all.reindex(month_ends[month_ends <= t])
        if restricted:
            X_all = X_all.mean(axis=1).to_frame("avg_week")
        if cfg.include_ar_term:
            X_all["ar1"] = y.shift(1)
        if cfg.seasonal_dummies:
            X_all = X_all.join(dummies)

        train_idx = _get_window_indices(X_all.index, t, cfg)
        df_train = pd.concat([y.loc[train_idx], X_all.loc[train_idx]], axis=1).dropna()
        if df_train.empty:
            continue
        y_reg = df_train["y"]
        X_reg = sm.add_constant(df_train.drop(columns=["y"]), has_constant="add")
        res = sm.OLS(y_reg, X_reg).fit()
        coeffs[t] = res.params

        X_t = X_all.loc[[t]]
        X_t = sm.add_constant(X_t, has_constant="add")
        d_pred = float(res.predict(X_t))
        if cfg.target_transform == "diff":
            level_prev = y_level.loc[:t].iloc[-2]
            pred_level = level_prev + d_pred
        else:
            pred_level = d_pred
        preds.append(pred_level)
        acts.append(y_level.loc[t])

    name = f"MIDAS_{gt_week.name}"
    if restricted:
        name += "_restricted"
    pred_s = pd.Series(preds, index=eval_dates[: len(preds)], name=name)
    act_s = pd.Series(acts, index=pred_s.index, name="Unemp")
    err_s = pred_s - act_s
    return ModelResult(
        name=name,
        predictions=pred_s,
        actuals=act_s,
        errors=err_s,
        metrics=compute_metrics(pred_s, act_s),
        coeff_history=coeffs,
    )


# ---------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------


def diebold_mariano(e1: pd.Series, e2: pd.Series, h: int = 1, crit: str = "MSE") -> Tuple[float, float]:
    """Simple DM test using Newey-West variance estimator."""
    common = e1.index.intersection(e2.index)
    d = (e1.loc[common] ** 2 - e2.loc[common] ** 2) if crit == "MSE" else (np.abs(e1.loc[common]) - np.abs(e2.loc[common]))
    d = d.dropna()
    if d.empty:
        return np.nan, np.nan
    T = len(d)
    d_bar = d.mean()
    # Newey-West with lag h-1
    gamma = [np.sum((d - d_bar)[:-k] * (d - d_bar)[k:]) / T for k in range(h)]
    var = gamma[0] + 2 * np.sum(gamma[1:])
    dm_stat = d_bar / np.sqrt(var / T)
    from scipy import stats

    pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, pval


def results_table(results: List[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"model": r.name, "RMSE": r.metrics["RMSE"], "MAE": r.metrics["MAE"]})
    return pd.DataFrame(rows).sort_values("RMSE")


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


def run_all_models(cfg: Optional[ModelConfig] = None) -> Dict[str, object]:
    cfg = cfg or ModelConfig()
    ensure_output_dirs(cfg)
    set_seed(7)

    panel, unemp, gt_week = make_monthly_panel(cfg)

    results: List[ModelResult] = []

    if "AR" in cfg.enabled_models:
        results.append(fit_predict_ar(panel, cfg))

    if "ARX" in cfg.enabled_models:
        for scheme in cfg.biweekly_schemes:
            cols = [c for c in panel.columns if c.endswith(f"_{scheme}")]
            for col in cols:
                results.append(fit_predict_arx(panel, col, cfg))

    if "MIDAS" in cfg.enabled_models:
        for kw in cfg.keywords:
            series = gt_week[kw]
            res = fit_predict_midas(series, unemp, cfg, restricted=False)
            results.append(res)
            if "MIDAS_restricted" in cfg.enabled_models:
                res_r = fit_predict_midas(series, unemp, cfg, restricted=True)
                results.append(res_r)

    table = results_table(results)
    table.to_csv(Path(cfg.output_dir) / "leaderboard.csv", index=False)

    # DM vs AR benchmark
    ar_res = next((r for r in results if r.name == "AR"), None)
    dm_rows = []
    if ar_res is not None:
        for r in results:
            if r is ar_res:
                continue
            dm_stat, pval = diebold_mariano(r.errors, ar_res.errors)
            dm_rows.append({"model": r.name, "dm_stat": dm_stat, "p_value": pval})
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(Path(cfg.output_dir) / "dm_vs_ar.csv", index=False)

    return {
        "config": cfg,
        "panel": panel,
        "unemp": unemp,
        "gt_week": gt_week,
        "results": results,
        "leaderboard": table,
        "dm": dm_df,
    }


# Convenience default config
default_config = ModelConfig()


