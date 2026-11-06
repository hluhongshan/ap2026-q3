#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
DEFAULT_PANEL = CLEAN / "macro_panel.csv"
DEFAULT_META = CLEAN / "macro_meta.csv"
DEFAULT_OUT = ROOT / "results" / "approach1" / "macro"
CACHE = ROOT / ".cache"

FACTOR_EXCLUSIONS = ("gdp_yoy", "cgb10y", "spread_10y_2y", "shibor_3m")
MAIN_CONTROLS = ("cgb10y", "spread_10y_2y")
SHIBOR_CONTROL = "shibor_3m"
REQUIRED_SERIES = ("pmi", "gdp_yoy", "cgb10y", "spread_10y_2y", "shibor_3m")
LABEL_RULES = ("two_month", "one_month")


@dataclass
class RunConfig:
    seed: int = 42
    min_train: int = 60
    tuning_months: int = 36
    max_rank: int = 8
    min_coverage: float = 0.5
    ridge: float = 1e-4
    nw_lag: int = 12
    cluster_window: int = 36
    cluster_step: int = 12
    threshold: float = 50.0
    sens_threshold: float = 49.5
    min_pair_corr_obs: int = 18


console = Console(record=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q3 Option A: classical PMI regime forecasting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument("--tuning-months", type=int, default=36)
    parser.add_argument("--max-rank", type=int, default=8)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--nw-lag", type=int, default=12)
    parser.add_argument("--cluster-window", type=int, default=36)
    parser.add_argument("--cluster-step", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--sens-threshold", type=float, default=49.5)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        seed=args.seed,
        min_train=args.min_train,
        tuning_months=args.tuning_months,
        max_rank=args.max_rank,
        min_coverage=args.min_coverage,
        ridge=args.ridge,
        nw_lag=args.nw_lag,
        cluster_window=args.cluster_window,
        cluster_step=args.cluster_step,
        threshold=args.threshold,
        sens_threshold=args.sens_threshold,
    )


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_inputs(panel_path: Path, meta_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], list[str]]:
    panel = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    meta = pd.read_csv(meta_path)
    if "series" not in meta.columns or "lag" not in meta.columns:
        raise ValueError("macro_meta.csv must contain series and lag columns")
    lags = {str(row.series): int(row.lag) for row in meta.itertuples()}
    series = [str(col) for col in panel.columns if col != "date"]
    if len(series) != len(set(series)):
        raise ValueError("duplicate panel series")
    missing_meta = [col for col in series if col not in lags]
    if missing_meta:
        raise ValueError(f"panel series missing from macro_meta: {missing_meta}")
    missing_panel = [name for name in lags if name not in panel.columns]
    if missing_panel:
        raise ValueError(f"macro_meta series missing from panel: {missing_panel}")
    required = [name for name in REQUIRED_SERIES if name not in panel.columns]
    if required:
        raise ValueError(f"required series missing from panel: {required}")
    if not panel["date"].is_monotonic_increasing:
        raise ValueError("panel dates must be sorted")
    if panel["date"].duplicated().any():
        raise ValueError("panel dates must be unique")
    if not panel["date"].dt.is_month_end.all():
        raise ValueError("panel dates must be month ends")
    if any(lag < 0 for lag in lags.values()):
        raise ValueError("lags must be non-negative")
    return panel, meta, lags, series


def align_information_set(panel: pd.DataFrame, lags: dict[str, int]) -> pd.DataFrame:
    """Shift each series forward by its availability lag so row t is public at t."""
    aligned = pd.DataFrame({"date": panel["date"].to_numpy()})
    for series in panel.columns:
        if series == "date":
            continue
        aligned[series] = panel[series].shift(int(lags[series])).to_numpy()
    return aligned


def build_label(aligned: pd.DataFrame, threshold: float = 50.0, rule: str = "two_month") -> np.ndarray:
    if rule not in LABEL_RULES:
        raise ValueError(f"unknown label rule {rule}")
    pmi = aligned["pmi"].to_numpy(dtype=float)
    below = np.isfinite(pmi) & (pmi < threshold)
    d = np.full(len(pmi), np.nan)
    if rule == "one_month":
        d[:] = below.astype(float)
        d[~np.isfinite(pmi)] = np.nan
    else:
        d[1:] = (below[1:] & below[:-1]).astype(float)
        d[~np.isfinite(pmi)] = np.nan
        previous_missing = ~np.isfinite(pmi[:-1])
        d[1:][previous_missing] = np.nan
    return d


def select_factor_series(
    aligned: pd.DataFrame,
    fit_idx: np.ndarray,
    exclusions: tuple[str, ...] = FACTOR_EXCLUSIONS,
    min_coverage: float = 0.5,
) -> list[str]:
    block = aligned.iloc[fit_idx]
    selected = []
    for series in aligned.columns:
        if series == "date" or series in exclusions:
            continue
        values = block[series]
        if float(values.notna().mean()) < min_coverage:
            continue
        observed = values.dropna()
        if len(observed) < 2 or float(observed.std(ddof=1)) == 0.0:
            continue
        selected.append(series)
    return selected


def fit_pca(
    aligned: pd.DataFrame,
    factor_series: list[str],
    fit_idx: np.ndarray,
    apply_idx: np.ndarray,
    max_rank: int,
) -> dict | None:
    if not factor_series:
        return None
    x_fit = aligned.iloc[fit_idx][factor_series].to_numpy(dtype=float)
    mean = np.nanmean(x_fit, axis=0)
    std = np.nanstd(x_fit, axis=0, ddof=1)
    good = np.isfinite(mean) & np.isfinite(std) & (std > 0.0)
    if not good.any():
        return None
    mean = mean[good]
    std = std[good]
    kept = [series for series, keep in zip(factor_series, good) if keep]
    x_fit = x_fit[:, good]
    z_fit = (x_fit - mean) / std
    z_fit[~np.isfinite(z_fit)] = 0.0
    x_apply = aligned.iloc[apply_idx][kept].to_numpy(dtype=float)
    z_apply = (x_apply - mean) / std
    z_apply[~np.isfinite(z_apply)] = 0.0
    _, singular, vt = np.linalg.svd(z_fit, full_matrices=False)
    total = float(np.sum(singular**2))
    if total <= 0.0:
        return None
    n_components = int(min(max_rank, len(singular), len(kept)))
    singular = singular[:n_components]
    vt = vt[:n_components].copy()
    for row in range(vt.shape[0]):
        anchor = int(np.argmax(np.abs(vt[row])))
        if vt[row, anchor] < 0.0:
            vt[row] *= -1.0
    scores_fit = z_fit @ vt.T
    scores_apply = z_apply @ vt.T
    return {
        "series": kept,
        "mean": mean,
        "std": std,
        "loadings": vt,
        "scores_fit": scores_fit,
        "scores_apply": scores_apply,
        "explained_variance_ratio": singular**2 / total,
        "n_components": n_components,
    }


def fit_standardizer(
    aligned: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    fit_idx: np.ndarray,
    min_coverage: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    means = []
    stds = []
    kept = []
    block = aligned.iloc[fit_idx]
    for column in columns:
        values = block[column]
        if float(values.notna().mean()) < min_coverage:
            continue
        observed = values.dropna()
        if len(observed) < 2 or float(observed.std(ddof=1)) == 0.0:
            continue
        kept.append(column)
        means.append(float(observed.mean()))
        stds.append(float(observed.std(ddof=1)))
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float), kept


def apply_standardizer(
    aligned: pd.DataFrame,
    idx: np.ndarray,
    columns: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    if not columns:
        return np.empty((len(idx), 0), dtype=float)
    x = aligned.iloc[idx][columns].to_numpy(dtype=float)
    z = (x - means) / stds
    z[~np.isfinite(z)] = 0.0
    return z


def probit_terms(beta: np.ndarray, x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta = x @ beta
    q = 2.0 * y - 1.0
    a = q * eta
    log_cdf = log_ndtr(a)
    log_pdf = -0.5 * a * a - 0.5 * math.log(2.0 * math.pi)
    ratio = np.exp(np.clip(log_pdf - log_cdf, -745.0, 700.0))
    lam = q * ratio
    loglik = float(np.sum(log_cdf))
    grad = x.T @ lam
    if ridge > 0.0:
        loglik -= 0.5 * ridge * float(np.sum(beta[1:] ** 2))
        grad[1:] -= ridge * beta[1:]
    weights = lam * (eta + lam)
    hessian = -((x * weights[:, None]).T @ x)
    if ridge > 0.0:
        hessian[1:, 1:] -= ridge * np.eye(len(beta) - 1)
    return loglik, grad, hessian, lam, weights


def fit_probit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-4, max_iter: int = 200, tol: float = 1e-9) -> dict:
    n, k = x.shape
    y = np.asarray(y, dtype=float)
    if n == 0:
        raise ValueError("empty probit sample")
    beta = np.zeros(k)
    share = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    beta[0] = float(np.log(share / (1.0 - share)))
    loglik, grad, hessian, _, _ = probit_terms(beta, x, y, ridge)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        try:
            step_direction = np.linalg.solve(-hessian, grad)
        except np.linalg.LinAlgError:
            step_direction = np.linalg.lstsq(-hessian, grad, rcond=None)[0]
        if not np.all(np.isfinite(step_direction)):
            break
        step = 1.0
        accepted = False
        for _ in range(50):
            candidate = beta + step * step_direction
            candidate_ll, candidate_grad, candidate_hessian, _, _ = probit_terms(candidate, x, y, ridge)
            if np.isfinite(candidate_ll) and candidate_ll >= loglik - 1e-12:
                beta = candidate
                loglik = candidate_ll
                grad = candidate_grad
                hessian = candidate_hessian
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(grad))) < tol or float(np.max(np.abs(step * step_direction))) < 1e-10:
            converged = True
            break
    if not converged:
        def objective(coef):
            value, gradient, _, _, _ = probit_terms(coef, x, y, ridge)
            return -value, -gradient

        result = minimize(objective, beta, jac=True, method="BFGS", options={"maxiter": 500, "gtol": 1e-8})
        if result.success or np.isfinite(result.fun):
            beta = result.x
            loglik, grad, hessian, _, _ = probit_terms(beta, x, y, ridge)
            converged = bool(result.success)
            n_iter += int(result.nit)
    return {
        "beta": beta,
        "loglik": loglik,
        "grad": grad,
        "hessian": hessian,
        "converged": converged,
        "n_iter": n_iter,
        "n": int(n),
        "ridge": ridge,
    }


def probit_standard_errors(x: np.ndarray, y: np.ndarray, beta: np.ndarray, nw_lag: int) -> dict:
    _, _, hessian, lam, _ = probit_terms(beta, x, y, 0.0)
    info = -hessian
    jitter = 1e-10 * max(1.0, float(np.max(np.abs(np.diag(info)))))
    info = info + jitter * np.eye(info.shape[0])
    try:
        info_inv = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        info_inv = np.linalg.pinv(info)
    scores = x * lam[:, None]
    meat = scores.T @ scores
    for lag in range(1, nw_lag + 1):
        weight = 1.0 - lag / (nw_lag + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    nw_cov = info_inv @ meat @ info_inv
    info_se = np.sqrt(np.maximum(np.diag(info_inv), 0.0))
    nw_se = np.sqrt(np.maximum(np.diag(nw_cov), 0.0))
    return {"info_se": info_se, "nw_se": nw_se, "info_inv": info_inv, "nw_cov": nw_cov}


def rmse(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


def auc_score(y_true, score) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == 0.0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((np.sum(ranks[y == 1.0]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_metrics(y_true, prob) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {"Brier": float("nan"), "AUC": float("nan"), "accuracy": float("nan"), "n": 0, "positives": 0, "class_share": float("nan")}
    return {
        "Brier": float(np.mean((p - y) ** 2)),
        "AUC": auc_score(y, p),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1.0))),
        "n": int(len(y)),
        "positives": int(np.sum(y == 1.0)),
        "class_share": float(np.mean(y)),
    }


def quarter_records(aligned: pd.DataFrame) -> list[dict]:
    periods = aligned["date"].dt.to_period("Q")
    records = []
    for period in periods.drop_duplicates():
        idx = np.flatnonzero((periods == period).to_numpy())
        gdp_row = int(idx.max() + 1)
        records.append(
            {
                "quarter": str(period),
                "start": int(idx.min()),
                "end": int(idx.max()),
                "gdp_row": gdp_row if gdp_row < len(aligned) else None,
            }
        )
    return records


def bridge_from_scores(
    scores: np.ndarray,
    rank: int,
    t: int,
    aligned: pd.DataFrame,
    quarters: list[dict],
) -> dict:
    rank = min(rank, scores.shape[1])
    train_x = []
    train_y = []
    for record in quarters:
        if record["gdp_row"] is None or record["gdp_row"] > t:
            continue
        value = aligned["gdp_yoy"].iloc[record["gdp_row"]]
        if not np.isfinite(value):
            continue
        months = np.arange(record["start"], record["end"] + 1)
        fbar = scores[months, :rank].mean(axis=0)
        train_x.append(np.concatenate([[1.0], fbar]))
        train_y.append(float(value))
    if len(train_y) < rank + 2:
        return {"nowcast": float("nan"), "rw_drift": float("nan"), "n_quarters": len(train_y), "rank_eff": rank}
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    coef, _, _, _ = np.linalg.lstsq(train_x, train_y, rcond=None)
    current = next(record for record in quarters if record["start"] <= t <= record["end"])
    months = np.arange(current["start"], t + 1)
    fbar = scores[months, :rank].mean(axis=0)
    nowcast = float(np.concatenate([[1.0], fbar]) @ coef)
    drift = float(np.mean(np.diff(train_y))) if len(train_y) > 1 else 0.0
    rw_drift = float(train_y[-1] + drift)
    return {
        "nowcast": nowcast,
        "rw_drift": rw_drift,
        "n_quarters": int(len(train_y)),
        "rank_eff": int(rank),
        "coef": coef,
    }


def prepare_origin(aligned: pd.DataFrame, t: int, cfg: RunConfig, controls: tuple[str, ...], rank: int | None = None) -> dict:
    public_idx = np.arange(0, t + 1)
    factor_series = select_factor_series(aligned, public_idx, FACTOR_EXCLUSIONS, cfg.min_coverage)
    pca = fit_pca(aligned, factor_series, public_idx, public_idx, cfg.max_rank)
    if pca is None:
        raise ValueError(f"no usable factor series at origin {t}")
    use_rank = pca["n_components"] if rank is None else min(rank, pca["n_components"])
    means, stds, control_series = fit_standardizer(aligned, controls, public_idx, cfg.min_coverage)
    return {
        "public_idx": public_idx,
        "factor_series": factor_series,
        "pca": pca,
        "rank": use_rank,
        "control_series": control_series,
        "control_means": means,
        "control_stds": stds,
    }


def fit_origin(aligned: pd.DataFrame, d: np.ndarray, t: int, cfg: RunConfig, rank: int, controls: tuple[str, ...]) -> dict:
    prep = prepare_origin(aligned, t, cfg, controls, rank)
    public_idx = prep["public_idx"]
    train_idx = np.arange(0, t)
    scores = prep["pca"]["scores_apply"]
    use_rank = prep["rank"]
    control_series = prep["control_series"]
    control_train = apply_standardizer(
        aligned, train_idx, control_series, prep["control_means"], prep["control_stds"]
    )
    control_t = apply_standardizer(
        aligned, np.array([t]), control_series, prep["control_means"], prep["control_stds"]
    )
    x_train = np.column_stack([np.ones(len(train_idx)), scores[train_idx, :use_rank], control_train])
    x_t = np.concatenate([[1.0], scores[t, :use_rank], control_t[0]])
    y_train = d[train_idx + 1]
    mask = np.isfinite(y_train)
    fit = fit_probit(x_train[mask], y_train[mask], ridge=cfg.ridge)
    probability = float(ndtr(float(x_t @ fit["beta"])))
    prior_probability = float(np.mean(y_train[mask]))
    return {
        "t": int(t),
        "z_t": x_t,
        "probability": probability,
        "prior_probability": prior_probability,
        "persistence_probability": float(d[t]),
        "beta": fit["beta"],
        "converged": fit["converged"],
        "n_train": int(mask.sum()),
        "train_target_idx": (train_idx + 1)[mask],
        "train_feature_idx": train_idx[mask],
        "rank": int(use_rank),
        "factor_series": prep["factor_series"],
        "control_series": control_series,
    }


def predict_at_origin(
    raw: pd.DataFrame,
    meta: pd.DataFrame,
    t: int,
    cfg: RunConfig,
    rank: int,
    controls: tuple[str, ...] = MAIN_CONTROLS,
    threshold: float | None = None,
    rule: str = "two_month",
) -> dict:
    lags = {str(row.series): int(row.lag) for row in meta.itertuples()}
    aligned = align_information_set(raw, lags)
    threshold = cfg.threshold if threshold is None else threshold
    d = build_label(aligned, threshold, rule)
    return fit_origin(aligned, d, t, cfg, rank, controls)


def walk_forward_probit(
    aligned: pd.DataFrame,
    d: np.ndarray,
    origins: np.ndarray,
    cfg: RunConfig,
    rank: int,
    controls: tuple[str, ...],
    block_start: int,
) -> pd.DataFrame:
    rows = []
    dates = aligned["date"].to_numpy()
    for t in origins:
        result = fit_origin(aligned, d, int(t), cfg, rank, controls)
        rows.append(
            {
                "origin_date": pd.Timestamp(dates[t]),
                "target_date": pd.Timestamp(dates[t + 1]),
                "origin_index": int(t),
                "block": "test" if t >= block_start else "tuning",
                "rank": result["rank"],
                "p_probit": result["probability"],
                "p_prior": result["prior_probability"],
                "p_persistence": result["persistence_probability"],
                "d_t": float(d[t]),
                "y_true": float(d[t + 1]),
                "n_train": result["n_train"],
                "converged": result["converged"],
            }
        )
    return pd.DataFrame(rows)


def select_rank_by_nowcast(
    aligned: pd.DataFrame,
    quarters: list[dict],
    origins: np.ndarray,
    cfg: RunConfig,
) -> tuple[int, pd.DataFrame]:
    rows = []
    for t in origins:
        prep = prepare_origin(aligned, t, cfg, MAIN_CONTROLS, None)
        scores = prep["pca"]["scores_apply"]
        current = next(record for record in quarters if record["start"] <= t <= record["end"])
        position = int(t - current["start"] + 1)
        actual = float("nan")
        if current["gdp_row"] is not None:
            actual = float(aligned["gdp_yoy"].iloc[current["gdp_row"]])
        for rank in range(1, cfg.max_rank + 1):
            bridge = bridge_from_scores(scores, rank, int(t), aligned, quarters)
            rows.append(
                {
                    "origin_index": int(t),
                    "origin_date": pd.Timestamp(aligned["date"].iloc[t]),
                    "quarter": current["quarter"],
                    "month_position": position,
                    "rank": rank,
                    "rank_eff": bridge["rank_eff"],
                    "nowcast": bridge["nowcast"],
                    "actual": actual,
                    "n_quarters": bridge["n_quarters"],
                }
            )
    frame = pd.DataFrame(rows)
    month_three = frame[(frame["month_position"] == 3) & frame["actual"].notna()]
    summary = (
        month_three.groupby("rank")
        .apply(lambda group: rmse(group["actual"], group["nowcast"]), include_groups=False)
        .rename("cv_rmse_month3")
        .reset_index()
    )
    summary["n"] = month_three.groupby("rank").size().reindex(summary["rank"]).to_numpy()
    chosen = int(summary.sort_values(["cv_rmse_month3", "rank"]).iloc[0]["rank"])
    return chosen, summary


def run_nowcast(
    aligned: pd.DataFrame,
    quarters: list[dict],
    origins: np.ndarray,
    cfg: RunConfig,
    rank: int,
    block_start: int,
) -> pd.DataFrame:
    rows = []
    for t in origins:
        prep = prepare_origin(aligned, t, cfg, MAIN_CONTROLS, rank)
        scores = prep["pca"]["scores_apply"]
        bridge = bridge_from_scores(scores, prep["rank"], int(t), aligned, quarters)
        current = next(record for record in quarters if record["start"] <= t <= record["end"])
        position = int(t - current["start"] + 1)
        actual = float("nan")
        if current["gdp_row"] is not None:
            actual = float(aligned["gdp_yoy"].iloc[current["gdp_row"]])
        rows.append(
            {
                "origin_date": pd.Timestamp(aligned["date"].iloc[t]),
                "quarter": current["quarter"],
                "month_position": position,
                "block": "test" if t >= block_start else "tuning",
                "rank": int(prep["rank"]),
                "nowcast_bridge": bridge["nowcast"],
                "nowcast_rw_drift": bridge["rw_drift"],
                "actual_gdp": actual,
                "error_bridge": bridge["nowcast"] - actual if np.isfinite(actual) else float("nan"),
                "error_rw_drift": bridge["rw_drift"] - actual if np.isfinite(actual) else float("nan"),
                "n_training_quarters": bridge["n_quarters"],
            }
        )
    return pd.DataFrame(rows)


def nowcast_rmse_table(paths: pd.DataFrame, exclude_covid: bool) -> dict:
    frame = paths[(paths["block"] == "test") & paths["actual_gdp"].notna()].copy()
    if exclude_covid:
        year = frame["quarter"].str.slice(0, 4).astype(int)
        frame = frame[~year.isin([2020, 2021])]
    result = {}
    for position in (1, 2, 3):
        subset = frame[frame["month_position"] == position]
        result[str(position)] = {
            "n": int(len(subset)),
            "bridge": rmse(subset["actual_gdp"], subset["nowcast_bridge"]),
            "rw_drift": rmse(subset["actual_gdp"], subset["nowcast_rw_drift"]),
        }
    return result


def descriptive_pca(aligned: pd.DataFrame, cfg: RunConfig) -> dict:
    idx = np.arange(len(aligned))
    factor_series = select_factor_series(aligned, idx, FACTOR_EXCLUSIONS, cfg.min_coverage)
    pca = fit_pca(aligned, factor_series, idx, idx, cfg.max_rank)
    if pca is None:
        raise ValueError("no usable factor series for descriptive PCA")
    top = {}
    for component in range(pca["n_components"]):
        order = np.argsort(-np.abs(pca["loadings"][component]))
        top[str(component + 1)] = [
            {"series": pca["series"][index], "loading": float(pca["loadings"][component, index])}
            for index in order[:5]
        ]
    return {
        "pca": pca,
        "explained_variance_ratio": [float(value) for value in pca["explained_variance_ratio"]],
        "top_loadings": top,
    }


def descriptive_probit(
    aligned: pd.DataFrame,
    d: np.ndarray,
    cfg: RunConfig,
    rank: int,
    controls: tuple[str, ...],
) -> dict:
    idx = np.arange(len(aligned))
    factor_series = select_factor_series(aligned, idx, FACTOR_EXCLUSIONS, cfg.min_coverage)
    pca = fit_pca(aligned, factor_series, idx, idx, cfg.max_rank)
    if pca is None:
        raise ValueError("no usable factor series for descriptive probit")
    use_rank = min(rank, pca["n_components"])
    means, stds, control_series = fit_standardizer(aligned, controls, idx, cfg.min_coverage)
    train_idx = np.arange(0, len(aligned) - 1)
    z = apply_standardizer(aligned, train_idx, control_series, means, stds)
    x = np.column_stack([np.ones(len(train_idx)), pca["scores_apply"][train_idx, :use_rank], z])
    y = d[train_idx + 1]
    mask = np.isfinite(y)
    fit = fit_probit(x[mask], y[mask], ridge=cfg.ridge)
    se = probit_standard_errors(x[mask], y[mask], fit["beta"], cfg.nw_lag)
    names = ["intercept"] + [f"PC{index + 1}" for index in range(use_rank)] + control_series
    table = []
    for index, name in enumerate(names):
        estimate = float(fit["beta"][index])
        info_se = float(se["info_se"][index])
        nw_se = float(se["nw_se"][index])
        table.append(
            {
                "term": name,
                "estimate": estimate,
                "info_se": info_se,
                "info_z": estimate / info_se if info_se > 0 else float("nan"),
                "nw_se": nw_se,
                "nw_z": estimate / nw_se if nw_se > 0 else float("nan"),
            }
        )
    return {
        "descriptive": True,
        "note": "full-sample descriptive fit; not used for walk-forward evaluation",
        "coefficients": table,
        "n": int(mask.sum()),
        "loglik": float(fit["loglik"]),
        "converged": bool(fit["converged"]),
        "nw_lag": int(cfg.nw_lag),
        "ridge": float(cfg.ridge),
        "rank": int(use_rank),
        "factor_series": pca["series"],
        "control_series": control_series,
    }


def cluster_reference_panel(raw: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, list[dict]]:
    dates = raw["date"]
    series = [column for column in raw.columns if column not in ("date", "gdp_yoy")]
    records = []
    windows = []
    start = 0
    while start + cfg.cluster_window <= len(raw):
        window = raw.iloc[start : start + cfg.cluster_window]
        coverage = window[series].notna().mean()
        keep = [column for column in series if float(coverage[column]) >= cfg.min_coverage]
        corr = window[keep].corr(min_periods=cfg.min_pair_corr_obs)
        while len(keep) > 3 and corr.isna().any().any():
            drop = corr.isna().sum().idxmax()
            keep.remove(drop)
            corr = window[keep].corr(min_periods=cfg.min_pair_corr_obs)
        if len(keep) < 3:
            start += cfg.cluster_step
            continue
        rho = corr.to_numpy(dtype=float)
        rho = np.clip(rho, -1.0, 1.0)
        np.fill_diagonal(rho, 1.0)
        distance = np.sqrt(np.maximum(2.0 * (1.0 - rho), 0.0))
        np.fill_diagonal(distance, 0.0)
        link = linkage(squareform(distance, checks=False), method="ward")
        best_k = None
        best_silhouette = -np.inf
        silhouette_by_k = {}
        max_k = min(10, len(keep) - 1)
        for k in range(2, max_k + 1):
            labels = fcluster(link, k, criterion="maxclust")
            if len(np.unique(labels)) < 2:
                continue
            value = float(silhouette_score(distance, labels, metric="precomputed"))
            silhouette_by_k[str(k)] = value
            if value > best_silhouette:
                best_silhouette = value
                best_k = k
        if best_k is None:
            start += cfg.cluster_step
            continue
        labels = fcluster(link, best_k, criterion="maxclust")
        window_info = {
            "start_index": int(start),
            "end_index": int(start + cfg.cluster_window - 1),
            "start_date": pd.Timestamp(dates.iloc[start]),
            "end_date": pd.Timestamp(dates.iloc[start + cfg.cluster_window - 1]),
            "series": keep,
            "labels": labels,
            "linkage": link,
            "distance": distance,
            "k": int(best_k),
            "silhouette": best_silhouette,
            "silhouette_by_k": silhouette_by_k,
        }
        windows.append(window_info)
        for column, label in zip(keep, labels):
            records.append(
                {
                    "window_start": window_info["start_date"],
                    "window_end": window_info["end_date"],
                    "series": column,
                    "cluster": int(label),
                    "k": int(best_k),
                    "silhouette": best_silhouette,
                }
            )
        start += cfg.cluster_step
    return pd.DataFrame(records), windows


def adjacent_ari(windows: list[dict]) -> list[dict]:
    rows = []
    for left, right in zip(windows[:-1], windows[1:]):
        common = [series for series in left["series"] if series in right["series"]]
        if len(common) < 3:
            rows.append(
                {
                    "window_left": left["end_date"],
                    "window_right": right["end_date"],
                    "n_common": len(common),
                    "ari": float("nan"),
                }
            )
            continue
        left_map = dict(zip(left["series"], left["labels"]))
        right_map = dict(zip(right["series"], right["labels"]))
        left_labels = [left_map[series] for series in common]
        right_labels = [right_map[series] for series in common]
        rows.append(
            {
                "window_left": left["end_date"],
                "window_right": right["end_date"],
                "n_common": len(common),
                "ari": float(adjusted_rand_score(left_labels, right_labels)),
            }
        )
    return rows


def lead_lag_table(raw: pd.DataFrame, window: dict, max_lag: int = 4) -> tuple[list[dict], dict[int, list[str]]]:
    periods = raw["date"].dt.to_period("Q")
    gdp = raw.groupby(periods)["gdp_yoy"].last()
    members = {}
    for label in sorted(set(window["labels"])):
        members[int(label)] = [
            series for series, assigned in zip(window["series"], window["labels"]) if int(assigned) == label
        ]
    standardized = pd.DataFrame(index=raw.index)
    for series in window["series"]:
        values = raw[series]
        observed = values.dropna()
        std = float(observed.std(ddof=1))
        if std > 0.0:
            standardized[series] = (values - float(observed.mean())) / std
    rows = []
    for label, names in members.items():
        names = [name for name in names if name in standardized.columns]
        if not names:
            continue
        member_frame = standardized[names]
        cluster_monthly = member_frame.mean(axis=1, skipna=True)
        cluster_quarterly = cluster_monthly.groupby(periods).mean()
        for lag in range(-max_lag, max_lag + 1):
            pair = pd.concat([cluster_quarterly, gdp.shift(-lag)], axis=1).dropna()
            value = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) >= 8 else float("nan")
            rows.append({"cluster": label, "lag": lag, "corr": value, "n": int(len(pair))})
    return rows, members


def write_html(figure: go.Figure, path: Path) -> None:
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def regime_figure(frame: pd.DataFrame) -> go.Figure:
    test = frame[frame["block"] == "test"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=test["target_date"],
            y=test["p_probit"],
            mode="lines",
            name="Probit",
            line=dict(color="#1f77b4", width=1.8),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=test["target_date"],
            y=test["p_prior"],
            mode="lines",
            name="Expanding prior",
            line=dict(color="#ff7f0e", width=1.2, dash="dot"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=test["target_date"],
            y=test["p_persistence"],
            mode="lines",
            name="Persistence",
            line=dict(color="#2ca02c", width=1.0, dash="dash"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=test["target_date"],
            y=test["y_true"],
            mode="markers",
            name="Realised d",
            marker=dict(color="#d62728", size=4, symbol="circle-open"),
        )
    )
    figure.add_hline(y=0.5, line_dash="dot", line_color="#888")
    figure.update_layout(
        title="Q3A test-block contraction-regime probabilities",
        xaxis_title="Target month",
        yaxis_title="P(d=1)",
        template="plotly_white",
        hovermode="x unified",
    )
    return figure


def stability_figure(windows: list[dict], ari_rows: list[dict]) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Adjusted Rand index, adjacent windows", "Chosen k by window", "Silhouette by window"),
    )
    ari_dates = [row["window_right"] for row in ari_rows]
    ari_values = [row["ari"] for row in ari_rows]
    figure.add_trace(
        go.Scatter(x=ari_dates, y=ari_values, mode="lines+markers", name="ARI", line=dict(color="#9467bd")),
        row=1,
        col=1,
    )
    end_dates = [window["end_date"] for window in windows]
    figure.add_trace(
        go.Scatter(
            x=end_dates,
            y=[window["k"] for window in windows],
            mode="lines+markers",
            name="k",
            line=dict(color="#1f77b4"),
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=end_dates,
            y=[window["silhouette"] for window in windows],
            mode="lines+markers",
            name="Silhouette",
            line=dict(color="#ff7f0e"),
        ),
        row=3,
        col=1,
    )
    figure.update_yaxes(title_text="ARI", row=1, col=1)
    figure.update_yaxes(title_text="k", row=2, col=1)
    figure.update_yaxes(title_text="Silhouette", row=3, col=1)
    figure.update_layout(
        title="Q3A clustering stability (reference-month series, descriptive only)",
        template="plotly_white",
        hovermode="x unified",
    )
    return figure


def dendrogram_figure(window: dict) -> go.Figure:
    tree = dendrogram(window["linkage"], labels=window["series"], no_plot=True)
    figure = go.Figure()
    for xs, ys in zip(tree["icoord"], tree["dcoord"]):
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#333333", width=1.0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    tickvals = [5 + 10 * index for index in range(len(tree["ivl"]))]
    figure.update_layout(
        title=f"Q3A Ward dendrogram, {window['start_date']:%Y-%m} to {window['end_date']:%Y-%m}",
        xaxis=dict(tickvals=tickvals, ticktext=tree["ivl"], tickangle=90),
        yaxis_title="Ward distance",
        template="plotly_white",
        height=700,
    )
    return figure


def nowcast_figure(paths: pd.DataFrame) -> go.Figure:
    test = paths[(paths["block"] == "test") & paths["actual_gdp"].notna()].copy()
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Nowcast revision paths, test quarters", "RMSE by month position"),
        column_widths=[0.62, 0.38],
    )
    for quarter, group in test.groupby("quarter"):
        group = group.sort_values("month_position")
        figure.add_trace(
            go.Scatter(
                x=group["month_position"],
                y=group["nowcast_bridge"],
                mode="lines+markers",
                line=dict(color="rgba(31,119,180,0.25)", width=1.0),
                marker=dict(size=3, color="rgba(31,119,180,0.35)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    actual_axis = test.groupby("quarter").agg(
        month_position=("month_position", "max"), actual_gdp=("actual_gdp", "first")
    )
    figure.add_trace(
        go.Scatter(
            x=actual_axis["month_position"],
            y=actual_axis["actual_gdp"],
            mode="markers",
            name="Realised GDP",
            marker=dict(color="#d62728", size=5),
        ),
        row=1,
        col=1,
    )
    positions = [1, 2, 3]
    bridge_rmse = []
    rw_rmse = []
    for position in positions:
        subset = test[test["month_position"] == position]
        bridge_rmse.append(rmse(subset["actual_gdp"], subset["nowcast_bridge"]))
        rw_rmse.append(rmse(subset["actual_gdp"], subset["nowcast_rw_drift"]))
    figure.add_trace(
        go.Bar(x=positions, y=bridge_rmse, name="Bridge", marker_color="#1f77b4"),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(x=positions, y=rw_rmse, name="RW with drift", marker_color="#ff7f0e"),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Months elapsed in quarter", row=1, col=1)
    figure.update_yaxes(title_text="GDP YoY nowcast", row=1, col=1)
    figure.update_xaxes(title_text="Month position", row=1, col=2)
    figure.update_yaxes(title_text="RMSE (pp)", row=1, col=2)
    figure.update_layout(
        title="Q3A GDP nowcast revisions and test RMSE",
        template="plotly_white",
        barmode="group",
    )
    return figure


def rich_table(title: str, columns: list[str], rows: list[list]) -> Table:
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*[("" if value is None else str(value)) for value in row])
    return table


def print_report(
    lags: dict[str, int],
    cfg: RunConfig,
    test_origins: np.ndarray,
    test_targets: np.ndarray,
    rank_summary: pd.DataFrame,
    chosen_rank: int,
    probit_metrics: dict,
    sensitivity: dict,
    nowcast_all: dict,
    nowcast_ex_covid: dict,
    coef_table: list[dict],
    windows: list[dict],
    ari_rows: list[dict],
    lead_lag_rows: list[dict],
) -> None:
    console.rule("Q3A classical PMI regime model")
    console.print(
        f"Seed={cfg.seed}  min_train={cfg.min_train}  tuning={cfg.tuning_months} months  "
        f"chosen PCA rank={chosen_rank}  ridge={cfg.ridge}  NW lag={cfg.nw_lag}"
    )
    lag_rows = [[series, lag] for series, lag in lags.items()]
    console.print(rich_table("Availability lags (months)", ["series", "lag"], lag_rows))
    console.print(
        rich_table(
            "Test block",
            ["items", "value"],
            [
                ["origin dates", f"{test_origins[0]} to {test_origins[-1]}"],
                ["target dates", f"{test_targets[0]} to {test_targets[-1]}"],
                ["origins", len(test_origins)],
                ["class share", f"{probit_metrics['probit']['class_share']:.4f}"],
                ["positives", probit_metrics["probit"]["positives"]],
            ],
        )
    )
    rank_rows = [
        [int(row["rank"]), f"{row['cv_rmse_month3']:.4f}", int(row["n"])]
        for _, row in rank_summary.iterrows()
    ]
    console.print(rich_table("PCA rank selection (walk-forward CV nowcast RMSE, month 3)", ["rank", "RMSE", "n"], rank_rows))
    metric_rows = []
    for model in ("probit", "prior", "persistence"):
        values = probit_metrics[model]
        metric_rows.append(
            [
                model,
                f"{values['Brier']:.4f}",
                "nan" if not np.isfinite(values["AUC"]) else f"{values['AUC']:.4f}",
                f"{values['accuracy']:.4f}",
            ]
        )
    console.print(rich_table("Test classification metrics", ["model", "Brier", "AUC", "accuracy"], metric_rows))
    sens_rows = []
    for name, block in sensitivity.items():
        for model in ("probit", "prior", "persistence"):
            values = block[model]
            sens_rows.append(
                [
                    name,
                    model,
                    f"{values['Brier']:.4f}",
                    "nan" if not np.isfinite(values["AUC"]) else f"{values['AUC']:.4f}",
                    f"{values['accuracy']:.4f}",
                    f"{values['class_share']:.4f}",
                ]
            )
    console.print(rich_table("Label sensitivity", ["label", "model", "Brier", "AUC", "accuracy", "share"], sens_rows))
    now_rows = []
    for position in ("1", "2", "3"):
        all_row = nowcast_all[position]
        ex_row = nowcast_ex_covid[position]
        now_rows.append(
            [
                position,
                all_row["n"],
                f"{all_row['bridge']:.4f}",
                f"{all_row['rw_drift']:.4f}",
                ex_row["n"],
                f"{ex_row['bridge']:.4f}",
                f"{ex_row['rw_drift']:.4f}",
            ]
        )
    console.print(
        rich_table(
            "Nowcast RMSE by month position (test block)",
            ["month", "n all", "bridge all", "RW all", "n ex-COVID", "bridge ex", "RW ex"],
            now_rows,
        )
    )
    coef_rows = [
        [
            row["term"],
            f"{row['estimate']:.4f}",
            f"{row['info_se']:.4f}",
            f"{row['nw_se']:.4f}",
            f"{row['nw_z']:.3f}",
        ]
        for row in coef_table
    ]
    console.print(
        rich_table(
            "Full-sample descriptive probit (information and Newey-West SE)",
            ["term", "estimate", "info SE", "NW SE", "NW z"],
            coef_rows,
        )
    )
    cluster_rows = []
    for index, window in enumerate(windows):
        ari_value = ari_rows[index]["ari"] if index < len(ari_rows) else float("nan")
        cluster_rows.append(
            [
                f"{window['end_date']:%Y-%m}",
                window["k"],
                f"{window['silhouette']:.3f}",
                len(window["series"]),
                "nan" if not np.isfinite(ari_value) else f"{ari_value:.3f}",
            ]
        )
    console.print(
        rich_table(
            "Clustering windows (reference-month series, descriptive only)",
            ["window end", "k", "silhouette", "series", "ARI to next"],
            cluster_rows,
        )
    )
    lead_clusters = sorted({row["cluster"] for row in lead_lag_rows})
    lead_table_rows = []
    for cluster in lead_clusters:
        values = [row for row in lead_lag_rows if row["cluster"] == cluster]
        by_lag = {row["lag"]: row["corr"] for row in values}
        lead_table_rows.append(
            [cluster] + [("nan" if not np.isfinite(by_lag[lag]) else f"{by_lag[lag]:.3f}") for lag in range(-4, 5)]
        )
    console.print(
        rich_table(
            "Lead-lag correlation of latest-window clusters with GDP",
            ["cluster"] + [str(lag) for lag in range(-4, 5)],
            lead_table_rows,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    start_time = time.perf_counter()
    np.random.seed(cfg.seed)
    raw, meta, lags, series = load_inputs(args.panel, args.meta)
    aligned = align_information_set(raw, lags)
    quarters = quarter_records(aligned)
    d = build_label(aligned, cfg.threshold, "two_month")
    T = len(aligned)
    origins = np.arange(cfg.min_train, T - 1)
    test_start = int(cfg.min_train + cfg.tuning_months)
    tuning_origins = origins[origins < test_start]
    test_origins = origins[origins >= test_start]
    if len(tuning_origins) != cfg.tuning_months:
        raise ValueError("tuning block does not match requested length")
    if len(test_origins) == 0:
        raise ValueError("empty test block")
    chosen_rank, rank_summary = select_rank_by_nowcast(aligned, quarters, tuning_origins, cfg)
    walk = walk_forward_probit(aligned, d, origins, cfg, chosen_rank, MAIN_CONTROLS, test_start)
    test_walk = walk[walk["block"] == "test"]
    probit_metrics = {
        model: classification_metrics(test_walk["y_true"], test_walk[f"p_{model}"])
        for model in ("probit", "prior", "persistence")
    }
    sensitivity = {}
    for name, threshold, rule in (
        (f"threshold_{cfg.sens_threshold:g}", cfg.sens_threshold, "two_month"),
        ("one_month", cfg.threshold, "one_month"),
    ):
        labels = build_label(aligned, threshold, rule)
        frame = walk_forward_probit(aligned, labels, origins, cfg, chosen_rank, MAIN_CONTROLS, test_start)
        test_frame = frame[frame["block"] == "test"]
        sensitivity[name] = {
            model: classification_metrics(test_frame["y_true"], test_frame[f"p_{model}"])
            for model in ("probit", "prior", "persistence")
        }
        sensitivity[name]["definition"] = {
            "threshold": threshold,
            "rule": rule,
            "test_class_share": float(np.nanmean(labels[test_origins + 1])),
        }
    shibor_walk = walk_forward_probit(
        aligned, d, origins, cfg, chosen_rank, MAIN_CONTROLS + (SHIBOR_CONTROL,), test_start
    )
    shibor_test = shibor_walk[shibor_walk["block"] == "test"]
    shibor_metrics = {
        model: classification_metrics(shibor_test["y_true"], shibor_test[f"p_{model}"])
        for model in ("probit", "prior", "persistence")
    }
    nowcast_origins = np.arange(cfg.min_train, T)
    nowcast_paths = run_nowcast(aligned, quarters, nowcast_origins, cfg, chosen_rank, test_start)
    nowcast_all = nowcast_rmse_table(nowcast_paths, exclude_covid=False)
    nowcast_ex_covid = nowcast_rmse_table(nowcast_paths, exclude_covid=True)
    descriptive = descriptive_pca(aligned, cfg)
    coefficient_table = descriptive_probit(aligned, d, cfg, chosen_rank, MAIN_CONTROLS)
    cluster_labels, windows = cluster_reference_panel(raw, cfg)
    ari_rows = adjacent_ari(windows)
    if windows:
        lead_lag_rows, members = lead_lag_table(raw, windows[-1])
    else:
        lead_lag_rows, members = [], {}
    elapsed = time.perf_counter() - start_time
    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_dates = {
        "origin_start": str(test_walk["origin_date"].iloc[0].date()),
        "origin_end": str(test_walk["origin_date"].iloc[-1].date()),
        "target_start": str(test_walk["target_date"].iloc[0].date()),
        "target_end": str(test_walk["target_date"].iloc[-1].date()),
        "origins": int(len(test_walk)),
    }
    params = {
        "approach": "Q3A",
        "seed": cfg.seed,
        "panel": str(args.panel.relative_to(ROOT)) if args.panel.is_relative_to(ROOT) else str(args.panel),
        "meta": str(args.meta.relative_to(ROOT)) if args.meta.is_relative_to(ROOT) else str(args.meta),
        "config": asdict(cfg),
        "label": {
            "definition": "d_t = 1{PMI_t < threshold and PMI_{t-1} < threshold}",
            "threshold": cfg.threshold,
            "rule": "two_month",
            "justification": "two consecutive prints reduce noise around the 50 boundary",
            "sensitivity": {
                "threshold": cfg.sens_threshold,
                "one_month_rule": "d_t = 1{PMI_t < threshold}",
            },
        },
        "information_set": "F_t = {x_{i,s}: s + L_i <= t}; one alignment function shifts each series by L_i",
        "preprocessing": "standardisation and PCA use only rows public at the origin; missing values become 0 after standardisation; series with below 50% training-window coverage are dropped",
        "lags": lags,
        "factor_exclusions": list(FACTOR_EXCLUSIONS),
        "controls": list(MAIN_CONTROLS),
        "shibor_sensitivity": SHIBOR_CONTROL,
        "validation": {
            "scheme": "expanding walk-forward, refit monthly",
            "min_train_months": cfg.min_train,
            "tuning_months": cfg.tuning_months,
            "rank_criterion": "walk-forward CV nowcast RMSE at month 3 on the tuning block",
            "test_block_only_for_reported_metrics": True,
            "test_dates": test_dates,
        },
        "clustering": {
            "note": "clustering uses reference-month series, studies co-movement, and is not used for forecasting",
            "window_months": cfg.cluster_window,
            "step_months": cfg.cluster_step,
            "distance": "sqrt(2*(1-rho))",
            "linkage": "Ward",
            "k_selection": "maximum silhouette",
        },
    }
    metrics = {
        "label": {
            "full_sample_regime_share": float(np.nanmean(d)),
            "test_class_share": float(probit_metrics["probit"]["class_share"]),
            "test_positives": int(probit_metrics["probit"]["positives"]),
        },
        "validation": {
            "tuning_origin_start": str(aligned["date"].iloc[tuning_origins[0]].date()),
            "tuning_origin_end": str(aligned["date"].iloc[tuning_origins[-1]].date()),
            "test_origin_start": test_dates["origin_start"],
            "test_origin_end": test_dates["origin_end"],
            "test_target_start": test_dates["target_start"],
            "test_target_end": test_dates["target_end"],
            "test_origins": test_dates["origins"],
            "test_target_dates": [str(date.date()) for date in test_walk["target_date"]],
        },
        "rank_selection": {
            "chosen_rank": int(chosen_rank),
            "criterion": "walk-forward CV nowcast RMSE at month 3 on the tuning block",
            "table": rank_summary.to_dict(orient="records"),
        },
        "factor_pca": {
            "explained_variance_ratio": descriptive["explained_variance_ratio"],
            "top_loadings": descriptive["top_loadings"],
            "n_series": len(descriptive["pca"]["series"]),
            "series": descriptive["pca"]["series"],
            "descriptive_full_sample": True,
        },
        "probit": {
            "test": probit_metrics,
            "threshold_sensitivity": sensitivity,
            "shibor_sensitivity": {
                "controls": list(MAIN_CONTROLS) + [SHIBOR_CONTROL],
                "test": shibor_metrics,
            },
            "coefficients_full_sample": coefficient_table,
        },
        "nowcast": {
            "test_rmse_by_month": nowcast_all,
            "test_rmse_ex_covid": nowcast_ex_covid,
            "covid_exclusion": "target quarters 2020Q1-2021Q4 excluded",
            "benchmark": "random walk with drift estimated from the same information set",
        },
        "clustering": {
            "note": "reference-month series, descriptive co-movement study, not used for forecasting",
            "windows": [
                {
                    "start_date": str(window["start_date"].date()),
                    "end_date": str(window["end_date"].date()),
                    "k": int(window["k"]),
                    "silhouette": float(window["silhouette"]),
                    "silhouette_by_k": window["silhouette_by_k"],
                    "series": window["series"],
                    "clusters": {
                        str(label): [
                            series
                            for series, assigned in zip(window["series"], window["labels"])
                            if int(assigned) == label
                        ]
                        for label in sorted(set(window["labels"]))
                    },
                }
                for window in windows
            ],
            "adjacent_ari": [
                {
                    "window_left": str(row["window_left"].date()),
                    "window_right": str(row["window_right"].date()),
                    "n_common": int(row["n_common"]),
                    "ari": row["ari"],
                }
                for row in ari_rows
            ],
            "latest_window_members": {str(label): names for label, names in members.items()},
            "lead_lag_with_gdp": lead_lag_rows,
        },
    }
    history = {
        "timings": {"total_seconds": elapsed},
        "origins": [
            {
                "origin_index": int(row.origin_index),
                "block": row.block,
                "rank": int(row.rank),
                "n_train": int(row.n_train),
                "converged": bool(row.converged),
            }
            for row in walk.itertuples()
        ],
        "warnings": [],
        "clustering_windows": len(windows),
    }
    result = {
        "params": to_jsonable(params),
        "metrics": to_jsonable(metrics),
        "fit_seconds": float(elapsed),
        "history": to_jsonable(history),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    walk.to_csv(args.out_dir / "probit_walkforward.csv", index=False)
    nowcast_paths.to_csv(args.out_dir / "nowcast_paths.csv", index=False)
    loadings_rows = []
    pca = descriptive["pca"]
    for component in range(pca["n_components"]):
        for series_name, loading in zip(pca["series"], pca["loadings"][component]):
            loadings_rows.append(
                {
                    "component": component + 1,
                    "series": series_name,
                    "loading": float(loading),
                    "explained_variance_ratio": float(pca["explained_variance_ratio"][component]),
                }
            )
    pd.DataFrame(loadings_rows).to_csv(args.out_dir / "loadings.csv", index=False)
    cluster_labels.to_csv(args.out_dir / "cluster_labels.csv", index=False)
    write_html(regime_figure(walk), args.out_dir / "regime.html")
    write_html(stability_figure(windows, ari_rows), args.out_dir / "stability.html")
    if windows:
        write_html(dendrogram_figure(windows[-1]), args.out_dir / "dendrogram.html")
    write_html(nowcast_figure(nowcast_paths), args.out_dir / "nowcast_revision.html")
    print_report(
        lags,
        cfg,
        test_walk["origin_date"].dt.date.to_numpy(),
        test_walk["target_date"].dt.date.to_numpy(),
        rank_summary,
        chosen_rank,
        probit_metrics,
        sensitivity,
        nowcast_all,
        nowcast_ex_covid,
        coefficient_table["coefficients"],
        windows,
        ari_rows,
        lead_lag_rows,
    )
    (CACHE / "q3a_run.log").parent.mkdir(parents=True, exist_ok=True)
    (CACHE / "q3a_run.log").write_text(console.export_text(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
