#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import approach5_q3a_classical as q3

RESULTS = ROOT / "results" / "approach5" / "macro"
METRICS = RESULTS / "metrics.json"
WALK = RESULTS / "probit_walkforward.csv"
NOWCAST = RESULTS / "nowcast_paths.csv"
RIDGE_GRID = (1e-4, 1e-2, 1e-1, 1.0, 10.0)
BOOTSTRAP_REPS = 5000
BOOTSTRAP_BLOCK = 12
BOOTSTRAP_SEED = 42
CONCRETE_QUARTER = "2025Q2"


def fit_origin_posthoc(
    aligned: pd.DataFrame,
    d: np.ndarray,
    t: int,
    cfg: q3.RunConfig,
    rank: int,
    controls: tuple[str, ...],
    ridge: float,
    include_dt: bool,
) -> dict:
    prep = q3.prepare_origin(aligned, t, cfg, controls, rank)
    train_idx = np.arange(0, t)
    scores = prep["pca"]["scores_apply"]
    use_rank = prep["rank"]
    control_train = q3.apply_standardizer(
        aligned, train_idx, prep["control_series"], prep["control_means"], prep["control_stds"]
    )
    control_t = q3.apply_standardizer(
        aligned, np.array([t]), prep["control_series"], prep["control_means"], prep["control_stds"]
    )
    x_train = np.column_stack([np.ones(len(train_idx)), scores[train_idx, :use_rank], control_train])
    x_t = np.concatenate([[1.0], scores[t, :use_rank], control_t[0]])
    if include_dt:
        x_train = np.column_stack([x_train, d[train_idx]])
        x_t = np.concatenate([x_t, [d[t]]])
    y_train = d[train_idx + 1]
    mask = np.isfinite(y_train)
    if include_dt:
        mask &= np.isfinite(d[train_idx])
    fit = q3.fit_probit(x_train[mask], y_train[mask], ridge=ridge)
    probability = float(q3.ndtr(float(x_t @ fit["beta"])))
    return {
        "probability": probability,
        "converged": bool(fit["converged"]),
        "n_train": int(mask.sum()),
        "rank": int(use_rank),
    }


def walk_predictions(
    aligned: pd.DataFrame,
    d: np.ndarray,
    origins: np.ndarray,
    cfg: q3.RunConfig,
    rank: int,
    ridge: float,
    include_dt: bool,
) -> pd.DataFrame:
    rows = []
    for t in origins:
        result = fit_origin_posthoc(aligned, d, int(t), cfg, rank, q3.MAIN_CONTROLS, ridge, include_dt)
        rows.append(
            {
                "origin_index": int(t),
                "origin_date": pd.Timestamp(aligned["date"].iloc[t]),
                "p": result["probability"],
                "y": float(d[t + 1]),
                "converged": result["converged"],
                "n_train": result["n_train"],
                "rank": result["rank"],
            }
        )
    return pd.DataFrame(rows)


def select_ridge(
    aligned: pd.DataFrame,
    d: np.ndarray,
    tuning_origins: np.ndarray,
    cfg: q3.RunConfig,
    rank: int,
    include_dt: bool,
) -> tuple[float, pd.DataFrame]:
    rows = []
    predictions = {}
    for ridge in RIDGE_GRID:
        frame = walk_predictions(aligned, d, tuning_origins, cfg, rank, ridge, include_dt)
        predictions[ridge] = frame
        rows.append(
            {
                "ridge": float(ridge),
                "tuning_brier": float(np.mean((frame["p"].to_numpy() - frame["y"].to_numpy()) ** 2)),
                "tuning_auc": q3.auc_score(frame["y"], frame["p"]),
                "tuning_accuracy": float(np.mean((frame["p"].to_numpy() >= 0.5) == (frame["y"].to_numpy() == 1.0))),
                "converged": int(frame["converged"].sum()),
                "n": int(len(frame)),
            }
        )
    table = pd.DataFrame(rows).sort_values(["tuning_brier", "ridge"]).reset_index(drop=True)
    chosen = float(table.iloc[0]["ridge"])
    return chosen, table


def extreme_share(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean((values > 0.99) | (values < 0.01)))


def moving_block_indices(n: int, block: int, reps: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    max_start = n - block
    starts = rng.integers(0, max_start + 1, size=(reps, n_blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(reps, -1)[:, :n]
    return [indices[rep] for rep in range(reps)]


def bootstrap_auc_differences(
    y: np.ndarray,
    model_probs: dict[str, np.ndarray],
    persistence: np.ndarray,
    block: int,
    reps: int,
    seed: int,
) -> dict:
    n = len(y)
    indices = moving_block_indices(n, block, reps, seed)
    result = {}
    base_auc = q3.auc_score(y, persistence)
    for name, probability in model_probs.items():
        differences = []
        for index in indices:
            yy = y[index]
            if len(np.unique(yy)) < 2:
                continue
            model_auc = q3.auc_score(yy, probability[index])
            persistence_auc = q3.auc_score(yy, persistence[index])
            if np.isfinite(model_auc) and np.isfinite(persistence_auc):
                differences.append(model_auc - persistence_auc)
        differences = np.asarray(differences)
        result[name] = {
            "point_auc": q3.auc_score(y, probability),
            "persistence_auc": base_auc,
            "point_difference": q3.auc_score(y, probability) - base_auc,
            "ci95_low": float(np.percentile(differences, 2.5)),
            "ci95_high": float(np.percentile(differences, 97.5)),
            "n_valid_replicates": int(len(differences)),
        }
    return result


def paired_nowcast_check(nowcast: pd.DataFrame, aligned: pd.DataFrame) -> dict:
    test = nowcast[(nowcast["block"] == "test") & nowcast["actual_gdp"].notna()].copy()
    counts = test.groupby("quarter")["month_position"].nunique()
    complete = [quarter for quarter, count in counts.items() if count == 3]
    paired = test[test["quarter"].isin(complete)].copy()
    year = paired["quarter"].str.slice(0, 4).astype(int)
    paired = paired[~year.isin([2020, 2021])].copy()
    rmse_by_position = {}
    for position in (1, 2, 3):
        subset = paired[paired["month_position"] == position]
        rmse_by_position[str(position)] = {
            "n_quarters": int(len(subset)),
            "bridge": q3.rmse(subset["actual_gdp"], subset["nowcast_bridge"]),
            "rw_drift": q3.rmse(subset["actual_gdp"], subset["nowcast_rw_drift"]),
        }
    wide = paired.pivot(index="quarter", columns="month_position", values=["error_bridge", "error_rw_drift", "actual_gdp"])
    wide = wide.dropna()
    se1 = (wide[("error_bridge", 1)].to_numpy() ** 2)
    se3 = (wide[("error_bridge", 3)].to_numpy() ** 2)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reps = 5000
    diffs = []
    for _ in range(reps):
        idx = rng.integers(0, len(se1), size=len(se1))
        diffs.append(math.sqrt(float(np.mean(se3[idx]))) - math.sqrt(float(np.mean(se1[idx]))))
    diffs = np.asarray(diffs)
    concrete = concrete_quarter_inputs(aligned, nowcast, CONCRETE_QUARTER)
    return {
        "paired_ex_covid": True,
        "n_quarters_with_all_three_positions": int(len(wide)),
        "rmse_by_position": rmse_by_position,
        "paired_rmse_month3_minus_month1": float(
            math.sqrt(float(np.mean(se3))) - math.sqrt(float(np.mean(se1)))
        ),
        "paired_rmse_difference_ci95": [
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)),
        ],
        "concrete_quarter": concrete,
        "assessment": "not a data-flow bug: month 3 uses strictly more aligned monthly factor rows and more public rows; the ex-COVID RMSE increase is small and its paired quarter-bootstrap interval is reported",
    }


def concrete_quarter_inputs(aligned: pd.DataFrame, nowcast: pd.DataFrame, quarter: str) -> dict:
    periods = aligned["date"].dt.to_period("Q").astype(str)
    rows = np.flatnonzero((periods == quarter).to_numpy())
    start = int(rows.min())
    output = {"quarter": quarter, "positions": []}
    for position in (1, 2, 3):
        t = start + position - 1
        months = [str(date.date()) for date in aligned["date"].iloc[start : t + 1]]
        factor_series = q3.select_factor_series(
            aligned, np.arange(0, t + 1), q3.FACTOR_EXCLUSIONS, 0.5
        )
        match = nowcast[(nowcast["quarter"] == quarter) & (nowcast["month_position"] == position)]
        n_training = int(match["n_training_quarters"].iloc[0]) if len(match) else None
        output["positions"].append(
            {
                "month_position": position,
                "origin_date": str(aligned["date"].iloc[t].date()),
                "aligned_input_months": months,
                "n_aligned_input_months": len(months),
                "n_public_rows_for_pca": t + 1,
                "n_factor_series": len(factor_series),
                "n_training_quarters": n_training,
            }
        )
    return output


def main() -> int:
    start = time.perf_counter()
    original = json.loads(METRICS.read_text(encoding="utf-8"))
    original_probit_block = json.loads(json.dumps(original["metrics"]["probit"]))
    cfg = q3.RunConfig(**original["params"]["config"])
    rank = int(original["metrics"]["rank_selection"]["chosen_rank"])
    raw, meta, lags, series = q3.load_inputs(q3.DEFAULT_PANEL, q3.DEFAULT_META)
    aligned = q3.align_information_set(raw, lags)
    d = q3.build_label(aligned, cfg.threshold, "two_month")
    T = len(aligned)
    test_start = int(cfg.min_train + cfg.tuning_months)
    tuning_origins = np.arange(cfg.min_train, test_start)
    test_origins = np.arange(test_start, T - 1)
    walk = pd.read_csv(WALK, parse_dates=["origin_date", "target_date"])
    walk = walk.sort_values("origin_index").reset_index(drop=True)
    test_walk = walk[walk["block"] == "test"].sort_values("origin_index").reset_index(drop=True)
    y = test_walk["y_true"].to_numpy(dtype=float)
    prior = test_walk["p_prior"].to_numpy(dtype=float)
    persistence = test_walk["p_persistence"].to_numpy(dtype=float)
    original_prob = test_walk["p_probit"].to_numpy(dtype=float)

    ridge_specs = {}
    test_predictions = {"original": original_prob}
    for name, include_dt in (("ridge_selected", False), ("ridge_selected_with_d_t", True)):
        chosen_ridge, tuning_table = select_ridge(
            aligned, d, tuning_origins, cfg, rank, include_dt
        )
        test_frame = walk_predictions(
            aligned, d, test_origins, cfg, rank, chosen_ridge, include_dt
        )
        test_frame = test_frame.sort_values("origin_index").reset_index(drop=True)
        if not np.array_equal(test_frame["y"].to_numpy(), y):
            raise ValueError("post-hoc test labels do not match original walk-forward labels")
        probability = test_frame["p"].to_numpy(dtype=float)
        metrics = q3.classification_metrics(y, probability)
        ridge_specs[name] = {
            "include_d_t": bool(include_dt),
            "extension_beyond_brief": bool(include_dt),
            "ridge_grid": [float(value) for value in RIDGE_GRID],
            "chosen_ridge": float(chosen_ridge),
            "tuning_table": tuning_table.to_dict(orient="records"),
            "test_metrics": metrics,
            "share_extreme_0p99_0p01": extreme_share(probability),
            "converged_origins": int(test_frame["converged"].sum()),
            "n_test_origins": int(len(test_frame)),
        }
        test_predictions[name] = probability

    original_metrics = q3.classification_metrics(y, original_prob)
    if not np.isclose(original_metrics["Brier"], original["metrics"]["probit"]["test"]["probit"]["Brier"]):
        raise ValueError("original probit Brier changed")
    if not np.isclose(original_metrics["AUC"], original["metrics"]["probit"]["test"]["probit"]["AUC"]):
        raise ValueError("original probit AUC changed")

    model_probs = {
        "original_probit": original_prob,
        "ridge_selected_probit": test_predictions["ridge_selected"],
        "ridge_selected_probit_with_d_t": test_predictions["ridge_selected_with_d_t"],
    }
    bootstrap = bootstrap_auc_differences(
        y, model_probs, persistence, BOOTSTRAP_BLOCK, BOOTSTRAP_REPS, BOOTSTRAP_SEED
    )
    table = []
    for name, probability in (
        ("original_probit", original_prob),
        ("ridge_selected_probit", test_predictions["ridge_selected"]),
        ("ridge_selected_probit_with_d_t", test_predictions["ridge_selected_with_d_t"]),
        ("prior", prior),
        ("persistence", persistence),
    ):
        metrics = q3.classification_metrics(y, probability)
        table.append(
            {
                "model": name,
                "Brier": metrics["Brier"],
                "AUC": metrics["AUC"],
                "accuracy": metrics["accuracy"],
                "share_extreme_0p99_0p01": extreme_share(probability),
                "n": metrics["n"],
                "positives": metrics["positives"],
            }
        )

    nowcast = pd.read_csv(NOWCAST, parse_dates=["origin_date"])
    nowcast_check = paired_nowcast_check(nowcast, aligned)
    post_hoc = {
        "disclosure": "POST-HOC: the test block had already been seen before this robustness check; these are disclosed post-hoc results, not pre-registered test results.",
        "label": "post-hoc robustness check",
        "rank_kept": int(rank),
        "ridge_selection": ridge_specs,
        "test_table": table,
        "auc_difference_vs_persistence": bootstrap,
        "bootstrap": {
            "method": "moving block bootstrap of 12 months",
            "block_length_months": BOOTSTRAP_BLOCK,
            "replicates_requested": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "paired_across_models": True,
            "note": "resamples with only one class are skipped",
        },
        "nowcast_check": nowcast_check,
        "elapsed_seconds": float(time.perf_counter() - start),
    }
    original["metrics"]["post_hoc"] = q3.to_jsonable(post_hoc)
    if original["metrics"]["probit"] != original_probit_block:
        raise ValueError("original probit metrics block changed")
    if list(original.keys()) != ["params", "metrics", "fit_seconds", "history"]:
        raise ValueError("top-level keys changed")
    METRICS.write_text(json.dumps(q3.to_jsonable(original), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(q3.to_jsonable(post_hoc), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
