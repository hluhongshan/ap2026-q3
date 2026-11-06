#!/usr/bin/env python3
"""Post-hoc diagnostics and retraining for Q3 Option B."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import approach1_q3a_classical as q3
import approach2_q3b_sequence as q3b

CACHE = ROOT / ".cache"
RESULTS = ROOT / "results" / "approach2" / "macro"
METRICS = RESULTS / "metrics.json"
DIAG_JSON = CACHE / "q3b_diagnostics.json"
DIAG_LOG = CACHE / "q3b_diagnostics.log"
RETRAIN_LOG = CACHE / "q3b_posthoc_retrain.log"

CURRENT_HP = {"hidden": 16, "layers": 1, "dropout": 0.2}
GRID = (
    {"hidden": 8, "layers": 1, "dropout": 0.0, "weight_decay": 1e-3},
    {"hidden": 8, "layers": 1, "dropout": 0.2, "weight_decay": 1e-3},
    {"hidden": 8, "layers": 1, "dropout": 0.2, "weight_decay": 1e-2},
    {"hidden": 16, "layers": 1, "dropout": 0.0, "weight_decay": 1e-3},
    {"hidden": 16, "layers": 1, "dropout": 0.2, "weight_decay": 1e-3},
    {"hidden": 16, "layers": 1, "dropout": 0.2, "weight_decay": 1e-2},
)
LEAK_PAIRS = (
    ("lstm_lag_discipline", "lstm_zero_lag", "lstm_disciplined"),
    ("lstm_full_sample_standardisation", "lstm_zero_lag_full_sample", "lstm_disciplined"),
    ("lstm_standardisation_only", "lstm_zero_lag_full_sample", "lstm_zero_lag"),
    ("probit_lag_discipline", "probit_zero_lag", "probit"),
)
ARM_ORDER = (
    "probit",
    "lstm_disciplined_original",
    "lstm_disciplined",
    "lstm_zero_lag",
    "lstm_zero_lag_full_sample",
    "probit_zero_lag",
    "prior",
    "persistence",
)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.use_deterministic_algorithms(True, warn_only=True)

console = Console(record=True)


class Log:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.start = time.perf_counter()

    def write(self, message: str) -> None:
        elapsed = time.perf_counter() - self.start
        line = f"[{elapsed:8.2f}s] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        console.print(line)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q3B post-hoc diagnostics and retraining")
    parser.add_argument("--mode", choices=["diagnose", "retrain", "all"], default="diagnose")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel", type=Path, default=q3b.DEFAULT_PANEL)
    parser.add_argument("--meta", type=Path, default=q3b.DEFAULT_META)
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument("--tuning-months", type=int, default=36)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--val-months", type=int, default=36)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--time-budget-minutes", type=float, default=30.0)
    return parser.parse_args(argv)


def current_args() -> argparse.Namespace:
    args = q3b.parse_args([])
    args.max_epochs = 300
    args.patience = 30
    args.val_months = 18
    args.min_epochs = 0
    return args


def current_cfg(args: argparse.Namespace) -> q3.RunConfig:
    return q3b.q3_config(args)


def train_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hp: dict,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    track_train_auc: bool = True,
) -> tuple[q3b.LSTMClassifier, dict]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    model = q3b.LSTMClassifier(
        input_dim=int(X_train.shape[2]),
        hidden=int(hp["hidden"]),
        layers=int(hp["layers"]),
        dropout=float(hp["dropout"]),
    )
    weight, n_pos, n_neg = q3b.positive_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_train = torch.as_tensor(X_train, dtype=torch.float32)
    t_train = torch.as_tensor(y_train, dtype=torch.float32)
    x_val = torch.as_tensor(X_val, dtype=torch.float32)
    t_val = torch.as_tensor(y_val, dtype=torch.float32)
    history = {
        "epoch": [],
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": [],
    }
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x_train)
        loss = criterion(logits, t_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_logits, _ = model(x_train)
            val_logits, _ = model(x_val)
            train_prob = torch.sigmoid(train_logits).numpy()
            val_prob = torch.sigmoid(val_logits).numpy()
            val_loss = float(
                nn.functional.binary_cross_entropy_with_logits(val_logits, t_val).item()
            )
        history["epoch"].append(int(epoch))
        history["train_loss"].append(float(loss.item()))
        history["train_auc"].append(
            q3.auc_score(y_train, train_prob) if track_train_auc else float("nan")
        )
        history["val_loss"].append(val_loss)
        history["val_auc"].append(q3.auc_score(y_val, val_prob))
    return model, history


def alignment_diagnostics(
    raw: pd.DataFrame,
    lags: dict[str, int],
    aligned: pd.DataFrame,
    d: np.ndarray,
    origins: list[int],
) -> list[dict]:
    rows = []
    for t in origins:
        last_window = int(t)
        target = int(t + 1)
        pmi_now = float(raw["pmi"].iloc[last_window])
        pmi_next = float(raw["pmi"].iloc[target])
        pmi_previous = float(raw["pmi"].iloc[last_window - 1])
        expected_now = float(pmi_now < 50.0 and pmi_previous < 50.0)
        expected_next = float(pmi_next < 50.0 and pmi_now < 50.0)
        rows.append(
            {
                "origin_index": int(t),
                "origin_date": str(aligned["date"].iloc[t].date()),
                "window_start_date": str(aligned["date"].iloc[t - 11].date()),
                "window_end_date": str(aligned["date"].iloc[last_window].date()),
                "target_index": target,
                "target_date": str(aligned["date"].iloc[target].date()),
                "target_is_t_plus_1": bool(target == t + 1),
                "d_t": float(d[t]),
                "d_t_plus_1": float(d[target]),
                "pmi_t": pmi_now,
                "pmi_t_plus_1": pmi_next,
                "label_check_t": bool(np.isclose(d[t], expected_now)),
                "label_check_t_plus_1": bool(np.isclose(d[target], expected_next)),
                "pmi_lag": int(lags["pmi"]),
            }
        )
    return rows


def shifted_label_test(
    aligned: pd.DataFrame,
    d: np.ndarray,
    cfg: q3.RunConfig,
    args: argparse.Namespace,
    t: int,
) -> dict:
    means, stds, columns = q3b.fit_feature_scaler(aligned, np.arange(0, t + 1), cfg.min_coverage)
    z = q3.apply_standardizer(aligned, np.arange(0, t + 1), columns, means, stds)
    ends = q3b.training_pair_ends(t, d, args.window)
    ends = ends[np.isfinite(d[ends])]
    features = np.stack([z[end - args.window + 1 : end + 1] for end in ends], axis=0)
    labels = d[ends].astype(float)
    hp = {"hidden": 16, "layers": 1, "dropout": 0.0}
    args_fixed = q3b.parse_args([])
    model, history = train_fixed_epochs(
        features,
        labels,
        features,
        labels,
        hp,
        args.seed,
        300,
        args_fixed.lr,
        args_fixed.weight_decay,
    )
    p_cal, p_raw, _ = q3b.predict_lstm(model, features, q3b.positive_weight(labels)[0])
    return {
        "origin_index": int(t),
        "n_pairs": int(len(ends)),
        "n_pos": int(np.sum(labels == 1.0)),
        "n_neg": int(np.sum(labels == 0.0)),
        "train_auc_shifted_label": q3.auc_score(labels, p_raw),
        "train_accuracy_shifted_label": float(np.mean((p_cal >= 0.5) == (labels == 1.0))),
        "final_train_loss": history["train_loss"][-1],
        "final_train_auc": history["train_auc"][-1],
        "epoch_train_auc": {
            str(epoch): history["train_auc"][epoch - 1]
            for epoch in (1, 10, 50, 100, 200, 300)
        },
    }


def overfit_test(
    aligned: pd.DataFrame,
    d: np.ndarray,
    cfg: q3.RunConfig,
    args: argparse.Namespace,
    t: int,
) -> dict:
    means, stds, columns = q3b.fit_feature_scaler(aligned, np.arange(0, t + 1), cfg.min_coverage)
    data = q3b.build_training_arrays(
        aligned, d, t, args.window, 18, columns, means, stds
    )
    hp = {"hidden": 16, "layers": 1, "dropout": 0.0}
    args_fixed = q3b.parse_args([])
    model, history = train_fixed_epochs(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        hp,
        args.seed,
        300,
        args_fixed.lr,
        args_fixed.weight_decay,
    )
    checkpoints = {}
    for epoch in (1, 10, 25, 50, 100, 150, 200, 250, 300):
        checkpoints[str(epoch)] = {
            "train_loss": history["train_loss"][epoch - 1],
            "train_auc": history["train_auc"][epoch - 1],
            "val_loss": history["val_loss"][epoch - 1],
            "val_auc": history["val_auc"][epoch - 1],
        }
    return {
        "origin_index": int(t),
        "n_train": int(len(data["y_train"])),
        "n_val": int(len(data["y_val"])),
        "n_pos_train": int(np.sum(data["y_train"] == 1.0)),
        "n_neg_train": int(np.sum(data["y_train"] == 0.0)),
        "n_pos_val": int(np.sum(data["y_val"] == 1.0)),
        "n_neg_val": int(np.sum(data["y_val"] == 0.0)),
        "checkpoints": checkpoints,
        "final": {
            "train_loss": history["train_loss"][-1],
            "train_auc": history["train_auc"][-1],
            "val_loss": history["val_loss"][-1],
            "val_auc": history["val_auc"][-1],
        },
        "history": history,
    }


def early_stopping_diagnostics(
    aligned: pd.DataFrame,
    d: np.ndarray,
    cfg: q3.RunConfig,
    args: argparse.Namespace,
    refits: list[int],
) -> list[dict]:
    rows = []
    for t in refits:
        means, stds, columns = q3b.fit_feature_scaler(aligned, np.arange(0, t + 1), cfg.min_coverage)
        z = q3.apply_standardizer(aligned, np.arange(0, t + 1), columns, means, stds)
        ends = q3b.training_pair_ends(t, d, args.window)
        features = np.stack([z[end - args.window + 1 : end + 1] for end in ends], axis=0)
        labels = d[ends + 1].astype(float)
        val_labels = labels[-18:]
        rows.append(
            {
                "refit_origin_index": int(t),
                "n_pairs": int(len(ends)),
                "n_train": int(len(labels) - 18),
                "n_val": int(len(val_labels)),
                "val_pos": int(np.sum(val_labels == 1.0)),
                "val_neg": int(np.sum(val_labels == 0.0)),
                "val_share": float(np.mean(val_labels)),
                "val_single_class": bool(len(np.unique(val_labels)) == 1),
            }
        )
    return rows


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    return np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, block: int, reps: int, seed: int) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = []
    attempts = 0
    while len(draws) < int(reps) and attempts < int(reps) * 20:
        attempts += 1
        idx = circular_block_indices(len(y), block, rng)
        value = q3.auc_score(y[idx], p[idx])
        if np.isfinite(value):
            draws.append(value)
    draws = np.asarray(draws, dtype=float)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "auc": q3.auc_score(y, p),
        "ci_low": float(low),
        "ci_high": float(high),
        "valid_replicates": int(len(draws)),
        "block_months": int(block),
    }


def bootstrap_auc_difference(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    block: int,
    reps: int,
    seed: int,
) -> dict:
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = []
    attempts = 0
    while len(draws) < int(reps) and attempts < int(reps) * 20:
        attempts += 1
        idx = circular_block_indices(len(y), block, rng)
        auc_a = q3.auc_score(y[idx], p_a[idx])
        auc_b = q3.auc_score(y[idx], p_b[idx])
        if np.isfinite(auc_a) and np.isfinite(auc_b):
            draws.append(auc_a - auc_b)
    draws = np.asarray(draws, dtype=float)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "difference": q3.auc_score(y, p_a) - q3.auc_score(y, p_b),
        "ci_low": float(low),
        "ci_high": float(high),
        "valid_replicates": int(len(draws)),
    }


def tuning_diagnostics(
    aligned: pd.DataFrame,
    d: np.ndarray,
    cfg: q3.RunConfig,
    args: argparse.Namespace,
    tuning_origins: np.ndarray,
    hp: dict,
) -> dict:
    seeds = [int(args.seed) + offset for offset in range(3)]
    predictions = {}
    for seed in seeds:
        run = q3b.run_lstm_walk(
            aligned, d, tuning_origins, cfg, seed, hp, args, "diagnostic_tuning", log=None
        )
        frame = run["frame"]
        predictions[seed] = frame["p"].to_numpy(dtype=float)
        console.print(
            f"tuning diagnostic seed {seed}: Brier={q3.classification_metrics(frame['y_true'], frame['p'])['Brier']:.4f} "
            f"AUC={q3.classification_metrics(frame['y_true'], frame['p'])['AUC']:.4f}"
        )
    frame = run["frame"]
    y = frame["y_true"].to_numpy(dtype=float)
    p_mean = np.nanmean(np.vstack([predictions[seed] for seed in seeds]), axis=0)
    base_rates = []
    years = pd.DatetimeIndex(frame["target_date"]).year
    for year in sorted(set(years)):
        mask = years == year
        base_rates.append(
            {
                "year": int(year),
                "n": int(mask.sum()),
                "regime_share": float(np.mean(y[mask])),
                "positives": int(np.sum(y[mask] == 1.0)),
            }
        )
    persistence = frame["d_t"].to_numpy(dtype=float)
    lstm_boot = bootstrap_auc_ci(y, p_mean, args.bootstrap_block, args.bootstrap_reps, args.seed)
    persistence_boot = bootstrap_auc_ci(
        y, persistence, args.bootstrap_block, args.bootstrap_reps, args.seed + 1
    )
    difference = bootstrap_auc_difference(
        y, p_mean, persistence, args.bootstrap_block, args.bootstrap_reps, args.seed + 2
    )
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = {
            "AUC": q3.auc_score(y, predictions[seed]),
            "Brier": float(np.mean((predictions[seed] - y) ** 2)),
            "AUC_inverted": q3.auc_score(y, 1.0 - predictions[seed]),
        }
    return {
        "n_months": int(len(y)),
        "positives": int(np.sum(y == 1.0)),
        "regime_share": float(np.mean(y)),
        "per_seed": per_seed,
        "seed_mean": {
            "AUC": q3.auc_score(y, p_mean),
            "Brier": float(np.mean((p_mean - y) ** 2)),
        },
        "base_rates_by_year": base_rates,
        "persistence": {
            "AUC": q3.auc_score(y, persistence),
            "Brier": float(np.mean((persistence - y) ** 2)),
            **persistence_boot,
        },
        "lstm_bootstrap": lstm_boot,
        "lstm_minus_persistence": difference,
    }


def rich_table(title: str, columns: list[str], rows: list[list]) -> Table:
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*[("" if value is None else str(value)) for value in row])
    return table


def run_diagnostics(args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw, meta, lags, series = q3.load_inputs(args.panel, args.meta)
    aligned = q3.align_information_set(raw, lags)
    cfg = current_cfg(args)
    d = q3.build_label(aligned, cfg.threshold, "two_month")
    T = len(aligned)
    origins = np.arange(cfg.min_train, T - 1)
    test_start = int(cfg.min_train + cfg.tuning_months)
    tuning_origins = origins[origins < test_start]
    refits = q3b.refit_schedule(origins, 6)
    current = current_args()
    current.window = args.window if hasattr(args, "window") else 12
    current.bootstrap_reps = args.bootstrap_reps
    current.bootstrap_block = args.bootstrap_block

    console.rule("Q3B post-hoc diagnostics")
    alignment = alignment_diagnostics(raw, lags, aligned, d, [96, 120, 198])
    console.print(
        rich_table(
            "Alignment at three origins",
            [
                "origin t",
                "origin date",
                "window start",
                "last window month",
                "target index",
                "target month",
                "d_t",
                "d_(t+1)",
                "pmi_t",
                "pmi_(t+1)",
            ],
            [
                [
                    row["origin_index"],
                    row["origin_date"],
                    row["window_start_date"],
                    row["window_end_date"],
                    row["target_index"],
                    row["target_date"],
                    f"{row['d_t']:.0f}",
                    f"{row['d_t_plus_1']:.0f}",
                    f"{row['pmi_t']:.2f}",
                    f"{row['pmi_t_plus_1']:.2f}",
                ]
                for row in alignment
            ],
        )
    )
    for row in alignment:
        if not row["target_is_t_plus_1"] or not row["label_check_t_plus_1"]:
            raise ValueError(f"alignment check failed at origin {row['origin_index']}")

    shifted = shifted_label_test(aligned, d, cfg, current, 198)
    console.print(
        rich_table(
            "Shifted-label in-sample test (target d_t, already in the input)",
            ["item", "value"],
            [
                ["origin", shifted["origin_index"]],
                ["pairs", shifted["n_pairs"]],
                ["positives / negatives", f"{shifted['n_pos']} / {shifted['n_neg']}"],
                ["train AUC", f"{shifted['train_auc_shifted_label']:.4f}"],
                ["train accuracy", f"{shifted['train_accuracy_shifted_label']:.4f}"],
                ["final train loss", f"{shifted['final_train_loss']:.4f}"],
            ],
        )
    )

    overfit = overfit_test(aligned, d, cfg, current, 198)
    checkpoint_rows = []
    for epoch, values in overfit["checkpoints"].items():
        checkpoint_rows.append(
            [
                epoch,
                f"{values['train_loss']:.4f}",
                "nan" if not np.isfinite(values["train_auc"]) else f"{values['train_auc']:.4f}",
                f"{values['val_loss']:.4f}",
                "nan" if not np.isfinite(values["val_auc"]) else f"{values['val_auc']:.4f}",
            ]
        )
    console.print(
        rich_table(
            "Overfit test, origin 198, no dropout, 300 epochs",
            ["epoch", "train loss", "train AUC", "val loss", "val AUC"],
            checkpoint_rows,
        )
    )
    console.print(
        f"overfit block: train n={overfit['n_train']} pos={overfit['n_pos_train']} "
        f"neg={overfit['n_neg_train']}; val n={overfit['n_val']} "
        f"pos={overfit['n_pos_val']} neg={overfit['n_neg_val']}"
    )

    early = early_stopping_diagnostics(aligned, d, cfg, current, refits)
    current_run = q3b.run_lstm_walk(
        aligned, d, origins, cfg, args.seed, CURRENT_HP, current, "diagnostic_current", log=None
    )
    epoch_map = {
        int(refit["refit_origin_index"]): refit
        for refit in current_run["refits"]
    }
    early_rows = []
    for row in early:
        refit = epoch_map.get(row["refit_origin_index"], {})
        early_rows.append(
            [
                row["refit_origin_index"],
                row["n_pairs"],
                row["n_train"],
                row["n_val"],
                row["val_pos"],
                row["val_neg"],
                f"{row['val_share']:.3f}",
                "yes" if row["val_single_class"] else "",
                refit.get("best_epoch", ""),
                refit.get("epochs_run", ""),
            ]
        )
    console.print(
        rich_table(
            "Current early-stopping blocks (18 val months, seed 42)",
            ["refit", "pairs", "train", "val", "val pos", "val neg", "val share", "single class", "best epoch", "epochs run"],
            early_rows,
        )
    )

    tuning = tuning_diagnostics(aligned, d, cfg, current, tuning_origins, CURRENT_HP)
    console.print(
        rich_table(
            "Tuning-block base rates by target year",
            ["year", "n", "regime share", "positives"],
            [
                [row["year"], row["n"], f"{row['regime_share']:.4f}", row["positives"]]
                for row in tuning["base_rates_by_year"]
            ],
        )
    )
    console.print(
        rich_table(
            "Tuning-block AUC diagnosis (36 months)",
            ["series", "AUC", "95% CI", "Brier"],
            [
                [
                    "LSTM seed mean",
                    f"{tuning['lstm_bootstrap']['auc']:.4f}",
                    f"[{tuning['lstm_bootstrap']['ci_low']:.4f}, {tuning['lstm_bootstrap']['ci_high']:.4f}]",
                    f"{tuning['seed_mean']['Brier']:.4f}",
                ],
                [
                    "persistence",
                    f"{tuning['persistence']['AUC']:.4f}",
                    f"[{tuning['persistence']['ci_low']:.4f}, {tuning['persistence']['ci_high']:.4f}]",
                    f"{tuning['persistence']['Brier']:.4f}",
                ],
                [
                    "LSTM - persistence",
                    f"{tuning['lstm_minus_persistence']['difference']:+.4f}",
                    f"[{tuning['lstm_minus_persistence']['ci_low']:+.4f}, {tuning['lstm_minus_persistence']['ci_high']:+.4f}]",
                    "",
                ],
            ],
        )
    )
    console.print(
        rich_table(
            "Tuning-block per-seed AUC",
            ["seed", "AUC", "inverted AUC", "Brier"],
            [
                [seed, f"{row['AUC']:.4f}", f"{row['AUC_inverted']:.4f}", f"{row['Brier']:.4f}"]
                for seed, row in tuning["per_seed"].items()
            ],
        )
    )

    result = {
        "alignment": alignment,
        "shifted_label": shifted,
        "overfit": overfit,
        "early_stopping": early,
        "tuning": tuning,
    }
    DIAG_JSON.write_text(json.dumps(q3.to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"diagnostics written to {DIAG_JSON}")
    return result


def current_test_predictions(path: Path) -> dict:
    frame = pd.read_csv(path)
    test = frame[frame["block"] == "test"].copy()
    base = test[test["arm"] == "probit"].sort_values("origin_index")
    origin_index = base["origin_index"].to_numpy(dtype=int)
    y = base["y_true"].to_numpy(dtype=float)
    origin_dates = pd.DatetimeIndex(base["origin_date"])
    arms = {}
    for arm in sorted(test["arm"].unique()):
        subset = test[test["arm"] == arm]
        per_seed = {}
        for seed, group in subset.groupby("seed"):
            group = group.sort_values("origin_index")
            if not np.array_equal(group["origin_index"].to_numpy(dtype=int), origin_index):
                raise ValueError(f"{arm} seed {seed} test rows are not aligned")
            per_seed[int(seed)] = group["p"].to_numpy(dtype=float)
        arms[str(arm)] = {
            "per_seed": per_seed,
            "mean": np.nanmean(np.vstack(list(per_seed.values())), axis=0),
        }
    return {"origin_index": origin_index, "y": y, "origin_dates": origin_dates, "arms": arms}


def premium_pair(
    y: np.ndarray,
    p_leaked: np.ndarray,
    p_disciplined: np.ndarray,
    block: int,
    reps: int,
    seed: int,
) -> dict:
    y = np.asarray(y, dtype=float)
    leaked = np.asarray(p_leaked, dtype=float)
    disciplined = np.asarray(p_disciplined, dtype=float)
    mask = np.isfinite(y) & np.isfinite(leaked) & np.isfinite(disciplined)
    y = y[mask]
    leaked = leaked[mask]
    disciplined = disciplined[mask]
    brier_leaked = float(np.mean((leaked - y) ** 2))
    brier_disciplined = float(np.mean((disciplined - y) ** 2))
    auc_leaked = q3.auc_score(y, leaked)
    auc_disciplined = q3.auc_score(y, disciplined)
    rng = np.random.default_rng(int(seed))
    draws_brier = []
    draws_auc = []
    for _ in range(int(reps)):
        idx = circular_block_indices(len(y), block, rng)
        draw_brier = float(
            np.mean((disciplined[idx] - y[idx]) ** 2)
            - np.mean((leaked[idx] - y[idx]) ** 2)
        )
        draws_brier.append(draw_brier)
        leaked_auc = q3.auc_score(y[idx], leaked[idx])
        disciplined_auc = q3.auc_score(y[idx], disciplined[idx])
        if np.isfinite(leaked_auc) and np.isfinite(disciplined_auc):
            draws_auc.append(leaked_auc - disciplined_auc)
    brier_low, brier_high = np.percentile(draws_brier, [2.5, 97.5])
    auc_low, auc_high = (
        np.percentile(draws_auc, [2.5, 97.5]) if draws_auc else (float("nan"), float("nan"))
    )
    return {
        "Brier_leaked": brier_leaked,
        "Brier_disciplined": brier_disciplined,
        "AUC_leaked": float(auc_leaked),
        "AUC_disciplined": float(auc_disciplined),
        "premium_brier": brier_disciplined - brier_leaked,
        "premium_brier_ci_low": float(brier_low),
        "premium_brier_ci_high": float(brier_high),
        "premium_auc": float(auc_leaked - auc_disciplined),
        "premium_auc_ci_low": float(auc_low),
        "premium_auc_ci_high": float(auc_high),
        "valid_auc_replicates": int(len(draws_auc)),
        "replicates": int(reps),
        "block_months": int(block),
        "sign": "positive means the leaked arm looks better (higher AUC or lower Brier)",
    }


def leakage_block(y: np.ndarray, arms: dict[str, np.ndarray], args: argparse.Namespace) -> dict:
    output = {}
    for name, leaked_arm, disciplined_arm in LEAK_PAIRS:
        output[name] = {
            "leaked_arm": leaked_arm,
            "disciplined_arm": disciplined_arm,
            **premium_pair(
                y,
                arms[leaked_arm],
                arms[disciplined_arm],
                args.bootstrap_block,
                args.bootstrap_reps,
                args.seed,
            ),
        }
    return output


def labels_from_members(members: dict, series: list[str]) -> np.ndarray:
    labels = []
    for name in series:
        found = None
        for key, names in members.items():
            if name in names:
                found = int(key)
                break
        if found is None:
            raise ValueError(f"series {name} missing from cluster members")
        labels.append(found)
    return np.asarray(labels, dtype=int)


def bridge_null_distribution(
    bridge: dict,
    latest: dict,
    permutations: int,
    seed: int,
) -> dict:
    series = list(latest["series"])
    reference = labels_from_members(bridge["approach5_members"], series)
    bridge_labels = labels_from_members(bridge["members"], series)
    rng = np.random.default_rng(int(seed))
    null_ari = []
    null_nmi = []
    for _ in range(int(permutations)):
        permuted = rng.permutation(reference)
        null_ari.append(adjusted_rand_score(permuted, bridge_labels))
        null_nmi.append(normalized_mutual_info_score(permuted, bridge_labels))
    null_ari = np.asarray(null_ari)
    null_nmi = np.asarray(null_nmi)
    observed_ari = float(adjusted_rand_score(reference, bridge_labels))
    observed_nmi = float(normalized_mutual_info_score(reference, bridge_labels))
    return {
        "permutations": int(permutations),
        "observed_ARI": observed_ari,
        "observed_NMI": observed_nmi,
        "null_ARI_mean": float(np.mean(null_ari)),
        "null_ARI_q95": float(np.percentile(null_ari, 95)),
        "null_ARI_p_value": float((1 + np.sum(null_ari >= observed_ari)) / (1 + permutations)),
        "null_NMI_mean": float(np.mean(null_nmi)),
        "null_NMI_q95": float(np.percentile(null_nmi, 95)),
        "null_NMI_p_value": float((1 + np.sum(null_nmi >= observed_nmi)) / (1 + permutations)),
        "note": "one-sided permutation test; approach5 indicator labels are permuted against fixed bridge cluster labels",
    }


def retrain(args: argparse.Namespace, diagnostics: dict) -> dict:
    start = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw, meta, lags, series = q3.load_inputs(args.panel, args.meta)
    aligned = q3.align_information_set(raw, lags)
    aligned_zero = q3.align_information_set(raw, {name: 0 for name in lags})
    cfg = current_cfg(args)
    d = q3.build_label(aligned, cfg.threshold, "two_month")
    d_zero = q3.build_label(aligned_zero, cfg.threshold, "two_month")
    T = len(aligned)
    origins = np.arange(cfg.min_train, T - 1)
    test_start = int(cfg.min_train + cfg.tuning_months)
    tuning_origins = origins[origins < test_start]
    test_origins = origins[origins >= test_start]
    seeds = [int(args.seed) + offset for offset in range(3)]
    retrain_args = current_args()
    retrain_args.val_months = int(args.val_months)
    retrain_args.min_epochs = int(args.min_epochs)
    retrain_args.max_epochs = 300
    retrain_args.patience = 30
    log = Log(RETRAIN_LOG)
    log.write(
        f"retrain start: val_months={retrain_args.val_months} min_epochs={retrain_args.min_epochs} "
        f"patience={retrain_args.patience} max_epochs={retrain_args.max_epochs} seeds={seeds}"
    )

    tuning_rows = []
    chosen_hp = None
    chosen_setting = None
    best_brier = float("inf")
    for setting, hp in enumerate(GRID, start=1):
        seed_metrics = {}
        for seed in seeds:
            run_args = copy.copy(retrain_args)
            run_args.weight_decay = float(hp["weight_decay"])
            run = q3b.run_lstm_walk(
                aligned, d, tuning_origins, cfg, seed, hp, run_args, f"posthoc_tuning_{setting}", log=None
            )
            metrics = q3.classification_metrics(run["frame"]["y_true"], run["frame"]["p"])
            seed_metrics[seed] = metrics
            if run["failed"]:
                log.write(f"tuning setting {setting} seed {seed} failures={run['failed']}")
        brier_values = np.asarray([seed_metrics[seed]["Brier"] for seed in seeds])
        auc_values = np.asarray([seed_metrics[seed]["AUC"] for seed in seeds])
        row = {
            "setting": int(setting),
            "hidden": int(hp["hidden"]),
            "layers": int(hp["layers"]),
            "dropout": float(hp["dropout"]),
            "weight_decay": float(hp["weight_decay"]),
            "Brier_mean": float(np.nanmean(brier_values)),
            "Brier_std": float(np.nanstd(brier_values, ddof=0)),
            "AUC_mean": float(np.nanmean(auc_values)),
            "AUC_std": float(np.nanstd(auc_values, ddof=0)),
            "per_seed": {str(seed): seed_metrics[seed] for seed in seeds},
        }
        tuning_rows.append(row)
        log.write(
            f"post-hoc tuning setting {setting} {hp}: Brier={row['Brier_mean']:.4f} "
            f"+- {row['Brier_std']:.4f} AUC={row['AUC_mean']:.4f}"
        )
        if row["Brier_mean"] < best_brier - 1e-12:
            best_brier = row["Brier_mean"]
            chosen_setting = setting
            chosen_hp = dict(hp)
    if chosen_hp is None:
        raise RuntimeError("post-hoc tuning produced no usable setting")
    log.write(f"post-hoc chosen setting {chosen_setting}: {chosen_hp}")
    console.print(
        rich_table(
            "Post-hoc LSTM selection on tuning block only",
            ["setting", "hidden", "dropout", "weight decay", "Brier mean", "Brier std", "AUC mean", "chosen"],
            [
                [
                    row["setting"],
                    row["hidden"],
                    f"{row['dropout']:.1f}",
                    f"{row['weight_decay']:.0e}",
                    f"{row['Brier_mean']:.4f}",
                    f"{row['Brier_std']:.4f}",
                    f"{row['AUC_mean']:.4f}",
                    "yes" if row["setting"] == chosen_setting else "",
                ]
                for row in tuning_rows
            ],
        )
    )

    final_runs = {}
    run_meta = []
    for arm, arm_aligned, arm_labels, full_scaler in (
        ("lstm_disciplined", aligned, d, False),
        ("lstm_zero_lag", aligned_zero, d_zero, False),
        ("lstm_zero_lag_full_sample", aligned_zero, d_zero, True),
    ):
        final_runs[arm] = {}
        for seed in seeds:
            run_args = copy.copy(retrain_args)
            run_args.weight_decay = float(chosen_hp["weight_decay"])
            run = q3b.run_lstm_walk(
                arm_aligned,
                arm_labels,
                origins,
                cfg,
                seed,
                chosen_hp,
                run_args,
                arm,
                full_sample_scaler=full_scaler,
                log=log,
            )
            final_runs[arm][seed] = run
            run_meta.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "n_refits": int(len(run["refits"])),
                    "n_failed": int(len(run["failed"])),
                }
            )
            log.write(f"post-hoc {arm} seed {seed}: {len(run['refits'])} refits, {len(run['failed'])} failed")

    current = current_test_predictions(RESULTS / "predictions.csv")
    current_payload = json.loads(METRICS.read_text(encoding="utf-8"))
    origin_index = current["origin_index"]
    y_test = current["y"]
    origin_dates = current["origin_dates"]

    retrained_arms = {}
    retrained_comparison = {}
    for arm in ("lstm_disciplined", "lstm_zero_lag", "lstm_zero_lag_full_sample"):
        per_seed = {}
        arrays = {}
        for seed in seeds:
            frame = final_runs[arm][seed]["frame"]
            test = frame[frame["block"] == "test"].set_index("origin_index").reindex(origin_index)
            if not np.allclose(test["y_true"].to_numpy(dtype=float), y_test, equal_nan=True):
                raise ValueError(f"{arm} seed {seed} targets are not aligned")
            values = test["p"].to_numpy(dtype=float)
            arrays[seed] = values
            per_seed[seed] = q3.classification_metrics(y_test, values)
        retrained_arms[arm] = {
            "per_seed": per_seed,
            "arrays": arrays,
            "mean": np.nanmean(np.vstack(list(arrays.values())), axis=0),
        }
        retrained_comparison[arm] = q3b.summarise_arm(per_seed)
        retrained_comparison[arm]["stochastic"] = True

    comparison = {
        "probit": current_payload["metrics"]["comparison"]["probit"],
        "lstm_disciplined_original": current_payload["metrics"]["comparison"]["lstm_disciplined"],
        "lstm_disciplined": retrained_comparison["lstm_disciplined"],
        "lstm_zero_lag": retrained_comparison["lstm_zero_lag"],
        "lstm_zero_lag_full_sample": retrained_comparison["lstm_zero_lag_full_sample"],
        "probit_zero_lag": current_payload["metrics"]["comparison"]["probit_zero_lag"],
        "prior": current_payload["metrics"]["comparison"]["prior"],
        "persistence": current_payload["metrics"]["comparison"]["persistence"],
    }

    old_arm_means = {arm: current["arms"][arm]["mean"] for arm in current["arms"]}
    new_arm_means = {
        "probit": current["arms"]["probit"]["mean"],
        "probit_zero_lag": current["arms"]["probit_zero_lag"]["mean"],
        "lstm_disciplined": retrained_arms["lstm_disciplined"]["mean"],
        "lstm_zero_lag": retrained_arms["lstm_zero_lag"]["mean"],
        "lstm_zero_lag_full_sample": retrained_arms["lstm_zero_lag_full_sample"]["mean"],
    }
    leakage = {
        "original_sign_corrected": leakage_block(y_test, old_arm_means, args),
        "retrained": leakage_block(y_test, new_arm_means, args),
    }

    first_run = final_runs["lstm_disciplined"][int(args.seed)]
    latent_rows, labels_by_k, hidden_test, latent_y, latent_dates = q3b.latent_month_analysis(
        first_run["hidden"], first_run["frame"], retrain_args
    )
    raw_cluster_labels, windows = q3.cluster_reference_panel(raw, cfg)
    latest = windows[-1]
    hidden_by_origin = {
        int(row.origin_index): first_run["hidden"][index]
        for index, row in enumerate(first_run["frame"].itertuples())
    }
    bridge = q3b.bridge_to_approach5_clusters(hidden_by_origin, aligned, latest, retrain_args)
    null = bridge_null_distribution(bridge, latest, args.permutations, args.seed)
    console.print(
        rich_table(
            "Post-hoc latent month clusters",
            ["k", "ARI", "NMI", "cluster sizes"],
            [
                [
                    row["k"],
                    f"{row['ARI']:.4f}",
                    f"{row['NMI']:.4f}",
                    ", ".join(str(size) for size in row["cluster_sizes"]),
                ]
                for row in latent_rows
            ],
        )
    )
    console.print(
        rich_table(
            "Post-hoc bridge and permutation null",
            ["item", "observed", "null mean", "null q95", "p-value"],
            [
                [
                    "ARI",
                    f"{null['observed_ARI']:.4f}",
                    f"{null['null_ARI_mean']:.4f}",
                    f"{null['null_ARI_q95']:.4f}",
                    f"{null['null_ARI_p_value']:.4f}",
                ],
                [
                    "NMI",
                    f"{null['observed_NMI']:.4f}",
                    f"{null['null_NMI_mean']:.4f}",
                    f"{null['null_NMI_q95']:.4f}",
                    f"{null['null_NMI_p_value']:.4f}",
                ],
            ],
        )
    )

    epoch_summary = {}
    refit_rows = []
    for arm in ("lstm_disciplined", "lstm_zero_lag", "lstm_zero_lag_full_sample"):
        for seed in seeds:
            refits = final_runs[arm][seed]["refits"]
            refit_rows.extend(refits)
            epochs = np.asarray([row["epochs_run"] for row in refits], dtype=int)
            best = np.asarray([row["best_epoch"] for row in refits], dtype=int)
            epoch_summary[f"{arm}_seed{seed}"] = {
                "n_refits": int(len(refits)),
                "epochs_run": epochs.tolist(),
                "best_epoch": best.tolist(),
                "epochs_mean": float(np.mean(epochs)),
                "epochs_median": float(np.median(epochs)),
                "epochs_min": int(np.min(epochs)),
                "epochs_max": int(np.max(epochs)),
            }
    failed_fits = [failure for run in final_runs.values() for result in run.values() for failure in result["failed"]]
    retrain_args_payload = {
        "validation_scheme": (
            "expanding split: all training pairs before the refit origin, with the last "
            f"{retrain_args.val_months} training-pair months held out for validation"
        ),
        "validation_reason": "18-month blocks contained single-class and very small-positive validation samples",
        "min_epochs": int(retrain_args.min_epochs),
        "patience": int(retrain_args.patience),
        "max_epochs": int(retrain_args.max_epochs),
        "seeds": seeds,
        "grid": GRID,
        "selection_criterion": "lowest mean Brier across 3 seeds on the 36-month tuning block only",
        "chosen_setting": int(chosen_setting),
        "chosen_hyperparameters": chosen_hp,
        "same_input_definition": "12-month standardized aligned panel, gdp_yoy excluded",
        "same_test_block": True,
    }
    comparison_table_rows = []
    for arm in ARM_ORDER:
        values = comparison[arm]
        comparison_table_rows.append(
            [
                arm,
                f"{values['Brier_mean']:.4f}",
                f"{values['Brier_std']:.4f}",
                f"{values['AUC_mean']:.4f}",
                f"{values['AUC_std']:.4f}",
                f"{values['accuracy_mean']:.4f}",
                f"{values['accuracy_std']:.4f}",
            ]
        )
    console.print(
        rich_table(
            "Post-hoc test comparison",
            ["arm", "Brier", "Brier std", "AUC", "AUC std", "accuracy", "accuracy std"],
            comparison_table_rows,
        )
    )
    for label, block in (("original, corrected sign", leakage["original_sign_corrected"]), ("retrained", leakage["retrained"])):
        console.print(
            rich_table(
                f"Leakage premium ({label})",
                ["pair", "premium AUC", "AUC 95% CI", "premium Brier", "Brier 95% CI"],
                [
                    [
                        name,
                        f"{row['premium_auc']:+.4f}",
                        f"[{row['premium_auc_ci_low']:+.4f}, {row['premium_auc_ci_high']:+.4f}]",
                        f"{row['premium_brier']:+.4f}",
                        f"[{row['premium_brier_ci_low']:+.4f}, {row['premium_brier_ci_high']:+.4f}]",
                    ]
                    for name, row in block.items()
                ],
            )
        )

    prediction_rows = []
    for arm in ("lstm_disciplined", "lstm_zero_lag", "lstm_zero_lag_full_sample"):
        for seed in seeds:
            frame = final_runs[arm][seed]["frame"]
            test = frame[frame["block"] == "test"].set_index("origin_index").reindex(origin_index)
            for position, origin in enumerate(origin_index):
                prediction_rows.append(
                    {
                        "arm": arm,
                        "seed": int(seed),
                        "origin_index": int(origin),
                        "origin_date": origin_dates[position],
                        "target_date": pd.Timestamp(test["target_date"].iloc[position]),
                        "block": "test",
                        "y_true": float(y_test[position]),
                        "p": float(test["p"].iloc[position]),
                        "p_raw": float(test["p_raw"].iloc[position]),
                        "refit_origin_index": int(test["refit_origin_index"].iloc[position]),
                        "feature_dim": int(test["feature_dim"].iloc[position]),
                        "pos_weight": float(test["pos_weight"].iloc[position]),
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(RESULTS / "post_hoc_predictions.csv", index=False)

    hidden_frame = pd.DataFrame(
        first_run["hidden"], columns=[f"h{index + 1}" for index in range(first_run["hidden"].shape[1])]
    )
    hidden_frame.insert(0, "origin_index", first_run["frame"]["origin_index"].to_numpy())
    hidden_frame.insert(1, "origin_date", first_run["frame"]["origin_date"].to_numpy())
    hidden_frame.insert(2, "target_date", first_run["frame"]["target_date"].to_numpy())
    hidden_frame.insert(3, "block", first_run["frame"]["block"].to_numpy())
    hidden_frame["d_t"] = first_run["frame"]["d_t"].to_numpy()
    hidden_frame["y_next"] = first_run["frame"]["y_true"].to_numpy()
    hidden_frame["model_refit_origin_index"] = first_run["frame"]["refit_origin_index"].to_numpy()
    label_lookup = {pd.Timestamp(date): index for index, date in enumerate(latent_dates)}
    for k, labels in labels_by_k.items():
        hidden_frame[f"kmeans_k{k}"] = [
            int(labels[label_lookup[pd.Timestamp(timestamp)]])
            if pd.Timestamp(timestamp) in label_lookup
            else -1
            for timestamp in hidden_frame["target_date"]
        ]
    hidden_frame.to_csv(RESULTS / "post_hoc_hidden_states.csv", index=False)

    diagnostics_path = RESULTS / "post_hoc_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(q3.to_jsonable(diagnostics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runtime = time.perf_counter() - start
    post_hoc = {
        "note": (
            "Post-hoc work done after the test block had already been seen. Selection used only the "
            "tuning block; test metrics are reported for transparency and are not used to choose models."
        ),
        "diagnostics": diagnostics,
        "retrain": {
            "protocol": retrain_args_payload,
            "tuning_table": tuning_rows,
            "comparison": comparison,
            "leakage": leakage,
            "latent": {
                "month_kmeans": latent_rows,
                "bridge": bridge,
                "null": null,
            },
            "epochs_per_run": epoch_summary,
            "refits": refit_rows,
            "failed_fits": failed_fits,
            "runtime_seconds": float(runtime),
        },
    }
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    if list(payload) != ["params", "metrics", "fit_seconds", "history"]:
        raise ValueError(f"unexpected top-level keys: {list(payload)}")
    payload["metrics"]["post_hoc"] = q3.to_jsonable(post_hoc)
    METRICS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.write(f"post-hoc retraining finished in {runtime:.2f}s; failed fits={len(failed_fits)}")
    (CACHE / "q3b_posthoc_terminal.txt").write_text(console.export_text(), encoding="utf-8")
    return post_hoc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    diagnostics = None
    if args.mode in ("diagnose", "all"):
        if DIAG_LOG.exists():
            DIAG_LOG.unlink()
        log = Log(DIAG_LOG)
        log.write(f"start diagnostics args={vars(args)}")
        diagnostics = run_diagnostics(args)
        log.write("diagnostics finished")
        (CACHE / "q3b_diagnostics_terminal.txt").write_text(console.export_text(), encoding="utf-8")
    if args.mode in ("retrain", "all"):
        if diagnostics is None:
            diagnostics = json.loads(DIAG_JSON.read_text(encoding="utf-8"))
        if RETRAIN_LOG.exists():
            RETRAIN_LOG.unlink()
        retrain(args, diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
