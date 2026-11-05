#!/usr/bin/env python3
"""Q3 Option B: LSTM PMI-regime forecasting and leakage ablations."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import approach5_q3a_classical as q3

DEFAULT_PANEL = ROOT / "data" / "clean" / "macro_panel.csv"
DEFAULT_META = ROOT / "data" / "clean" / "macro_meta.csv"
DEFAULT_OUT = ROOT / "results" / "approach6" / "macro"
DEFAULT_LOG = ROOT / ".cache" / "q3b_run.log"

SETTINGS = (
    {"hidden": 8, "layers": 1, "dropout": 0.0},
    {"hidden": 16, "layers": 1, "dropout": 0.0},
    {"hidden": 16, "layers": 1, "dropout": 0.2},
    {"hidden": 16, "layers": 2, "dropout": 0.2},
)
ARM_ORDER = (
    "probit",
    "lstm_disciplined",
    "lstm_zero_lag",
    "lstm_zero_lag_full_sample",
    "probit_zero_lag",
    "prior",
    "persistence",
)
LEAK_PAIRS = (
    ("lstm_lag_discipline", "lstm_zero_lag", "lstm_disciplined"),
    ("lstm_full_sample_standardisation", "lstm_zero_lag_full_sample", "lstm_disciplined"),
    ("lstm_standardisation_only", "lstm_zero_lag_full_sample", "lstm_zero_lag"),
    ("probit_lag_discipline", "probit_zero_lag", "probit"),
)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.use_deterministic_algorithms(True, warn_only=True)

console = Console(record=True)


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        elapsed = time.perf_counter() - self.start
        line = f"[{elapsed:8.2f}s] {message}"
        self.handle.write(line + "\n")
        self.handle.flush()
        console.print(line)

    def check_budget(self, limit_minutes: float, phase: str) -> None:
        elapsed = time.perf_counter() - self.start
        if elapsed > limit_minutes * 60.0:
            raise TimeoutError(
                f"runtime guard: {phase} exceeded {limit_minutes:.1f} minutes "
                f"(elapsed {elapsed / 60.0:.1f} min)"
            )

    def close(self) -> None:
        self.handle.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q3 Option B: LSTM regime forecasting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument("--tuning-months", type=int, default=36)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--refit-months", type=int, default=6)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--val-months", type=int, default=18)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--time-budget-minutes", type=float, default=30.0)
    return parser.parse_args(argv)


def q3_config(args: argparse.Namespace) -> q3.RunConfig:
    return q3.RunConfig(
        seed=args.seed,
        min_train=args.min_train,
        tuning_months=args.tuning_months,
        min_coverage=args.min_coverage,
        threshold=args.threshold,
    )


def zero_lag_meta(meta: pd.DataFrame) -> pd.DataFrame:
    frame = meta.copy()
    frame["lag"] = 0
    return frame


def feature_candidates(aligned: pd.DataFrame) -> list[str]:
    return [column for column in aligned.columns if column not in ("date", "gdp_yoy")]


def fit_feature_scaler(
    aligned: pd.DataFrame,
    fit_idx: np.ndarray,
    min_coverage: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return q3.fit_standardizer(aligned, feature_candidates(aligned), fit_idx, min_coverage)


def standardised_window(
    aligned: pd.DataFrame,
    t: int,
    window: int,
    columns: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    start = int(t) - int(window) + 1
    if start < 0:
        raise ValueError(f"origin {t} has fewer than {window} months of history")
    idx = np.arange(start, int(t) + 1)
    return q3.apply_standardizer(aligned, idx, columns, means, stds)


def training_pair_ends(t: int, d: np.ndarray, window: int) -> np.ndarray:
    ends = np.arange(int(window) - 1, int(t))
    if len(ends) == 0:
        return ends
    return ends[np.isfinite(d[ends + 1])]


def build_training_arrays(
    aligned: pd.DataFrame,
    d: np.ndarray,
    t: int,
    window: int,
    val_months: int,
    columns: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> dict:
    z = q3.apply_standardizer(aligned, np.arange(0, int(t) + 1), columns, means, stds)
    ends = training_pair_ends(t, d, window)
    if len(ends) <= val_months:
        raise ValueError(f"only {len(ends)} training pairs at origin {t}")
    features = np.stack([z[end - window + 1 : end + 1] for end in ends], axis=0)
    labels = d[ends + 1].astype(float)
    n_val = int(val_months)
    n_train = len(ends) - n_val
    if n_train < 5:
        raise ValueError(f"only {n_train} fitting pairs at origin {t}")
    return {
        "X_train": features[:n_train],
        "y_train": labels[:n_train],
        "X_val": features[n_train:],
        "y_val": labels[n_train:],
        "train_ends": ends[:n_train],
        "val_ends": ends[n_train:],
        "z": z,
    }


def positive_weight(y: np.ndarray) -> tuple[float, int, int]:
    values = np.asarray(y, dtype=float)
    n_pos = int(np.sum(values == 1.0))
    n_neg = int(np.sum(values == 0.0))
    if n_pos == 0:
        return 1.0, n_pos, n_neg
    return float(n_neg / n_pos), n_pos, n_neg


def calibrate_probability(p: float, pos_weight: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    weight = float(pos_weight)
    return p / (p + weight * (1.0 - p))


def calibrate_logit(logit: float, pos_weight: float) -> float:
    return float(1.0 / (1.0 + math.exp(-(float(logit) - math.log(float(pos_weight))))))


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.lstm(x)
        hidden = output[:, -1]
        logits = self.head(self.dropout(hidden)).squeeze(-1)
        return logits, hidden


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hp: dict,
    seed: int,
    args: argparse.Namespace,
) -> tuple[LSTMClassifier, dict]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    model = LSTMClassifier(
        input_dim=int(X_train.shape[2]),
        hidden=int(hp["hidden"]),
        layers=int(hp["layers"]),
        dropout=float(hp["dropout"]),
    )
    weight, n_pos, n_neg = positive_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_train = torch.as_tensor(X_train, dtype=torch.float32)
    t_train = torch.as_tensor(y_train, dtype=torch.float32)
    x_val = torch.as_tensor(X_val, dtype=torch.float32)
    t_val = torch.as_tensor(y_val, dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    train_loss = float("nan")
    epochs_run = 0
    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x_train)
        loss = criterion(logits, t_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss = float(loss.item())
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x_val)
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(val_logits, t_val).item())
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        epochs_run = epoch
        min_epochs = int(getattr(args, "min_epochs", 0))
        if epoch - best_epoch >= int(args.patience) and epoch >= min_epochs:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "train_loss": train_loss,
        "val_loss": float(best_val),
        "pos_weight": float(weight),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "min_epochs": int(getattr(args, "min_epochs", 0)),
        "converged": bool(best_epoch > 0),
    }
    return model, info


def predict_lstm(
    model: LSTMClassifier,
    X: np.ndarray,
    pos_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits, hidden = model(torch.as_tensor(X, dtype=torch.float32))
        p_raw = torch.sigmoid(logits)
        p_cal = torch.sigmoid(logits - math.log(float(pos_weight)))
    return p_cal.numpy(), p_raw.numpy(), hidden.numpy()


def refit_schedule(origins: np.ndarray, refit_every: int) -> list[int]:
    if len(origins) == 0:
        return []
    first = int(origins[0])
    return [int(t) for t in origins if (int(t) - first) % int(refit_every) == 0]


def run_lstm_walk(
    aligned: pd.DataFrame,
    d: np.ndarray,
    origins: np.ndarray,
    cfg: q3.RunConfig,
    seed: int,
    hp: dict,
    args: argparse.Namespace,
    arm: str,
    full_sample_scaler: bool = False,
    log: RunLog | None = None,
) -> dict:
    origins = np.asarray([int(t) for t in origins], dtype=int)
    refits_at = set(refit_schedule(origins, args.refit_months))
    full_scaler = None
    if full_sample_scaler:
        full_scaler = fit_feature_scaler(aligned, np.arange(len(aligned)), cfg.min_coverage)
        if not full_scaler[2]:
            raise ValueError("full-sample scaler kept no feature series")
    current = None
    rows = []
    hidden_rows = []
    refits = []
    failures = []
    block_start = int(cfg.min_train + cfg.tuning_months)
    for t in origins:
        if t in refits_at:
            if log is not None:
                log.write(f"{arm}: seed {seed} refit origin {t}")
            try:
                scaler_idx = np.arange(len(aligned)) if full_sample_scaler else np.arange(0, t + 1)
                means, stds, columns = (
                    full_scaler if full_scaler is not None else fit_feature_scaler(aligned, scaler_idx, cfg.min_coverage)
                )
                if not columns:
                    raise ValueError("no feature series survived coverage filter")
                data = build_training_arrays(
                    aligned, d, t, args.window, args.val_months, columns, means, stds
                )
                model, info = train_lstm(
                    data["X_train"], data["y_train"], data["X_val"], data["y_val"], hp, seed, args
                )
                current = {
                    "model": model,
                    "means": means,
                    "stds": stds,
                    "columns": columns,
                    "refit_origin_index": t,
                    "feature_dim": int(len(columns)),
                    **info,
                }
                refits.append(
                    {
                        "arm": arm,
                        "seed": int(seed),
                        "refit_origin_index": int(t),
                        "refit_origin_date": str(aligned["date"].iloc[t].date()),
                        "feature_dim": int(len(columns)),
                        "n_train": int(info["n_train"]),
                        "n_val": int(info["n_val"]),
                        "n_pos_train": int(info["n_pos"]),
                        "n_neg_train": int(info["n_neg"]),
                        "pos_weight": float(info["pos_weight"]),
                        "best_epoch": int(info["best_epoch"]),
                        "epochs_run": int(info["epochs_run"]),
                        "min_epochs": int(info["min_epochs"]),
                        "train_loss": float(info["train_loss"]),
                        "val_loss": float(info["val_loss"]),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "arm": arm,
                        "seed": int(seed),
                        "refit_origin_index": int(t),
                        "stage": "fit",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if log is not None:
                    log.write(f"{arm}: seed {seed} refit origin {t} FAILED {type(exc).__name__}: {exc}")
        base_row = {
            "arm": arm,
            "seed": int(seed),
            "origin_index": int(t),
            "origin_date": aligned["date"].iloc[t],
            "target_date": aligned["date"].iloc[t + 1],
            "block": "test" if t >= block_start else "tuning",
            "y_true": float(d[t + 1]),
            "d_t": float(d[t]),
            "p": float("nan"),
            "p_raw": float("nan"),
            "refit_origin_index": int(current["refit_origin_index"]) if current is not None else -1,
            "feature_dim": int(current["feature_dim"]) if current is not None else 0,
            "pos_weight": float(current["pos_weight"]) if current is not None else float("nan"),
            "n_train": int(current["n_train"]) if current is not None else 0,
            "n_val": int(current["n_val"]) if current is not None else 0,
        }
        if current is None:
            rows.append(base_row)
            hidden_rows.append(np.full(int(hp["hidden"]), np.nan))
            continue
        try:
            window = standardised_window(
                aligned, t, args.window, current["columns"], current["means"], current["stds"]
            )
            p_cal, p_raw, hidden = predict_lstm(
                current["model"], window[None, :, :], current["pos_weight"]
            )
            base_row["p"] = float(p_cal[0])
            base_row["p_raw"] = float(p_raw[0])
            hidden_rows.append(hidden[0].astype(float))
        except Exception as exc:
            failures.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "origin_index": int(t),
                    "stage": "predict",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            hidden_rows.append(np.full(int(hp["hidden"]), np.nan))
        rows.append(base_row)
    frame = pd.DataFrame(rows)
    return {
        "frame": frame,
        "hidden": np.vstack(hidden_rows),
        "refits": refits,
        "failed": failures,
        "full_sample_scaler": bool(full_sample_scaler),
    }


def summarise_arm(per_seed: dict[int, dict]) -> dict:
    seeds = sorted(per_seed)
    values = [per_seed[seed] for seed in seeds]
    summary = {"per_seed": per_seed, "n_seeds": len(seeds)}
    for metric in ("Brier", "AUC", "accuracy"):
        array = np.asarray([value[metric] for value in values], dtype=float)
        summary[f"{metric}_mean"] = float(np.nanmean(array))
        summary[f"{metric}_std"] = float(np.nanstd(array, ddof=0))
    for key in ("n", "positives", "class_share"):
        summary[key] = values[0][key]
    return summary


def block_bootstrap_premium(
    y: np.ndarray,
    p_leaked: np.ndarray,
    p_disciplined: np.ndarray,
    reps: int,
    block: int,
    seed: int,
) -> dict:
    y = np.asarray(y, dtype=float)
    leaked = np.asarray(p_leaked, dtype=float)
    disciplined = np.asarray(p_disciplined, dtype=float)
    mask = np.isfinite(y) & np.isfinite(leaked) & np.isfinite(disciplined)
    y = y[mask]
    squared_diff = (leaked[mask] - y) ** 2 - (disciplined[mask] - y) ** 2
    n = len(squared_diff)
    rng = np.random.default_rng(int(seed))
    n_blocks = int(math.ceil(n / block))
    draws = np.empty(int(reps), dtype=float)
    for rep in range(int(reps)):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
        draws[rep] = float(np.mean(squared_diff[idx]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "premium": float(np.mean(squared_diff)),
        "ci_low": float(low),
        "ci_high": float(high),
        "replicates": int(reps),
        "block_months": int(block),
        "n": int(n),
    }


def latent_month_analysis(
    hidden: np.ndarray,
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[int, np.ndarray], np.ndarray, np.ndarray, pd.DatetimeIndex]:
    test_mask = (frame["block"] == "test").to_numpy()
    hidden_test = hidden[test_mask]
    y_test = frame.loc[test_mask, "y_true"].to_numpy(dtype=float)
    dates_test = pd.DatetimeIndex(frame.loc[test_mask, "target_date"])
    keep = np.isfinite(hidden_test).all(axis=1) & np.isfinite(y_test)
    hidden_test = hidden_test[keep]
    y_test = y_test[keep]
    dates_test = dates_test[keep]
    scaled = StandardScaler().fit_transform(hidden_test)
    rows = []
    labels_by_k = {}
    for k in range(int(args.k_min), int(args.k_max) + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=int(args.seed))
        labels = model.fit_predict(scaled)
        labels_by_k[k] = labels
        sizes = np.bincount(labels, minlength=k)
        rows.append(
            {
                "k": int(k),
                "ARI": float(adjusted_rand_score(y_test, labels)),
                "NMI": float(normalized_mutual_info_score(y_test, labels)),
                "inertia": float(model.inertia_),
                "cluster_sizes": [int(size) for size in sizes],
                "n_months": int(len(y_test)),
            }
        )
    return rows, labels_by_k, hidden_test, y_test, dates_test


def bridge_to_approach5_clusters(
    hidden_by_origin: dict[int, np.ndarray],
    aligned: pd.DataFrame,
    latest: dict,
    args: argparse.Namespace,
) -> dict:
    idx = np.arange(int(latest["start_index"]), int(latest["end_index"]) + 1)
    hidden = np.vstack([hidden_by_origin[int(t)] for t in idx])
    series = list(latest["series"])
    corr = np.zeros((len(series), hidden.shape[1]), dtype=float)
    for i, name in enumerate(series):
        values = aligned[name].iloc[idx].to_numpy(dtype=float)
        for j in range(hidden.shape[1]):
            mask = np.isfinite(values) & np.isfinite(hidden[:, j])
            if mask.sum() < 12:
                continue
            x = values[mask]
            h = hidden[mask, j]
            if float(np.std(x)) == 0.0 or float(np.std(h)) == 0.0:
                continue
            corr[i, j] = float(np.corrcoef(x, h)[0, 1])
    k = int(latest["k"])
    model = KMeans(n_clusters=k, n_init=20, random_state=int(args.seed))
    labels = model.fit_predict(corr)
    reference = np.asarray(latest["labels"], dtype=int)
    ari = float(adjusted_rand_score(reference, labels))
    nmi = float(normalized_mutual_info_score(reference, labels))
    members = {}
    for label in sorted(set(labels)):
        members[str(int(label))] = [series[i] for i in range(len(series)) if int(labels[i]) == label]
    reference_members = {}
    for label in sorted(set(reference)):
        reference_members[str(int(label))] = [
            series[i] for i in range(len(series)) if int(reference[i]) == label
        ]
    return {
        "construction": (
            "Direct ARI between Option A indicator clusters and month-level h_t clusters is undefined. "
            "Each indicator is mapped to a vector of Pearson correlations between its aligned series and "
            f"each of the {hidden.shape[1]} hidden dimensions over the latest Option A 36-month window "
            f"({latest['start_date']:%Y-%m} to {latest['end_date']:%Y-%m}); those correlation vectors are "
            f"clustered with k={k}, the k of approach5's latest window, and compared with approach5's "
            "latest indicator labels. Both sides then partition indicators."
        ),
        "window_start": str(latest["start_date"].date()),
        "window_end": str(latest["end_date"].date()),
        "k": k,
        "n_indicators": int(len(series)),
        "ARI": ari,
        "NMI": nmi,
        "cluster_sizes": [int(size) for size in np.bincount(labels, minlength=k)],
        "members": members,
        "approach5_members": reference_members,
    }


def comparison_table(comparison: dict) -> Table:
    table = Table(
        title=f"Q3B test-block comparison (same {comparison['probit']['n']} months as approach5)"
    )
    for column in ("arm", "Brier mean", "Brier std", "AUC mean", "AUC std", "accuracy mean", "accuracy std"):
        table.add_column(column)
    for arm in ARM_ORDER:
        values = comparison[arm]
        table.add_row(
            arm,
            f"{values['Brier_mean']:.4f}",
            f"{values['Brier_std']:.4f}",
            "nan" if not np.isfinite(values["AUC_mean"]) else f"{values['AUC_mean']:.4f}",
            f"{values['AUC_std']:.4f}",
            f"{values['accuracy_mean']:.4f}",
            f"{values['accuracy_std']:.4f}",
        )
    return table


def tuning_table(rows: list[dict], chosen_setting: int) -> Table:
    table = Table(title="LSTM hyperparameter selection on the tuning block only")
    for column in ("setting", "hidden", "layers", "dropout", "Brier mean", "Brier std", "AUC mean", "chosen"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row["setting"]),
            str(row["hidden"]),
            str(row["layers"]),
            f"{row['dropout']:.1f}",
            f"{row['Brier_mean']:.4f}",
            f"{row['Brier_std']:.4f}",
            f"{row['AUC_mean']:.4f}",
            "yes" if row["setting"] == chosen_setting else "",
        )
    return table


def leakage_table(leakage: dict) -> Table:
    table = Table(title="Leakage premium (leaked arm minus disciplined twin, Brier; >0 means leaked is worse)")
    for column in ("pair", "leaked", "disciplined", "premium", "95% CI", "Brier leaked", "Brier disciplined"):
        table.add_column(column)
    for name, _, _ in LEAK_PAIRS:
        row = leakage[name]
        table.add_row(
            name,
            row["leaked_arm"],
            row["disciplined_arm"],
            f"{row['premium']:+.4f}",
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]",
            f"{row['Brier_leaked']:.4f}",
            f"{row['Brier_disciplined']:.4f}",
        )
    return table


def latent_table(rows: list[dict]) -> Table:
    table = Table(title="Month-level k-means on final disciplined LSTM hidden states (test months)")
    for column in ("k", "ARI vs regime", "NMI vs regime", "cluster sizes"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row["k"]),
            f"{row['ARI']:.4f}",
            f"{row['NMI']:.4f}",
            ", ".join(str(size) for size in row["cluster_sizes"]),
        )
    return table


def probabilities_figure(pred_long: pd.DataFrame, comparison: dict) -> go.Figure:
    test = pred_long[pred_long["block"] == "test"]
    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Test-block P(d=1), seed mean", "Brier by arm"),
        row_heights=[0.68, 0.32],
    )
    colors = {
        "probit": "#1f77b4",
        "lstm_disciplined": "#d62728",
        "lstm_zero_lag": "#ff7f0e",
        "lstm_zero_lag_full_sample": "#8c564b",
        "probit_zero_lag": "#9467bd",
        "prior": "#2ca02c",
        "persistence": "#7f7f7f",
    }
    for arm in ARM_ORDER:
        subset = test[test["arm"] == arm]
        grouped = subset.groupby("target_date")["p"].mean().sort_index()
        figure.add_trace(
            go.Scatter(
                x=grouped.index,
                y=grouped.to_numpy(),
                mode="lines",
                name=arm,
                line=dict(color=colors[arm], width=1.4),
            ),
            row=1,
            col=1,
        )
    realised = test[test["arm"] == "persistence"].sort_values("target_date")
    figure.add_trace(
        go.Scatter(
            x=realised["target_date"],
            y=realised["y_true"],
            mode="markers",
            name="Realised d",
            marker=dict(color="#111111", size=4, symbol="circle-open"),
        ),
        row=1,
        col=1,
    )
    figure.add_hline(y=0.5, line_dash="dot", line_color="#888888", row=1, col=1)
    figure.add_trace(
        go.Bar(
            x=list(ARM_ORDER),
            y=[comparison[arm]["Brier_mean"] for arm in ARM_ORDER],
            marker_color=[colors[arm] for arm in ARM_ORDER],
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="P(d=1)", row=1, col=1)
    figure.update_yaxes(title_text="Brier", row=2, col=1)
    figure.update_layout(
        title="Q3B regime probabilities and Brier scores",
        xaxis_title="Target month",
        template="plotly_white",
        hovermode="x unified",
    )
    return figure


def latent_figure(
    rows: list[dict],
    hidden_test: np.ndarray,
    y_test: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cluster agreement with regime labels", "Hidden-state PCA, test months"),
    )
    figure.add_trace(
        go.Scatter(
            x=[row["k"] for row in rows],
            y=[row["ARI"] for row in rows],
            mode="lines+markers",
            name="ARI",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[row["k"] for row in rows],
            y=[row["NMI"] for row in rows],
            mode="lines+markers",
            name="NMI",
            line=dict(color="#d62728"),
        ),
        row=1,
        col=1,
    )
    coordinates = PCA(n_components=2, random_state=int(args.seed)).fit_transform(
        StandardScaler().fit_transform(hidden_test)
    )
    k_two = labels_by_k.get(2, np.zeros(len(hidden_test), dtype=int))
    for regime in (0.0, 1.0):
        for cluster in sorted(set(k_two)):
            mask = (y_test == regime) & (k_two == cluster)
            if not mask.any():
                continue
            figure.add_trace(
                go.Scatter(
                    x=coordinates[mask, 0],
                    y=coordinates[mask, 1],
                    mode="markers",
                    name=f"d={int(regime)}, cluster {int(cluster)}",
                    marker=dict(size=7, symbol="circle" if regime == 0 else "diamond"),
                ),
                row=1,
                col=2,
            )
    figure.update_xaxes(title_text="k", row=1, col=1)
    figure.update_yaxes(title_text="score", row=1, col=1)
    figure.update_xaxes(title_text="PC1", row=1, col=2)
    figure.update_yaxes(title_text="PC2", row=1, col=2)
    figure.update_layout(
        title="Q3B latent states of the final disciplined LSTM",
        template="plotly_white",
        hovermode="closest",
    )
    return figure


def write_html(figure: go.Figure, path: Path) -> None:
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = q3_config(args)
    log = RunLog(args.log_path)
    start_time = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log.write(f"start args={vars(args)}")
    log.write(f"torch threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    raw, meta, lags, series = q3.load_inputs(args.panel, args.meta)
    aligned = q3.align_information_set(raw, lags)
    meta_zero = zero_lag_meta(meta)
    aligned_zero = q3.align_information_set(raw, {name: 0 for name in lags})
    d = q3.build_label(aligned, cfg.threshold, "two_month")
    d_zero = q3.build_label(aligned_zero, cfg.threshold, "two_month")
    if not np.allclose(np.nan_to_num(d, nan=-1.0), np.nan_to_num(d_zero, nan=-1.0)):
        raise ValueError("zero-lag label differs from disciplined label")
    T = len(aligned)
    origins = np.arange(cfg.min_train, T - 1)
    test_start = int(cfg.min_train + cfg.tuning_months)
    tuning_origins = origins[origins < test_start]
    test_origins = origins[origins >= test_start]
    if len(tuning_origins) != cfg.tuning_months:
        raise ValueError("tuning block does not match requested length")
    if len(test_origins) == 0:
        raise ValueError("empty test block")
    seeds = [int(args.seed) + offset for offset in range(int(args.n_seeds))]
    log.write(
        f"panel rows={T}  origins={len(origins)}  tuning={len(tuning_origins)}  "
        f"test={len(test_origins)}  seeds={seeds}"
    )

    quarters = q3.quarter_records(aligned)
    chosen_rank, rank_summary = q3.select_rank_by_nowcast(aligned, quarters, tuning_origins, cfg)
    log.write(f"approach5 PCA rank selected on tuning block: {chosen_rank}")
    log.check_budget(args.time_budget_minutes, "probit rank selection")

    tuning_rows = []
    selected_setting = None
    selected_hp = None
    best_brier = float("inf")
    for setting, hp in enumerate(SETTINGS, start=1):
        seed_metrics = {}
        for seed in seeds:
            run = run_lstm_walk(
                aligned, d, tuning_origins, cfg, seed, hp, args, "tuning_lstm", log=log
            )
            metrics = q3.classification_metrics(run["frame"]["y_true"], run["frame"]["p"])
            seed_metrics[seed] = metrics
            if run["failed"]:
                log.write(f"tuning setting {setting} seed {seed} failed fits: {run['failed']}")
        brier_values = np.asarray([seed_metrics[seed]["Brier"] for seed in seeds], dtype=float)
        auc_values = np.asarray([seed_metrics[seed]["AUC"] for seed in seeds], dtype=float)
        row = {
            "setting": int(setting),
            "hidden": int(hp["hidden"]),
            "layers": int(hp["layers"]),
            "dropout": float(hp["dropout"]),
            "Brier_mean": float(np.nanmean(brier_values)),
            "Brier_std": float(np.nanstd(brier_values, ddof=0)),
            "AUC_mean": float(np.nanmean(auc_values)),
            "AUC_std": float(np.nanstd(auc_values, ddof=0)),
            "per_seed": {str(seed): seed_metrics[seed] for seed in seeds},
        }
        tuning_rows.append(row)
        log.write(
            f"tuning setting {setting} {hp}: Brier={row['Brier_mean']:.4f} "
            f"+- {row['Brier_std']:.4f} AUC={row['AUC_mean']:.4f}"
        )
        if row["Brier_mean"] < best_brier - 1e-12:
            best_brier = row["Brier_mean"]
            selected_setting = setting
            selected_hp = dict(hp)
        log.check_budget(args.time_budget_minutes, f"tuning setting {setting}")
    if selected_hp is None:
        raise RuntimeError("no LSTM setting could be evaluated")
    log.write(f"chosen setting {selected_setting}: {selected_hp}")
    console.print(tuning_table(tuning_rows, selected_setting))

    probit_walk = q3.walk_forward_probit(
        aligned, d, origins, cfg, chosen_rank, q3.MAIN_CONTROLS, test_start
    )
    probit_zero_walk = q3.walk_forward_probit(
        aligned_zero, d_zero, origins, cfg, chosen_rank, q3.MAIN_CONTROLS, test_start
    )
    log.write("probit baselines finished")
    log.check_budget(args.time_budget_minutes, "probit walk-forward")

    lstm_runs = {}
    run_meta = []
    for arm, arm_aligned, arm_labels, full_scaler in (
        ("lstm_disciplined", aligned, d, False),
        ("lstm_zero_lag", aligned_zero, d_zero, False),
        ("lstm_zero_lag_full_sample", aligned_zero, d_zero, True),
    ):
        lstm_runs[arm] = {}
        for seed in seeds:
            run = run_lstm_walk(
                arm_aligned,
                arm_labels,
                origins,
                cfg,
                seed,
                selected_hp,
                args,
                arm,
                full_sample_scaler=full_scaler,
                log=log,
            )
            lstm_runs[arm][seed] = run
            run_meta.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "n_refits": int(len(run["refits"])),
                    "n_failed": int(len(run["failed"])),
                    "failed": run["failed"],
                }
            )
            log.write(
                f"{arm} seed {seed}: {len(run['refits'])} refits, {len(run['failed'])} failed"
            )
            log.check_budget(args.time_budget_minutes, f"{arm} seed {seed}")

    test_probit = probit_walk[probit_walk["block"] == "test"].sort_values("origin_index").reset_index(drop=True)
    test_probit_zero = probit_zero_walk[
        probit_zero_walk["block"] == "test"
    ].sort_values("origin_index").reset_index(drop=True)
    y_test = test_probit["y_true"].to_numpy(dtype=float)
    origin_index = test_probit["origin_index"].to_numpy(dtype=int)
    target_dates = pd.DatetimeIndex(test_probit["target_date"])
    origin_dates = pd.DatetimeIndex(test_probit["origin_date"])
    if not np.allclose(
        test_probit_zero["y_true"].to_numpy(dtype=float), y_test, equal_nan=True
    ):
        raise ValueError("test targets differ between lag arms")

    arm_predictions: dict[str, dict[int, np.ndarray]] = {
        "probit": {int(args.seed): test_probit["p_probit"].to_numpy(dtype=float)},
        "probit_zero_lag": {int(args.seed): test_probit_zero["p_probit"].to_numpy(dtype=float)},
        "prior": {int(args.seed): test_probit["p_prior"].to_numpy(dtype=float)},
        "persistence": {int(args.seed): test_probit["p_persistence"].to_numpy(dtype=float)},
    }
    for arm in ("lstm_disciplined", "lstm_zero_lag", "lstm_zero_lag_full_sample"):
        arm_predictions[arm] = {}
        for seed in seeds:
            frame = lstm_runs[arm][seed]["frame"]
            test_frame = frame[frame["block"] == "test"].set_index("origin_index")
            test_frame = test_frame.reindex(origin_index)
            if not np.allclose(
                test_frame["y_true"].to_numpy(dtype=float), y_test, equal_nan=True
            ):
                raise ValueError(f"{arm} seed {seed} target rows do not align with probit")
            arm_predictions[arm][seed] = test_frame["p"].to_numpy(dtype=float)

    comparison = {}
    stochastic = {"lstm_disciplined", "lstm_zero_lag", "lstm_zero_lag_full_sample"}
    for arm in ARM_ORDER:
        per_seed = {
            seed: q3.classification_metrics(y_test, values)
            for seed, values in arm_predictions[arm].items()
        }
        comparison[arm] = summarise_arm(per_seed)
        comparison[arm]["stochastic"] = arm in stochastic
    console.print(comparison_table(comparison))

    seed_mean = {
        arm: np.nanmean(
            np.vstack([arm_predictions[arm][seed] for seed in sorted(arm_predictions[arm])]), axis=0
        )
        for arm in ARM_ORDER
    }
    leakage = {}
    for name, leaked_arm, disciplined_arm in LEAK_PAIRS:
        bootstrap = block_bootstrap_premium(
            y_test,
            seed_mean[leaked_arm],
            seed_mean[disciplined_arm],
            args.bootstrap_reps,
            args.bootstrap_block,
            args.seed,
        )
        leakage[name] = {
            "leaked_arm": leaked_arm,
            "disciplined_arm": disciplined_arm,
            "Brier_leaked": float(np.mean((seed_mean[leaked_arm] - y_test) ** 2)),
            "Brier_disciplined": float(np.mean((seed_mean[disciplined_arm] - y_test) ** 2)),
            **bootstrap,
        }
    console.print(leakage_table(leakage))

    first_seed_run = lstm_runs["lstm_disciplined"][int(args.seed)]
    latent_rows, labels_by_k, hidden_test, latent_y, latent_dates = latent_month_analysis(
        first_seed_run["hidden"], first_seed_run["frame"], args
    )
    console.print(latent_table(latent_rows))
    raw_cluster_labels, windows = q3.cluster_reference_panel(raw, cfg)
    if not windows:
        raise ValueError("approach5 clustering produced no windows")
    latest = windows[-1]
    hidden_by_origin = {
        int(row.origin_index): first_seed_run["hidden"][index]
        for index, row in enumerate(first_seed_run["frame"].itertuples())
    }
    bridge = bridge_to_approach5_clusters(hidden_by_origin, aligned, latest, args)
    console.print(
        rich_table(
            "Bridge to Option A latest indicator clusters",
            ["item", "value"],
            [
                ["latest window", f"{bridge['window_start']} to {bridge['window_end']}"],
                ["k (approach5 latest)", bridge["k"]],
                ["indicators", bridge["n_indicators"]],
                ["ARI vs approach5 labels", f"{bridge['ARI']:.4f}"],
                ["NMI vs approach5 labels", f"{bridge['NMI']:.4f}"],
                ["construction", bridge["construction"]],
            ],
        )
    )

    failed_fits = []
    for meta_row in run_meta:
        failed_fits.extend(meta_row["failed"])
    history = {
        "timings": {"total_seconds": float(time.perf_counter() - start_time)},
        "seeds": seeds,
        "chosen_setting": int(selected_setting),
        "chosen_hyperparameters": selected_hp,
        "tuning": tuning_rows,
        "refits": [refit for arm in lstm_runs for seed in seeds for refit in lstm_runs[arm][seed]["refits"]],
        "run_summary": run_meta,
        "failed_fits": failed_fits,
        "warnings": [
            "NBS PMI has L_i=0 in both lag arms and the label is built from PMI, so the lag leakage premium is structurally limited.",
        ],
    }
    elapsed = time.perf_counter() - start_time
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    for arm in ARM_ORDER:
        for seed, values in arm_predictions[arm].items():
            frame = lstm_runs[arm][seed]["frame"] if arm in stochastic else None
            for position, origin in enumerate(origin_index):
                row = {
                    "arm": arm,
                    "stochastic": bool(arm in stochastic),
                    "seed": int(seed),
                    "origin_index": int(origin),
                    "origin_date": origin_dates[position],
                    "target_date": target_dates[position],
                    "block": "test",
                    "y_true": float(y_test[position]),
                    "p": float(values[position]),
                }
                if frame is not None:
                    match = frame[frame["origin_index"] == origin].iloc[0]
                    row["p_raw"] = float(match["p_raw"])
                    row["refit_origin_index"] = int(match["refit_origin_index"])
                    row["feature_dim"] = int(match["feature_dim"])
                    row["pos_weight"] = float(match["pos_weight"])
                else:
                    row["p_raw"] = float("nan")
                    row["refit_origin_index"] = (
                        int(origin) if arm.startswith("probit") else -1
                    )
                    row["feature_dim"] = (
                        int(chosen_rank + len(q3.MAIN_CONTROLS) + 1)
                        if arm.startswith("probit")
                        else float("nan")
                    )
                    row["pos_weight"] = float("nan")
                prediction_rows.append(row)
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)

    hidden_frame = pd.DataFrame(first_seed_run["hidden"], columns=[f"h{index + 1}" for index in range(first_seed_run["hidden"].shape[1])])
    hidden_frame.insert(0, "origin_index", first_seed_run["frame"]["origin_index"].to_numpy())
    hidden_frame.insert(1, "origin_date", first_seed_run["frame"]["origin_date"].to_numpy())
    hidden_frame.insert(2, "target_date", first_seed_run["frame"]["target_date"].to_numpy())
    hidden_frame.insert(3, "block", first_seed_run["frame"]["block"].to_numpy())
    hidden_frame["d_t"] = first_seed_run["frame"]["d_t"].to_numpy()
    hidden_frame["y_next"] = first_seed_run["frame"]["y_true"].to_numpy()
    hidden_frame["model_refit_origin_index"] = first_seed_run["frame"]["refit_origin_index"].to_numpy()
    label_lookup = {}
    for index, origin in enumerate(latent_dates):
        label_lookup[pd.Timestamp(origin)] = index
    for k, labels in labels_by_k.items():
        values = []
        for timestamp in hidden_frame["target_date"]:
            position = label_lookup.get(pd.Timestamp(timestamp))
            values.append(int(labels[position]) if position is not None else -1)
        hidden_frame[f"kmeans_k{k}"] = values
    hidden_frame.to_csv(args.out_dir / "hidden_states.csv", index=False)

    params = {
        "approach": "Q3B",
        "seed": int(args.seed),
        "seeds": seeds,
        "panel": str(args.panel.relative_to(ROOT)) if args.panel.is_relative_to(ROOT) else str(args.panel),
        "meta": str(args.meta.relative_to(ROOT)) if args.meta.is_relative_to(ROOT) else str(args.meta),
        "config": q3.to_jsonable(asdict(cfg)),
        "information_set": "F_t = {x_{i,s}: s + L_i <= t}; disciplined arms shift each series by L_i",
        "label": {
            "definition": "d_t = 1{PMI_t < threshold and PMI_{t-1} < threshold}",
            "threshold": float(cfg.threshold),
            "rule": "two_month",
        },
        "lags": lags,
        "lstm": {
            "architecture": "LSTM at most 2 layers, dropout applied before a linear logit head",
            "window_months": int(args.window),
            "input": "12 months of the standardized aligned panel, x_{t-11:t}, excluding gdp_yoy",
            "excluded_series": ["gdp_yoy"],
            "refit_scheme": "expanding walk-forward, refit every 6 months",
            "differs_from_approach5": "approach5 refits the probit every month; this refits the LSTM every 6 months and reuses the latest refit between refits",
            "standardisation": "at each refit, means and standard deviations are fitted on rows public at that refit origin (<= t); the same scaler is applied at later origins until the next refit, so no data after the prediction origin is read",
            "validation": "the last 18 training-pair months before each refit origin are held out for early stopping",
            "training_pairs": "(window ending s, d_{s+1}) with s + 1 <= origin t",
            "selected_setting": int(selected_setting),
            "hyperparameters": selected_hp,
            "grid": SETTINGS,
            "selection_criterion": "lowest mean Brier across 3 seeds on the 36-month tuning block only",
            "tuning_table": tuning_rows,
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "optimizer": "Adam",
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch": "full batch",
            "pos_weight": "n_neg / n_pos computed on the fitting pairs only (validation block excluded)",
            "odds_correction": "p_cal = p_w / (p_w + w*(1-p_w)), equivalently odds_true = odds_weighted / w, with w=pos_weight; implemented as sigmoid(logit_w - log(w))",
        },
        "leakage_arms": {
            "lstm_disciplined": "aligned by L_i, origin-only standardization",
            "lstm_zero_lag": "every series at its reference month, L_i=0, origin-only standardization",
            "lstm_zero_lag_full_sample": "L_i=0 plus full-sample standardization (stronger leakage arm)",
            "probit": "approach5 probit baseline, monthly refit, aligned panel",
            "probit_zero_lag": "approach5 probit specification on the L_i=0 panel",
            "prior": "expanding training-block frequency",
            "persistence": "d_t",
        },
        "bootstrap": {
            "scheme": "circular 12-month block bootstrap on paired monthly squared-error differences",
            "replicates": int(args.bootstrap_reps),
            "block_months": int(args.bootstrap_block),
            "seed": int(args.seed),
            "premium_definition": "Brier(leaked arm) - Brier(disciplined twin) using seed-averaged probabilities; positive means leakage worsens Brier",
        },
        "validation": {
            "scheme": "expanding walk-forward; LSTM refit every 6 months, probit monthly",
            "min_train_months": int(cfg.min_train),
            "tuning_months": int(cfg.tuning_months),
            "test_block_only_for_reported_metrics": True,
            "test_dates": {
                "origin_start": str(origin_dates[0].date()),
                "origin_end": str(origin_dates[-1].date()),
                "target_start": str(target_dates[0].date()),
                "target_end": str(target_dates[-1].date()),
                "origins": int(len(test_origins)),
            },
        },
        "latent": {
            "model": "final disciplined LSTM of the first seed",
            "month_clustering": "k-means on standardized hidden states h_t for k=2..6, test months",
            "bridge": bridge["construction"],
        },
    }
    metrics = {
        "label": {
            "full_sample_regime_share": float(np.nanmean(d)),
            "test_class_share": float(comparison["probit"]["class_share"]),
            "test_positives": int(comparison["probit"]["positives"]),
        },
        "validation": params["validation"],
        "rank_selection": {
            "chosen_rank": int(chosen_rank),
            "criterion": "approach5 walk-forward CV nowcast RMSE at month 3 on the tuning block",
            "table": rank_summary.to_dict(orient="records"),
        },
        "comparison": comparison,
        "leakage": leakage,
        "latent": {
            "month_kmeans": latent_rows,
            "bridge": bridge,
            "note": "Option A partitions indicators; month-level h_t clusters partition months, so their direct ARI is undefined.",
        },
        "failed_fits": failed_fits,
        "limitations": [
            "NBS PMI has L_i=0 in both lag arms and the label is built from PMI, which limits the measured lag leakage premium.",
            "The final LSTM embeds test months in-sample because it is trained through 2026-09; latent results are descriptive.",
        ],
    }
    result = {
        "params": q3.to_jsonable(params),
        "metrics": q3.to_jsonable(metrics),
        "fit_seconds": float(elapsed),
        "history": q3.to_jsonable(history),
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html(probabilities_figure(predictions, comparison), args.out_dir / "probabilities.html")
    write_html(
        latent_figure(latent_rows, hidden_test, latent_y, labels_by_k, args),
        args.out_dir / "latent.html",
    )
    log.write(f"wrote {args.out_dir}")
    log.write(f"total fit_seconds={elapsed:.2f}")
    console.print(
        rich_table(
            "Run summary",
            ["item", "value"],
            [
                ["seeds", str(seeds)],
                ["chosen setting", str(selected_setting)],
                ["chosen hyperparameters", str(selected_hp)],
                ["failed fits", str(len(failed_fits))],
                ["runtime seconds", f"{elapsed:.2f}"],
            ],
        )
    )
    terminal_path = args.log_path.with_name("q3b_terminal.txt")
    terminal_path.write_text(console.export_text(), encoding="utf-8")
    log.close()
    return 0


def rich_table(title: str, columns: list[str], rows: list[list]) -> Table:
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*[("" if value is None else str(value)) for value in row])
    return table


if __name__ == "__main__":
    raise SystemExit(main())
