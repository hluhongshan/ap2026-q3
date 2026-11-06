#!/usr/bin/env python3
"""Cross-approach comparison tables, read only from results/**/metrics.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_HTML = RESULTS / "comparison.html"

console = Console()


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_pm(mean, std, digits: int = 4) -> str:
    if mean is None:
        return "n/a"
    if std is None or not math.isfinite(float(std)):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} ± {fmt(std, digits)}"


def load_all() -> list[dict]:
    paths = sorted(RESULTS.glob("approach*/*/metrics.json"))
    if not paths:
        raise SystemExit(f"no metrics.json files under {RESULTS}")
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if list(payload) != ["params", "metrics", "fit_seconds", "history"]:
            raise ValueError(f"{path}: unexpected top-level keys {list(payload)}")
        payloads.append(
            {
                "path": path,
                "rel": str(path.relative_to(ROOT)),
                "approach": path.parts[-3],
                "dataset": path.parts[-2],
                "payload": payload,
            }
        )
    return payloads


def coverage_rows(payloads: list[dict], used: dict[str, str]) -> list[list]:
    rows = []
    for item in payloads:
        rows.append(
            [
                item["rel"].removeprefix("results/").removesuffix("/metrics.json"),
                item["approach"],
                item["dataset"],
                fmt(item["payload"]["fit_seconds"], 2),
                used.get(item["rel"], "UNUSED"),
            ]
        )
    unused = [row for row in rows if row[-1] == "UNUSED"]
    if unused:
        raise ValueError(f"metrics.json files excluded silently: {unused}")
    return rows


def rich_table(title: str, columns: list[str], rows: list[list]) -> Table:
    table = Table(title=title, show_lines=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*[str(value) for value in row])
    return table


def make_plotly_table(section_rows: list[list[str]], header_color: str = "#1f4e79") -> go.Table:
    n_cols = len(section_rows[0])
    columns = [[row[i] for row in section_rows[1:]] for i in range(n_cols)]
    return go.Table(
        header=dict(
            values=section_rows[0],
            fill_color=header_color,
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=columns,
            fill_color=["#f7f9fc", "#eef2f7"],
            align="left",
            font=dict(size=11),
            height=24,
        ),
        columnwidth=[1.2] + [1.0] * (n_cols - 1),
    )


def summary_block(block: dict) -> tuple[float, float, float, float, float, float, int | None]:
    if "Brier_mean" in block:
        return (
            block["Brier_mean"],
            block["Brier_std"],
            block["AUC_mean"],
            block["AUC_std"],
            block["accuracy_mean"],
            block["accuracy_std"],
            block.get("n"),
        )
    return (
        block["Brier"],
        0.0,
        block["AUC"],
        0.0,
        block["accuracy"],
        0.0,
        block.get("n"),
    )


def q3_rows(payloads: list[dict]) -> tuple[list[dict], dict[str, str]]:
    rows = []
    used = {}
    for item in payloads:
        metrics = item["payload"]["metrics"]
        if item["approach"] == "approach1":
            if "probit" in metrics and "test" in metrics["probit"]:
                for arm, block in metrics["probit"]["test"].items():
                    brier, brier_std, auc, auc_std, accuracy, accuracy_std, n = summary_block(block)
                    rows.append(
                        {
                            "source": item["rel"],
                            "arm": arm,
                            "vintage": "original",
                            "brier": brier,
                            "brier_std": brier_std,
                            "auc": auc,
                            "auc_std": auc_std,
                            "accuracy": accuracy,
                            "accuracy_std": accuracy_std,
                            "n": n,
                        }
                    )
            post_hoc = metrics.get("post_hoc", {})
            for block in post_hoc.get("test_table", []):
                brier, brier_std, auc, auc_std, accuracy, accuracy_std, n = summary_block(block)
                rows.append(
                    {
                        "source": item["rel"],
                        "arm": block["model"],
                        "vintage": "post_hoc",
                        "brier": brier,
                        "brier_std": brier_std,
                        "auc": auc,
                        "auc_std": auc_std,
                        "accuracy": accuracy,
                        "accuracy_std": accuracy_std,
                        "n": n,
                    }
                )
            used[item["rel"]] = "Q3 original and post_hoc probit arms"
        elif item["approach"] == "approach2":
            for arm, block in metrics.get("comparison", {}).items():
                brier, brier_std, auc, auc_std, accuracy, accuracy_std, n = summary_block(block)
                rows.append(
                    {
                        "source": item["rel"],
                        "arm": arm,
                        "vintage": "original",
                        "brier": brier,
                        "brier_std": brier_std,
                        "auc": auc,
                        "auc_std": auc_std,
                        "accuracy": accuracy,
                        "accuracy_std": accuracy_std,
                        "n": n,
                    }
                )
            retrain = metrics.get("post_hoc", {}).get("retrain", {})
            for arm, block in retrain.get("comparison", {}).items():
                brier, brier_std, auc, auc_std, accuracy, accuracy_std, n = summary_block(block)
                rows.append(
                    {
                        "source": item["rel"],
                        "arm": arm,
                        "vintage": "post_hoc",
                        "brier": brier,
                        "brier_std": brier_std,
                        "auc": auc,
                        "auc_std": auc_std,
                        "accuracy": accuracy,
                        "accuracy_std": accuracy_std,
                        "n": n,
                    }
                )
            used[item["rel"]] = "Q3 original and post_hoc LSTM/probit arms"
    rows.sort(key=lambda row: float(row["brier"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, used


def build_html(coverage: list[list[str]], q3: list[list[str]]) -> go.Figure:
    sections = [
        ("Source coverage: every results/approach*/*/metrics.json", coverage),
        ("Q3: Brier, AUC and accuracy, original and post-hoc arms (ranked by Brier)", q3),
    ]
    counts = [max(1, len(section[1]) - 1) for section in sections]
    total = sum(counts)
    heights = [max(0.08, count / total) for count in counts]
    figure = make_subplots(
        rows=len(sections), cols=1,
        specs=[[{"type": "table"}] for _ in sections],
        subplot_titles=[section[0] for section in sections],
        vertical_spacing=0.035, row_heights=heights,
    )
    for index, section in enumerate(sections, start=1):
        figure.add_trace(make_plotly_table(section[1]), row=index, col=1)
    figure.update_layout(
        title="Q3 cross-approach comparison (all values read from metrics.json)",
        height=max(900, 320 + 34 * sum(counts)),
        margin=dict(l=20, r=20, t=80, b=20), template="plotly_white",
    )
    figure.update_annotations(font_size=13)
    return figure


def main() -> int:
    payloads = load_all()
    used: dict[str, str] = {}
    q3, used_q3 = q3_rows(payloads)
    used.update(used_q3)
    coverage = coverage_rows(payloads, used)
    console.print(rich_table(
        "Source coverage: every results/approach*/*/metrics.json",
        ["metrics.json", "approach", "dataset", "fit_seconds", "used for"], coverage,
    ))
    q3_table = [[
        row["rank"], row["source"].removeprefix("results/").rsplit("/", 1)[0], row["arm"],
        row["vintage"], fmt_pm(row["brier"], row["brier_std"]),
        fmt_pm(row["auc"], row["auc_std"]), fmt_pm(row["accuracy"], row["accuracy_std"]), row["n"],
    ] for row in q3]
    console.print(rich_table(
        "Q3: Brier, AUC and accuracy, original and post-hoc arms (ranked by Brier)",
        ["rank", "source", "arm", "vintage", "Brier", "AUC", "accuracy", "n"], q3_table,
    ))
    figure = build_html(coverage, q3_table)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(OUT_HTML, include_plotlyjs=True, full_html=True)
    console.print(f"wrote {OUT_HTML.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
