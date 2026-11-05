"""Leakage, label, calibration, signal and determinism tests for Q3 Option B."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

Q3A_SPEC = importlib.util.spec_from_file_location(
    "approach5_q3a_classical", CODE / "approach5_q3a_classical.py"
)
assert Q3A_SPEC is not None and Q3A_SPEC.loader is not None
Q3A = importlib.util.module_from_spec(Q3A_SPEC)
sys.modules[Q3A_SPEC.name] = Q3A
Q3A_SPEC.loader.exec_module(Q3A)

Q3B_SPEC = importlib.util.spec_from_file_location(
    "approach6_q3b_sequence", CODE / "approach6_q3b_sequence.py"
)
assert Q3B_SPEC is not None and Q3B_SPEC.loader is not None
Q3B = importlib.util.module_from_spec(Q3B_SPEC)
sys.modules[Q3B_SPEC.name] = Q3B
Q3B_SPEC.loader.exec_module(Q3B)

PANEL = ROOT / "data" / "clean" / "macro_panel.csv"
META = ROOT / "data" / "clean" / "macro_meta.csv"


def synthetic_panel(months: int = 200) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    state = np.zeros(months)
    for index in range(1, months):
        state[index] = 0.8 * state[index - 1] + rng.normal(scale=0.6)
    pmi = 50.0 + 3.0 * state + rng.normal(scale=1.0, size=months)
    aligned = pd.DataFrame(
        {
            "date": dates,
            "pmi": pmi,
            "signal": state + rng.normal(scale=0.1, size=months),
            "noise": rng.normal(size=months),
            "other": rng.normal(size=months),
        }
    )
    label = Q3A.build_label(aligned, 50.0, "two_month")
    return aligned, label


class TestQ3BLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.meta, cls.lags, cls.series = Q3A.load_inputs(PANEL, META)
        cls.aligned = Q3A.align_information_set(cls.raw, cls.lags)
        cls.cfg = Q3A.RunConfig()
        cls.origin = 120
        cls.window = 12

    def test_windows_at_origin_ignore_future_values(self):
        fit_idx = np.arange(0, self.origin + 1)
        means, stds, columns = Q3B.fit_feature_scaler(self.aligned, fit_idx, 0.5)
        baseline = Q3B.standardised_window(
            self.aligned, self.origin, self.window, columns, means, stds
        )
        mutated = self.raw.copy()
        rows = np.arange(len(mutated))
        for name in self.series:
            lag = self.lags[name]
            future = rows + lag > self.origin
            mutated.loc[future, name] = 1e9
        aligned_mutated = Q3A.align_information_set(mutated, self.lags)
        means_m, stds_m, columns_m = Q3B.fit_feature_scaler(aligned_mutated, fit_idx, 0.5)
        self.assertEqual(columns, columns_m)
        np.testing.assert_array_equal(means, means_m)
        np.testing.assert_array_equal(stds, stds_m)
        perturbed = Q3B.standardised_window(
            aligned_mutated, self.origin, self.window, columns_m, means_m, stds_m
        )
        np.testing.assert_array_equal(baseline, perturbed)

    def test_training_labels_never_exceed_origin(self):
        d = Q3A.build_label(self.aligned, 50.0, "two_month")
        ends = Q3B.training_pair_ends(self.origin, d, self.window)
        self.assertGreater(len(ends), 0)
        self.assertLessEqual(int(np.max(ends + 1)), self.origin)
        self.assertLessEqual(int(np.max(ends)), self.origin - 1)


class TestQ3BCalibration(unittest.TestCase):
    def test_hand_computed_odds_correction(self):
        self.assertAlmostEqual(Q3B.calibrate_probability(0.8, 4.0), 0.5, places=12)
        self.assertAlmostEqual(Q3B.calibrate_probability(0.5, 2.0), 1.0 / 3.0, places=12)
        raw = 0.8
        weight = 4.0
        odds_weighted = raw / (1.0 - raw)
        expected = odds_weighted / weight / (1.0 + odds_weighted / weight)
        self.assertAlmostEqual(Q3B.calibrate_probability(raw, weight), expected, places=12)
        logit = np.log(raw / (1.0 - raw))
        self.assertAlmostEqual(Q3B.calibrate_logit(logit, weight), 0.5, places=12)

    def test_pos_weight_uses_only_training_labels(self):
        weight, n_pos, n_neg = Q3B.positive_weight(np.array([0.0, 0.0, 0.0, 1.0]))
        self.assertAlmostEqual(weight, 3.0, places=12)
        self.assertEqual((n_pos, n_neg), (1, 3))


class TestQ3BSignal(unittest.TestCase):
    def test_lstm_beats_prior_on_synthetic_signal(self):
        aligned, label = synthetic_panel()
        cfg = Q3A.RunConfig(seed=42, min_train=80, tuning_months=40, min_coverage=0.5)
        args = Q3B.parse_args([])
        args.max_epochs = 200
        args.patience = 40
        args.lr = 5e-3
        origins = np.arange(80, len(aligned) - 1)
        hp = {"hidden": 8, "layers": 1, "dropout": 0.0}
        run = Q3B.run_lstm_walk(
            aligned, label, origins, cfg, 42, hp, args, "synthetic", log=None
        )
        test = run["frame"][run["frame"]["block"] == "test"]
        lstm_brier = float(np.mean((test["p"] - test["y_true"]) ** 2))
        prior = np.array(
            [np.nanmean(label[1 : int(origin) + 1]) for origin in test["origin_index"]]
        )
        prior_brier = float(np.mean((prior - test["y_true"].to_numpy()) ** 2))
        self.assertLess(lstm_brier, prior_brier - 0.01)
        self.assertLess(lstm_brier, 0.20)

    def test_min_epochs_blocks_early_stopping(self):
        aligned, label = synthetic_panel()
        means, stds, columns = Q3B.fit_feature_scaler(aligned, np.arange(0, 121), 0.5)
        data = Q3B.build_training_arrays(
            aligned, label, 120, 12, 18, columns, means, stds
        )
        args = Q3B.parse_args([])
        args.max_epochs = 40
        args.patience = 1
        args.min_epochs = 10
        hp = {"hidden": 8, "layers": 1, "dropout": 0.2}
        _, info = Q3B.train_lstm(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"], hp, 7, args
        )
        self.assertGreaterEqual(info["epochs_run"], 10)
        self.assertEqual(info["min_epochs"], 10)

    def test_same_seed_same_result(self):
        aligned, label = synthetic_panel()
        cfg = Q3A.RunConfig(seed=42, min_train=80, tuning_months=0, min_coverage=0.5)
        means, stds, columns = Q3B.fit_feature_scaler(aligned, np.arange(0, 121), 0.5)
        data = Q3B.build_training_arrays(
            aligned, label, 120, 12, 18, columns, means, stds
        )
        args = Q3B.parse_args([])
        args.max_epochs = 40
        args.patience = 20
        hp = {"hidden": 8, "layers": 1, "dropout": 0.2}
        model_a, _ = Q3B.train_lstm(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"], hp, 7, args
        )
        model_b, _ = Q3B.train_lstm(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"], hp, 7, args
        )
        window = Q3B.standardised_window(aligned, 120, 12, columns, means, stds)
        p_a, raw_a, h_a = Q3B.predict_lstm(model_a, window[None], 2.0)
        p_b, raw_b, h_b = Q3B.predict_lstm(model_b, window[None], 2.0)
        np.testing.assert_array_equal(p_a, p_b)
        np.testing.assert_array_equal(raw_a, raw_b)
        np.testing.assert_array_equal(h_a, h_b)


if __name__ == "__main__":
    unittest.main()
