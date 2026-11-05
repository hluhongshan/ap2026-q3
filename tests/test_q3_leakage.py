"""Leakage, alignment, probit, RMSE and AUC tests for Q3 Option A."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "approach5_q3a_classical.py"
SPEC = importlib.util.spec_from_file_location("approach5_q3a_classical", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
Q3A = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = Q3A
SPEC.loader.exec_module(Q3A)

PANEL = ROOT / "data" / "clean" / "macro_panel.csv"
META = ROOT / "data" / "clean" / "macro_meta.csv"


class TestQ3ALeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.meta, cls.lags, cls.series = Q3A.load_inputs(PANEL, META)
        cls.cfg = Q3A.RunConfig()
        cls.origin = 120
        cls.rank = 3

    def test_alignment_shifts_by_lag(self):
        aligned = Q3A.align_information_set(self.raw, self.lags)
        for name in ("pmi", "gdp_yoy", "cgb10y"):
            lag = self.lags[name]
            left = aligned[name].iloc[50]
            right = self.raw[name].iloc[50 - lag]
            if np.isnan(left) or np.isnan(right):
                self.assertTrue(np.isnan(left) and np.isnan(right))
            else:
                self.assertEqual(left, right)
        lag = self.lags["gdp_yoy"]
        self.assertTrue(np.isnan(aligned["gdp_yoy"].iloc[lag - 1]))

    def test_future_values_cannot_change_z_or_probability(self):
        baseline = Q3A.predict_at_origin(
            self.raw, self.meta, self.origin, self.cfg, self.rank
        )
        mutated = self.raw.copy()
        for name in self.series:
            lag = self.lags[name]
            rows = np.arange(len(mutated))
            future = rows + lag > self.origin
            mutated.loc[future, name] = 1e9
        perturbed = Q3A.predict_at_origin(
            mutated, self.meta, self.origin, self.cfg, self.rank
        )
        np.testing.assert_array_equal(baseline["z_t"], perturbed["z_t"])
        self.assertEqual(baseline["probability"], perturbed["probability"])

    def test_training_labels_never_exceed_origin(self):
        result = Q3A.predict_at_origin(
            self.raw, self.meta, self.origin, self.cfg, self.rank
        )
        self.assertLessEqual(int(np.max(result["train_target_idx"])), self.origin)
        self.assertLessEqual(int(np.max(result["train_feature_idx"])), self.origin - 1)


class TestQ3AEstimators(unittest.TestCase):
    def test_probit_recovers_simulated_coefficients(self):
        rng = np.random.default_rng(2026)
        n = 6000
        x = np.column_stack(
            [np.ones(n), rng.normal(size=(n, 3))]
        )
        beta = np.array([-0.3, 0.8, -0.5, 0.4])
        y = (rng.random(n) < ndtr(x @ beta)).astype(float)
        fit = Q3A.fit_probit(x, y, ridge=1e-6)
        self.assertTrue(fit["converged"])
        self.assertLess(float(np.max(np.abs(fit["beta"] - beta))), 0.15)
        se = Q3A.probit_standard_errors(x, y, fit["beta"], nw_lag=4)
        self.assertTrue(np.all(np.isfinite(se["info_se"])))
        self.assertTrue(np.all(se["info_se"] > 0.0))
        self.assertTrue(np.all(np.isfinite(se["nw_se"])))
        self.assertTrue(np.all(se["nw_se"] > 0.0))

    def test_rmse(self):
        self.assertAlmostEqual(Q3A.rmse([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]), np.sqrt(4.0 / 3.0))
        self.assertEqual(Q3A.rmse([1.0, np.nan], [1.0, 2.0]), 0.0)

    def test_auc_matches_sklearn_with_ties(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.35])
        expected = roc_auc_score(y, p)
        self.assertAlmostEqual(Q3A.auc_score(y, p), expected, places=12)
        y2 = np.array([0, 1, 0, 1])
        p2 = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(Q3A.auc_score(y2, p2), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
