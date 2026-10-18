"""Regression checks for the statistical contracts, not just the script output."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import data
import run_baseline
from risk_model import (
    capacity_threshold, export_model, fit_logistic, predict,
    probability_metrics, reliability_table, review_metrics,
)


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_weighted_intercept_and_odds_correction(self):
        x = torch.zeros((200, 1))
        y = torch.tensor([1.] * 20 + [0.] * 180)
        plain, _ = fit_logistic(x, y)
        weighted, _ = fit_logistic(x, y, 9.0)
        np.testing.assert_allclose(predict(export_model(plain), x), .1, atol=1e-5)
        np.testing.assert_allclose(predict(export_model(weighted), x), .5, atol=1e-5)
        corrected = export_model(weighted, math.log(9))
        np.testing.assert_allclose(predict(corrected, x), .1, atol=1e-5)

    def test_correction_preserves_ranking_and_json_roundtrip(self):
        x = np.arange(-5, 6, dtype=float).reshape(-1, 1)
        spec = {"weights": [1.7], "bias": -.2, "logit_offset": 0.0}
        raw = predict(spec, x)
        spec["logit_offset"] = math.log(20)
        adjusted = predict(json.loads(json.dumps(spec)), x)
        np.testing.assert_array_equal(np.argsort(raw), np.argsort(adjusted))
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1])
        self.assertEqual(probability_metrics(y, raw)["average_precision"],
                         probability_metrics(y, adjusted)["average_precision"])

    def test_capacity_respects_ties_and_constant_scores(self):
        p = np.array([.9, .9, .5, .3, .2, .1, .1, .1, .1, .1])
        self.assertGreater(capacity_threshold(p, .1), 1)
        threshold = capacity_threshold(p, .3)
        self.assertEqual(threshold, .5)
        self.assertEqual(int((p >= threshold).sum()), 3)
        self.assertGreater(capacity_threshold(np.full(100, .03)), 1)

    def test_frozen_threshold_does_not_refill_the_queue(self):
        threshold = capacity_threshold(np.linspace(.01, .99, 100))
        shifted = np.full(100, .1)
        result = review_metrics(np.array([0, 1] * 50), shifted, threshold)
        self.assertEqual(result["reviewed"], 0)
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0)

    def test_metrics_and_bins_include_probability_endpoints(self):
        y, p = np.array([0, 0, 1, 1]), np.array([0., .25, .75, 1.])
        result = probability_metrics(y, p)
        self.assertAlmostEqual(result["brier"], .03125)
        self.assertEqual(result["roc_auc"], 1)
        self.assertEqual(reliability_table(y, p)["rows"].sum(), 4)
        self.assertIsNone(probability_metrics([0, 0], [.1, .2])["roc_auc"])
        with self.assertRaises(ValueError):
            probability_metrics([0, 1], [.1, np.nan])
        with self.assertRaises(ValueError):
            fit_logistic(torch.zeros(3, 1), torch.zeros(3))

    def test_development_rows_cannot_see_future_reports(self):
        months = pd.period_range("2005-01", "2012-12", freq="M")
        performance = pd.DataFrame({
            "loan_id": "FTEST", "period": months, "dq_status": 0.,
            "current_upb": 100_000., "loan_age": np.arange(len(months)),
            "zero_balance_code": "", "exits_in_default": False,
        })
        origination = pd.DataFrame({
            "loan_id": ["FTEST"], "first_payment_month": [months[0]],
            "credit_score": [720.], "orig_ltv": [80.], "orig_dti": [30.],
            "orig_upb": [100_000.], "orig_rate": [5.],
        })
        changed = performance.copy()
        changed.loc[changed["period"] >= pd.Period("2011-01", "M"), "dq_status"] = 6.
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "sample_orig_test.txt").touch()
            (directory / "sample_svcg_test.txt").touch()
            with patch.object(data, "load_origination", return_value=origination):
                with patch.object(data, "load_performance", return_value=performance):
                    original = data.prepare_rows(directory, "2010-12")
                with patch.object(data, "load_performance", return_value=changed):
                    rebuilt = data.prepare_rows(directory, "2010-12")
        pd.testing.assert_frame_equal(original, rebuilt)
        self.assertLessEqual(original["period"].max(), pd.Period("2009-12", "M"))

    def test_second_evaluation_stops_before_loading_test_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "frozen.json").write_text(json.dumps({"code_sha256": "c", "data_sha256": "d"}))
            (output / "test-started.json").write_text("{}")
            with patch.object(run_baseline, "code_fingerprint", return_value="c"), \
                 patch.object(run_baseline, "data_fingerprint", return_value="d"), \
                 patch.object(run_baseline, "prepare_rows") as prepare:
                with self.assertRaises(FileExistsError):
                    run_baseline.evaluate(output, output)
                prepare.assert_not_called()
            with self.assertRaises(FileExistsError):
                run_baseline.develop(output, output)


if __name__ == "__main__":
    unittest.main()
