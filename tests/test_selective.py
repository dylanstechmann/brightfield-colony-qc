from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from colonyqc.cli import demo, main, score_path
from colonyqc.selective import (
    ABSTAIN_LABEL,
    DEFAULT_ABSTAIN_THRESHOLD,
    confidence,
    confidence_ranks_correctness,
    risk_coverage_curve,
    selective_calls,
    selective_metrics,
)


class SelectiveRuleTests(unittest.TestCase):
    def test_confidence_is_the_top_label_probability(self):
        probabilities = np.array([[0.7, 0.2, 0.1], [0.34, 0.33, 0.33]])
        self.assertEqual(confidence(probabilities).tolist(), [0.7, 0.34])

    def test_confidence_rejects_malformed_arrays(self):
        for bad in (np.array([]), np.array([0.5, 0.5]), np.array([[0.5, np.nan]]),
                    np.array([[-0.1, 1.1]])):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    confidence(bad)

    def test_calls_are_withheld_below_the_threshold(self):
        probabilities = np.array([[0.9, 0.1], [0.55, 0.45]])
        calls = selective_calls(["clean", "debris"], probabilities, threshold=0.6)
        self.assertEqual(calls, ["clean", ABSTAIN_LABEL])
        self.assertEqual(selective_calls(["clean", "debris"], probabilities, threshold=0.0),
                         ["clean", "debris"])
        with self.assertRaises(ValueError):
            selective_calls(["clean"], probabilities, threshold=0.6)
        with self.assertRaises(ValueError):
            selective_calls(["clean", "debris"], probabilities, threshold=1.5)

    def test_metrics_separate_coverage_from_accuracy(self):
        truth = ["a", "a", "b", "b"]
        predicted = ["a", "b", "b", "a"]
        probabilities = np.array([[0.95, 0.05], [0.45, 0.55], [0.1, 0.9], [0.52, 0.48]])
        result = selective_metrics(truth, predicted, probabilities, threshold=0.6)
        self.assertEqual(result["n_answered"], 2)
        self.assertEqual(result["n_abstained"], 2)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["selective_accuracy"], 1.0)
        self.assertEqual(result["selective_error_rate"], 0.0)
        self.assertEqual(result["full_coverage_accuracy"], 0.5)
        # Both abstained rows were wrong anyway, so deferring them was precise.
        self.assertEqual(result["abstained_would_have_been_correct"], 0)
        self.assertEqual(result["abstention_precision"], 1.0)

    def test_answering_nothing_reports_no_accuracy_rather_than_one(self):
        result = selective_metrics(["a"], ["a"], np.array([[0.5, 0.5]]), threshold=0.9)
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["selective_accuracy"])
        self.assertIsNone(result["selective_error_rate"])
        self.assertEqual(result["full_coverage_accuracy"], 1.0)

    def test_metrics_reject_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            selective_metrics(["a", "b"], ["a"], np.array([[0.9, 0.1], [0.8, 0.2]]))
        with self.assertRaises(ValueError):
            selective_metrics(["a"], ["a"], np.array([[0.9, 0.1]]), threshold=-0.1)

    def test_curve_covers_fixed_thresholds_and_states_its_method(self):
        truth = ["a", "b", "a", "b"]
        predicted = ["a", "b", "b", "a"]
        probabilities = np.array([[0.99, 0.01], [0.01, 0.99], [0.45, 0.55], [0.55, 0.45]])
        curve = risk_coverage_curve(truth, predicted, probabilities)
        thresholds = [point["threshold"] for point in curve["points"]]
        self.assertEqual(thresholds, sorted(thresholds))
        self.assertEqual(curve["points"][0]["coverage"], 1.0)
        self.assertIn("no threshold was selected using these same rows", curve["method"])
        self.assertTrue(any("never alone" in item for item in curve["limitations"]))
        with self.assertRaises(ValueError):
            risk_coverage_curve(truth, predicted, probabilities, thresholds=[1.2])

    def test_diagnostic_detects_informative_confidence(self):
        truth = ["a"] * 10
        predicted = ["a"] * 5 + ["b"] * 5
        probabilities = np.array([[0.95, 0.05]] * 5 + [[0.5, 0.5]] * 5)
        curve = risk_coverage_curve(truth, predicted, probabilities)
        diagnostic = curve["confidence_diagnostic"]
        self.assertEqual(diagnostic["status"], "improves")
        self.assertGreater(diagnostic["selective_accuracy_gain"], 0)
        self.assertIn("ranks correctness", diagnostic["detail"])

    def test_diagnostic_detects_anticorrelated_confidence(self):
        # The confident rows are the wrong ones, so abstaining makes accuracy worse.
        truth = ["a"] * 10
        predicted = ["b"] * 5 + ["a"] * 5
        probabilities = np.array([[0.95, 0.05]] * 5 + [[0.5, 0.5]] * 5)
        diagnostic = risk_coverage_curve(truth, predicted, probabilities)["confidence_diagnostic"]
        self.assertEqual(diagnostic["status"], "does_not_improve")
        self.assertLess(diagnostic["selective_accuracy_gain"], 0)
        self.assertIn("buys nothing", diagnostic["detail"])

    def test_diagnostic_calls_out_a_ceiling_instead_of_crediting_abstention(self):
        truth = predicted = ["a"] * 6
        probabilities = np.array([[0.9, 0.1]] * 3 + [[0.5, 0.5]] * 3)
        diagnostic = risk_coverage_curve(truth, predicted, probabilities)["confidence_diagnostic"]
        self.assertEqual(diagnostic["status"], "no_headroom")
        self.assertIn("synthetic generator", diagnostic["detail"])

    def test_diagnostic_reports_when_coverage_is_too_low_to_assess(self):
        points = [selective_metrics(["a"], ["a"], np.array([[0.5, 0.5]]), threshold=0.9)]
        self.assertEqual(confidence_ranks_correctness(points)["status"], "not_assessable")


class SelectiveBakeOffTests(unittest.TestCase):
    def test_bake_off_reports_coverage_and_preserves_the_blur_ablation(self):
        result = demo(0)
        self.assertAlmostEqual(result["holdout_accuracy"], 1.0)
        self.assertAlmostEqual(result["blurred_holdout_accuracy"], 0.7667, places=3)
        selective = result["selective_prediction"]
        self.assertEqual(selective["abstain_threshold"], DEFAULT_ABSTAIN_THRESHOLD)
        self.assertEqual(selective["clean"]["coverage"], 1.0)
        self.assertLess(selective["blurred"]["coverage"], 1.0)
        self.assertAlmostEqual(selective["blurred"]["full_coverage_accuracy"],
                               result["blurred_holdout_accuracy"], places=4)
        self.assertIn("read it with its", result["note"])
        json.dumps(result, allow_nan=False)

    def test_blurred_confidence_is_reported_as_uninformative(self):
        # A real negative finding: on blurred fields the model's confidence does not
        # rank correctness, so abstention does not help. The report must say so.
        selective = demo(0)["selective_prediction"]
        self.assertEqual(selective["clean_risk_coverage"]["confidence_diagnostic"]["status"],
                         "no_headroom")
        blurred = selective["blurred_risk_coverage"]["confidence_diagnostic"]
        self.assertEqual(blurred["status"], "does_not_improve")
        self.assertLess(blurred["selective_accuracy_gain"], 0)


class SelectivePredictTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.json"
        self.assertEqual(main(["train", "--out", str(self.model), "--seed", "0"]), 0)
        from colonyqc.synthetic import dataset, write_pgm

        images, _labels = dataset(n_per_class=1, seed=1)
        self.image = self.root / "field.pgm"
        write_pgm(str(self.image), images[0])

    def test_threshold_zero_answers_and_threshold_one_abstains(self):
        answered = score_path(str(self.model), str(self.image),
                              input_domain="synthetic-demo", abstain_threshold=0.0)
        self.assertFalse(answered["selective_prediction"]["abstained"])
        self.assertIsNotNone(answered["demonstration_class"])
        self.assertNotIn("abstained_low_confidence", answered["flags"])

        withheld = score_path(str(self.model), str(self.image),
                              input_domain="synthetic-demo", abstain_threshold=1.0)
        selective = withheld["selective_prediction"]
        self.assertTrue(selective["abstained"])
        self.assertIsNone(withheld["demonstration_class"])
        self.assertEqual(selective["withheld_demonstration_class"], answered["demonstration_class"])
        self.assertIn("abstained_low_confidence", withheld["flags"])
        self.assertIn("review this field manually", withheld["next_human_step"])
        self.assertIn("not a statement about a culture", selective["note"])

    def test_confidence_is_recorded_and_probabilities_are_unchanged_by_abstaining(self):
        answered = score_path(str(self.model), str(self.image),
                              input_domain="synthetic-demo", abstain_threshold=0.0)
        withheld = score_path(str(self.model), str(self.image),
                              input_domain="synthetic-demo", abstain_threshold=1.0)
        self.assertEqual(answered["probabilities"], withheld["probabilities"])
        self.assertEqual(answered["selective_prediction"]["top_label_confidence"],
                         withheld["selective_prediction"]["top_label_confidence"])
        self.assertEqual(max(answered["probabilities"].values()),
                         round(answered["selective_prediction"]["top_label_confidence"], 4))

    def test_rejected_inputs_do_not_gain_a_selective_block(self):
        unverified = score_path(str(self.model), str(self.image), input_domain="unverified")
        self.assertEqual(unverified["prediction_status"], "unscorable")
        self.assertNotIn("selective_prediction", unverified)

    def test_cli_validates_the_threshold(self):
        with self.assertRaises(SystemExit):
            main(["predict", "--model", str(self.model), "--image", str(self.image),
                  "--input-domain", "synthetic-demo", "--abstain-threshold", "1.5"])


if __name__ == "__main__":
    unittest.main()
