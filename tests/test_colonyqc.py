import os
import tempfile
import unittest

import numpy as np

from colonyqc.cli import demo, score_path, training_domain
from colonyqc.features import FEATURE_NAMES, featurize
from colonyqc.model import SoftmaxQC, accuracy
from colonyqc.report import DISCLAIMER, build_report
from colonyqc.synthetic import LABELS, box_blur, dataset, read_pgm, render, write_pgm


class FeatureTests(unittest.TestCase):
    def test_vector_shape_and_finite(self):
        img = render(np.random.default_rng(0), "undifferentiated")
        feat = featurize(img)
        self.assertEqual(feat.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(feat).all())

    def test_blank_black_white_and_near_uniform_fields_are_rejected(self):
        for image in [np.zeros((32, 32)), np.ones((32, 32)), np.full((32, 32), 0.5)]:
            with self.assertRaisesRegex(ValueError, "blank or near-uniform"):
                featurize(image)

    def test_filaments_are_thinner_than_colonies(self):
        rng = np.random.default_rng(1)
        colony = np.mean([featurize(render(rng, "undifferentiated"))[FEATURE_NAMES.index("thin_fraction")] for _ in range(15)])
        hypha = np.mean([featurize(render(rng, "contamination_suspect"))[FEATURE_NAMES.index("thin_fraction")] for _ in range(15)])
        self.assertGreater(hypha, colony + 0.02)


class ModelTests(unittest.TestCase):
    def test_beats_majority_and_blur_hurts(self):
        report = demo(0)
        self.assertGreater(report["holdout_accuracy"], report["majority_baseline"] + 0.5)
        self.assertGreater(report["holdout_accuracy"], 0.9)
        self.assertLess(report["blurred_holdout_accuracy"], report["holdout_accuracy"] - 0.05)

    def test_roundtrip_and_pgm(self):
        images, labels = dataset(n_per_class=20, seed=2)
        x = np.vstack([featurize(im) for im in images])
        model = SoftmaxQC().fit(x, labels, epochs=300, seed=2)
        self.assertGreater(accuracy(labels, model.predict(x)), 0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            pgm = os.path.join(tmp, "f.pgm")
            model.save(path)
            loaded = SoftmaxQC.load(path)
            self.assertEqual(loaded.training_provenance["source"], "unknown_not_recorded")
            write_pgm(pgm, images[0])
            restored = read_pgm(pgm)
            self.assertEqual(restored.shape, images[0].shape)
            pred = loaded.predict(featurize(restored).reshape(1, -1))[0]
            self.assertIn(pred, LABELS)


class ReportTests(unittest.TestCase):
    def test_disclaimer_and_contamination_flag(self):
        proba = {name: 0.0 for name in LABELS}
        proba["contamination_suspect"] = 0.8
        report = build_report("contamination_suspect", proba, [0.0] * len(FEATURE_NAMES))
        self.assertEqual(report["disclaimer"], DISCLAIMER)
        self.assertIn("contamination_triage", report["flags"])
        self.assertIn("mycoplasma", report["next_human_step"].lower())

    def test_html_report_generation(self):
        from colonyqc.report import generate_html_report
        proba = {name: 0.25 for name in LABELS}
        proba["undifferentiated"] = 0.7
        features = [0.12] * len(FEATURE_NAMES)
        report = build_report("undifferentiated", proba, features)
        img = np.ones((32, 32), dtype=np.float64) * 0.5
        html = generate_html_report(report, image_array=img, image_name="test_colony.png")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("undifferentiated", html)
        self.assertIn("fg_fraction", html)
        self.assertIn("Regulatory & Research Disclaimer", html)
        self.assertIn("Training data provenance is not confirmed", html)

    def test_training_provenance_roundtrips_and_is_in_prediction_report(self):
        images, labels = dataset(n_per_class=12, seed=3)
        x = np.vstack([featurize(im) for im in images])
        provenance = {
            "source": "synthetic_generator",
            "generator_seed": 3,
            "training_matrix_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.json")
            image_path = os.path.join(tmp, "field.pgm")
            provenance["training_source_sha256"] = "b" * 64
            model = SoftmaxQC().fit(
                x, labels, epochs=100, seed=3, provenance=provenance,
                training_domain=training_domain(images),
            )
            model.save(model_path)
            write_pgm(image_path, images[0])
            report = score_path(model_path, image_path)
        self.assertEqual(report["training_provenance"], provenance)
        self.assertEqual(len(report["model_sha256"]), 64)
        self.assertEqual(report["call"], "unscorable")
        self.assertEqual(report["prediction_status"], "unscorable")
        self.assertIn("explicit_synthetic_demo", report["rejection_reason"])
        self.assertEqual(len(report["input_image_sha256"]), 64)
        self.assertEqual(report["training_source_sha256"], "b" * 64)
        self.assertEqual(report["model_schema_version"], 2)
        self.assertEqual(report["feature_schema_version"], 1)
        self.assertEqual(report["training_data_status"], "synthetic")
        self.assertTrue(report["synthetic_training"])
        html = __import__("colonyqc.report", fromlist=["generate_html_report"]).generate_html_report(report)
        self.assertIn("Model schema version", html)
        self.assertIn("Feature schema version", html)
        self.assertIn("training_source_sha256", html)

        # The explicit mode demonstrates software behavior but never emits a culture call.
        with tempfile.TemporaryDirectory() as tmp:
            model_path, image_path = os.path.join(tmp, "model.json"), os.path.join(tmp, "field.pgm")
            model = SoftmaxQC().fit(
                x, labels, epochs=100, seed=3, provenance=provenance,
                training_domain=training_domain(images),
            )
            model.save(model_path)
            write_pgm(image_path, images[0])
            report = score_path(model_path, image_path, input_domain="synthetic-demo")
        self.assertEqual(report["call"], "synthetic_demo_only")
        self.assertIn(report["demonstration_class"], LABELS)
        self.assertEqual(report["flags"], [])
        self.assertIn("Do not use this result", report["next_human_step"])
        self.assertNotIn("quarantine", report["next_human_step"].lower())
        self.assertNotIn("mycoplasma", report["next_human_step"].lower())

    def test_synthetic_model_rejects_blank_saturated_blurred_shifted_and_wrong_scale_fields(self):
        images, labels = dataset(n_per_class=20, seed=11)
        x = np.vstack([featurize(im) for im in images])
        provenance = {
            "source": "synthetic_generator",
            "training_source_sha256": "c" * 64,
        }
        model = SoftmaxQC().fit(
            x, labels, epochs=100, seed=11, provenance=provenance,
            training_domain=training_domain(images),
        )
        cases = {
            "blank": np.full_like(images[0], 0.5),
            "saturated": np.ones_like(images[0]),
            "blurred": box_blur(box_blur(images[0])),
            "intensity_shift": np.clip(images[0] + 0.15, 0, 1),
            "contrast_scale": np.clip((images[0] - 0.5) * 0.5 + 0.5, 0, 1),
            "wrong_scale": images[0][::2, ::2],
        }
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.json")
            model.save(model_path)
            statuses = {}
            reasons = {}
            for name, image in cases.items():
                image_path = os.path.join(tmp, f"{name}.pgm")
                write_pgm(image_path, image)
                report = score_path(model_path, image_path, input_domain="synthetic-demo")
                statuses[name] = report["prediction_status"]
                reasons[name] = report.get("rejection_reason", "")
        self.assertEqual(statuses["blank"], "unscorable")
        self.assertEqual(statuses["saturated"], "unscorable")
        for name in ["blurred", "intensity_shift", "contrast_scale", "wrong_scale"]:
            self.assertEqual(statuses[name], "out_of_domain", (name, reasons[name]))
        self.assertTrue(all("demonstration_class" not in reasons[name] for name in reasons))

    def test_predict_png_and_tiff_with_html(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL required for PNG/TIFF tests")
        from colonyqc.cli import main, score_path
        images, labels = dataset(n_per_class=10, seed=42)
        x = np.vstack([featurize(im) for im in images])
        model = SoftmaxQC().fit(x, labels, epochs=50, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            m_path = os.path.join(tmp, "model.json")
            model.save(m_path)

            # Test PNG
            png_path = os.path.join(tmp, "colony.png")
            html_png = os.path.join(tmp, "report_png.html")
            Image.fromarray((images[0] * 255).astype(np.uint8)).save(png_path)
            res_png = score_path(m_path, png_path, html_path=html_png)
            self.assertIn("call", res_png)
            self.assertTrue(os.path.exists(html_png))
            self.assertGreater(os.path.getsize(html_png), 500)

            # Test TIFF
            tif_path = os.path.join(tmp, "colony.tif")
            html_tif = os.path.join(tmp, "report_tif.html")
            Image.fromarray((images[0] * 255).astype(np.uint8)).save(tif_path)
            res_tif = score_path(m_path, tif_path, html_path=html_tif)
            self.assertIn("call", res_tif)
            self.assertTrue(os.path.exists(html_tif))
            self.assertGreater(os.path.getsize(html_tif), 500)

            # Test CLI predict with --html
            cli_html = os.path.join(tmp, "cli_report.html")
            code = main(["predict", "--model", m_path, "--image", png_path, "--html", cli_html])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(cli_html))


if __name__ == "__main__":
    unittest.main()

