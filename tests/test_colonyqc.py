import os
import tempfile
import unittest

import numpy as np

from colonyqc.cli import demo
from colonyqc.features import FEATURE_NAMES, featurize
from colonyqc.model import SoftmaxQC, accuracy
from colonyqc.report import DISCLAIMER, build_report
from colonyqc.synthetic import LABELS, dataset, read_pgm, render, write_pgm


class FeatureTests(unittest.TestCase):
    def test_vector_shape_and_finite(self):
        img = render(np.random.default_rng(0), "undifferentiated")
        feat = featurize(img)
        self.assertEqual(feat.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(feat).all())

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

