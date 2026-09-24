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


if __name__ == "__main__":
    unittest.main()
