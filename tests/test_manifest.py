import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from colonyqc.features import featurize
from colonyqc.manifest import export_features, read_image
from colonyqc.model import SoftmaxQC
from colonyqc.synthetic import render, write_pgm


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_pgm(str(self.root / "field.pgm"), render(np.random.default_rng(0), "undifferentiated"))
        self.manifest = self.root / "manifest.csv"
        self.manifest.write_text("sample_id,image_path,label,group_id,donor_id\ns1,field.pgm,undifferentiated,plate1,d1\n")

    def test_export_preserves_groups_and_source_hashes(self):
        output = self.root / "features.csv"
        export_features(self.manifest, output)
        with output.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["donor_id"], "d1")
        self.assertEqual(rows[0]["group_id"], "plate1")
        self.assertEqual(len(rows[0]["image_sha256"]), 64)
        provenance = json.loads(output.with_suffix(".csv.provenance.json").read_text())
        self.assertEqual(provenance["images"][0]["image_sha256"], rows[0]["image_sha256"])
        with self.assertRaises(ValueError):
            export_features(self.manifest, output)

    def test_duplicate_image_under_new_id_is_rejected(self):
        with self.manifest.open("a") as handle:
            handle.write("s2,field.pgm,undifferentiated,plate2,d2\n")
        output = self.root / "features.csv"
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            export_features(self.manifest, output)
        self.assertFalse(output.exists())

    def test_invalid_pixels_do_not_become_predictions(self):
        for image in [np.empty((0, 0)), np.array([[np.nan]]), np.array([[255]])]:
            with self.assertRaises(ValueError):
                featurize(image)

    def test_png_retains_fixed_scale(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("optional images extra is not installed")
        Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(self.root / "field.png")
        np.testing.assert_allclose(read_image(self.root / "field.png"), [[0, 128 / 255, 1]])

    def test_model_save_creates_documented_artifact_directory(self):
        model = SoftmaxQC().fit(np.array([[0.0], [1.0]]), ["debris", "undifferentiated"], epochs=2)
        model.save(str(self.root / "new" / "model.json"))
        self.assertTrue((self.root / "new" / "model.json").exists())
