import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from colonyqc.features import FEATURE_NAMES, featurize
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

    def test_provenance_hashes_match_the_decoded_input_snapshots(self):
        image = self.root / "field.pgm"
        manifest_bytes = self.manifest.read_bytes()
        image_bytes = image.read_bytes()
        expected_features = featurize(read_image(image))
        read_bytes = Path.read_bytes
        reads = {self.manifest: 0, image: 0}

        def change_after_read(path):
            data = read_bytes(path)
            if path == self.manifest:
                reads[path] += 1
                self.manifest.write_text("invalid manifest", encoding="utf-8")
            elif path == image:
                reads[path] += 1
                write_pgm(str(image), np.ones((96, 96)))
            return data

        output = self.root / "features.csv"
        with patch.object(Path, "read_bytes", change_after_read):
            export_features(self.manifest, output)
        self.assertEqual(reads, {self.manifest: 1, image: 1})
        provenance = json.loads(output.with_suffix(".csv.provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["manifest_sha256"], hashlib.sha256(manifest_bytes).hexdigest())
        self.assertEqual(provenance["images"][0]["image_sha256"], hashlib.sha256(image_bytes).hexdigest())
        with output.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        np.testing.assert_allclose([float(row[f"f_{name}"]) for name in FEATURE_NAMES], expected_features)

    def test_export_columns_form_the_contract_regen_benchmark_kit_reads(self):
        # regenbench groups on donor_id/batch_id/group_id and takes every f_* column as a feature.
        self.manifest.write_text("sample_id,image_path,label,group_id,donor_id,batch_id\n"
                                 "s1,field.pgm,undifferentiated,plate1,d1,b1\n")
        output = self.root / "contract.csv"
        export_features(self.manifest, output)
        with output.open() as handle:
            reader = csv.DictReader(handle)
            header, rows = reader.fieldnames, list(reader)
        self.assertEqual(header[:6], ["sample_id", "label", "group_id", "donor_id", "batch_id", "image_sha256"])
        self.assertEqual([c for c in header if c.startswith("f_")], [f"f_{n}" for n in FEATURE_NAMES])
        self.assertEqual(header, header[:6] + [f"f_{n}" for n in FEATURE_NAMES])
        self.assertEqual((rows[0]["donor_id"], rows[0]["batch_id"]), ("d1", "b1"))
        for name in FEATURE_NAMES:
            self.assertTrue(np.isfinite(float(rows[0][f"f_{name}"])))

    def test_duplicate_image_under_new_id_is_rejected(self):
        with self.manifest.open("a") as handle:
            handle.write("s2,field.pgm,undifferentiated,plate2,d2\n")
        output = self.root / "features.csv"
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            export_features(self.manifest, output)
        self.assertFalse(output.exists())

    def test_acquisition_metadata_is_preserved_and_measured_values_are_checked(self):
        header = ["sample_id", "image_path", "label", "group_id", "microscope_id",
                  "objective_magnification", "pixel_size_um", "exposure_ms", "illumination_mode"]
        values = ["s1", "field.pgm", "undifferentiated", "plate1", "scope-1", "20", "0.65", "40", "phase"]
        self.manifest.write_text(",".join(header) + "\n" + ",".join(values) + "\n", encoding="utf-8")
        output = self.root / "metadata-features.csv"
        export_features(self.manifest, output)
        with output.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["microscope_id"], "scope-1")
        self.assertEqual(row["objective_magnification"], "20")
        self.assertEqual(row["pixel_size_um"], "0.65")
        self.assertEqual(row["illumination_mode"], "phase")

        for field, bad_value in (("objective_magnification", "0"), ("pixel_size_um", "nan"),
                                 ("exposure_ms", "not-a-number")):
            with self.subTest(field=field, bad_value=bad_value):
                invalid = list(values)
                invalid[header.index(field)] = bad_value
                self.manifest.write_text(",".join(header) + "\n" + ",".join(invalid) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, field):
                    export_features(self.manifest, self.root / f"invalid-{field}.csv")

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
        model = SoftmaxQC().fit(np.tile([[0.0], [1.0]], (1, len(FEATURE_NAMES))), ["debris", "undifferentiated"], epochs=2)
        model.save(str(self.root / "new" / "model.json"))
        self.assertTrue((self.root / "new" / "model.json").exists())
