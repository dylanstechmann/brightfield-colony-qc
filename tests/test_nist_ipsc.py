import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

from colonyqc import nist_ipsc as nist


@unittest.skipIf(Image is None, "optional images extra is not installed")
class NistImporterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.phase = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8) * 4)
        self.mask = Image.fromarray((np.indices((8, 8)).sum(axis=0) % 2).astype(np.uint8))
        self.asset = self.make_archive()

    def make_archive(self, mask=None):
        path = self.root / "training_low.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, img in [("img_phase.tif", self.phase),
                              ("img_segmented.tif", self.mask if mask is None else mask),
                              ("img_fg.tif", Image.new("L", (8, 8), 0))]:
                buffer = BytesIO()
                img.save(buffer, format="TIFF")
                archive.writestr(f"training_low/{name}", buffer.getvalue())
            archive.writestr("../unwanted.txt", "must not extract this")
        return {"name": "training_low", "density": "low", "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def export(self, name, **kwargs):
        with patch.object(nist, "ASSETS", (self.asset,)), patch.object(nist, "IMAGE_SIZE", (8, 8)):
            return nist.export_nist(self.root, self.root / name, tile_size=4, tiles_per_well=3, **kwargs)

    def test_export_uses_nuclear_mask_and_is_reproducible(self):
        report = self.export("first")
        self.assertEqual(report, self.export("second"))
        table = self.root / "first/features.csv"
        self.assertEqual(table.read_bytes(), (self.root / "second/features.csv").read_bytes())
        with table.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(float(row["reference_nuclear_fraction"]) == 0.5 for row in rows))
        self.assertEqual(len({row["image_sha256"] for row in rows}), 3)
        self.assertEqual(report["features_sha256"], nist.file_sha256(table))
        self.assertEqual(json.loads((table.parent / "provenance.json").read_text()),
                         json.loads(json.dumps(report)))
        self.assertFalse((self.root / "unwanted.txt").exists())
        with self.assertRaises(FileExistsError):
            self.export("first")

    def test_reference_mask_cannot_change_features(self):
        coordinates = [(0, 0), (4, 4)]
        a = list(nist.tile_rows(self.phase, self.mask, self.asset, coordinates, 4))
        b = list(nist.tile_rows(self.phase, Image.new("L", (8, 8), 1), self.asset, coordinates, 4))
        for left, right in zip(a, b):
            self.assertNotEqual(left.pop("reference_nuclear_fraction"), right.pop("reference_nuclear_fraction"))
            self.assertEqual(left, right)

    def test_corrupt_cache_is_rejected_without_download(self):
        path = self.root / "training_low.zip"
        raw = bytearray(path.read_bytes())
        raw[20] ^= 1
        path.write_bytes(raw)
        with patch.object(nist, "urlopen") as fetch:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.export("bad", download=True)
            fetch.assert_not_called()
        self.assertFalse((self.root / "bad").exists())

    def test_failed_download_cleans_partial_file(self):
        cache = self.root / "empty"
        with patch.object(nist, "urlopen", return_value=BytesIO(b"invalid")):
            with self.assertRaises(ValueError):
                nist.cached_archive(cache, self.asset, download=True)
        self.assertEqual(list(cache.iterdir()), [])
        with self.assertRaises(FileNotFoundError):
            nist.cached_archive(cache, self.asset)

    def test_geometry_and_binary_mask_are_checked(self):
        for bad in [Image.new("L", (8, 8), 2), Image.new("L", (4, 4), 1), Image.new("RGB", (8, 8))]:
            self.asset = self.make_archive(bad)
            with self.assertRaises(ValueError):
                self.export("bad")

    def test_nonoverlapping_content_independent_sampling(self):
        a = nist.select_tiles((17, 19), 4, 16, np.random.default_rng(2))
        self.assertEqual(len(set(a)), 16)
        self.assertTrue(all(x % 4 == y % 4 == 0 and x + 4 <= 17 and y + 4 <= 19 for x, y in a))
        self.assertEqual(a, nist.select_tiles((17, 19), 4, 16, np.random.default_rng(2)))
        for size, count in [(0, 2), (4, 17), (4, 0), (2.5, 2)]:
            with self.assertRaises(ValueError):
                nist.select_tiles((17, 19), size, count, np.random.default_rng(2))

    def test_pixel_limit_is_restored_on_error(self):
        before = Image.MAX_IMAGE_PIXELS
        with self.assertRaises(RuntimeError):
            with nist.nist_pixel_limit():
                self.assertEqual(Image.MAX_IMAGE_PIXELS, 230_000_000)
                raise RuntimeError()
        self.assertEqual(Image.MAX_IMAGE_PIXELS, before)


if __name__ == "__main__":
    unittest.main()
