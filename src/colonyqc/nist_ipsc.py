"""Reproducible phase-image features from the NIST mds2-2960 training archives.

The target is fluorescence-derived nuclear mask area, not a cell-health label.
Raw archives stay in the user's cache. No NIST model or code is imported.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from urllib.request import urlopen
import zipfile

import numpy as np

from colonyqc import __version__
from colonyqc.features import FEATURE_NAMES, featurize


SOURCE_DOI = "https://doi.org/10.18434/mds2-2960"
SOURCE_LICENSE = "https://www.nist.gov/open/license"
IMAGE_SIZE = (15104, 15104)
ASSETS = (
    {"name": "training_low", "density": "low", "size": 151280344,
     "sha256": "c01c0c8ef599ede091377279ec0329997c5c102a9250c89e4cb11002214460fb"},
    {"name": "training_medium", "density": "medium", "size": 160367521,
     "sha256": "970488bd1360f0aa1877e0b3542ca63385d10e844500b6af6b997b04da6c9396"},
    {"name": "training_high", "density": "high", "size": 189214249,
     "sha256": "10dcf99f2bd573c0d6b58970548e6c60ca57d93877d397e09860abaeb368c0e2"},
)
MEMBERS = ("img_phase.tif", "img_segmented.tif")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_url(asset):
    return f"https://data.nist.gov/od/ds/mds2-2960/{asset['name']}.zip"


def verify_archive(path, asset):
    if Path(path).stat().st_size != asset["size"] or file_sha256(path) != asset["sha256"]:
        raise ValueError(f"archive size or SHA-256 mismatch: {Path(path).name}")


def cached_archive(cache, asset, *, download=False):
    cache = Path(cache)
    path = cache / f"{asset['name']}.zip"
    if path.exists():
        verify_archive(path, asset)
        return path
    if not download:
        raise FileNotFoundError(f"missing {path.name}; use --download to fetch the three archives (~501 MB)")
    cache.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            total = 0
            with urlopen(asset_url(asset), timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > asset["size"]:
                        raise ValueError("download exceeds the pinned archive size")
                    handle.write(chunk)
        verify_archive(temporary, asset)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def extract_pair(archive, asset, destination):
    """Extract only two fixed members to seekable files; enforce size and CRC."""
    details = {}
    with zipfile.ZipFile(archive) as bundle:
        for name in MEMBERS:
            full_name = f"{asset['name']}/{name}"
            if bundle.namelist().count(full_name) != 1:
                raise ValueError(f"expected exactly one archive member: {full_name}")
            info = bundle.getinfo(full_name)
            if not 0 < info.file_size <= 300_000_000:
                raise ValueError(f"unexpected TIFF size: {full_name}")
            path = Path(destination) / name
            digest = hashlib.sha256()
            with bundle.open(info) as source, path.open("wb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
                    digest.update(chunk)
            details[name] = {"archive_path": full_name, "size": info.file_size,
                             "sha256": digest.hexdigest()}
    return details


@contextmanager
def nist_pixel_limit():
    from PIL import Image
    # This ceiling is scoped to the pinned 15104-square source images.
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 230_000_000
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def select_tiles(size, tile_size, count, rng):
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("tile size must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("tiles per well must be a positive integer")
    width, height = size
    nx, ny = width // tile_size, height // tile_size
    if count > nx * ny:
        raise ValueError("requested more tiles than the full nonoverlapping grid contains")
    # Coordinates depend only on geometry and the seed, never on target or image content.
    indices = np.sort(rng.choice(nx * ny, size=count, replace=False))
    return [(int(i % nx) * tile_size, int(i // nx) * tile_size) for i in indices]


def tile_rows(phase, mask, asset, coordinates, tile_size):
    for x, y in coordinates:
        box = (x, y, x + tile_size, y + tile_size)
        pixels = np.asarray(phase.crop(box), dtype=np.uint8)
        nuclei = np.asarray(mask.crop(box), dtype=np.uint8)
        row = {"sample_id": f"{asset['name']}_x{x:05d}_y{y:05d}",
               "source_well": asset["name"], "density_condition": asset["density"],
               "x_px": x, "y_px": y, "tile_size_px": tile_size,
               "reference_nuclear_fraction": float(np.count_nonzero(nuclei) / nuclei.size),
               "image_sha256": hashlib.sha256(pixels.tobytes(order="C")).hexdigest()}
        row.update(zip((f"f_{name}" for name in FEATURE_NAMES),
                       map(float, featurize(pixels.astype(np.float64) / 255.0))))
        yield row


def export_nist(cache, output, *, download=False, tile_size=512, tiles_per_well=64, seed=0,
                progress=None):
    from PIL import Image, __version__ as pillow_version

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # Validate geometry before downloading anything.
    select_tiles(IMAGE_SIZE, tile_size, tiles_per_well, np.random.default_rng(seed))
    rows, sources = [], []
    for asset in ASSETS:
        if progress:
            progress(f"Verifying and extracting {asset['name']}...")
        archive = cached_archive(cache, asset, download=download)
        coordinates = select_tiles(IMAGE_SIZE, tile_size, tiles_per_well, rng)
        with tempfile.TemporaryDirectory(prefix="colonyqc-nist-") as scratch:
            members = extract_pair(archive, asset, scratch)
            with nist_pixel_limit(), Image.open(Path(scratch) / MEMBERS[0]) as phase, \
                    Image.open(Path(scratch) / MEMBERS[1]) as mask:
                for label, img in [("phase", phase), ("nuclear mask", mask)]:
                    if img.size != IMAGE_SIZE or img.mode != "L" or getattr(img, "n_frames", 1) != 1:
                        raise ValueError(f"unexpected {label} image geometry, mode or frame count")
                phase.load()
                mask.load()
                if mask.getextrema()[1] > 1:
                    raise ValueError("expected a binary 0/1 nuclear mask")
                rows.extend(tile_rows(phase, mask, asset, coordinates, tile_size))
        sources.append({**asset, "url": asset_url(asset), "members": members,
                        "selected_xy": coordinates})
    if len({row["image_sha256"] for row in rows}) != len(rows):
        raise ValueError("duplicate phase tiles detected; inspect the source before evaluation")
    provenance = {
        "schema_version": 1, "source_doi": SOURCE_DOI, "source_license": SOURCE_LICENSE,
        "paper_doi": "https://doi.org/10.1371/journal.pone.0298446",
        "catalog_metadata_version": "1.1.0", "sources": sources,
        "configuration": {"tile_size": tile_size, "tiles_per_well": tiles_per_well, "seed": seed,
            "sampling": "PCG64; sorted choice without replacement from full nonoverlapping grid; low, medium, high order",
            "normalization": "uint8 phase pixels / 255; no learned image normalization",
            "features": list(FEATURE_NAMES),
            "target": "nonzero pixels in img_segmented.tif / tile pixel count",
            "target_column": "reference_nuclear_fraction", "group_column": "source_well",
            "image_sha256": "SHA-256 of raw uint8 phase-tile pixels in C row-major order",
            "exclusions": "partial edge tiles only; no target-based or image-content filtering"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "Pillow": pillow_version, "colonyqc": __version__},
        "n_rows": len(rows),
        "limitations": ["Three source wells from one study; density is confounded with well.",
            "Reference masks are automatically derived from fluorescence, not manual ground truth.",
            "Nuclear mask area is not cell count, confluence, viability, pluripotency or senescence.",
            "img_fg.tif is a different foreground mask and is not used.",
            "No pretrained WSDOM model is used; these archives were training data in the original study."],
        "attribution": "Derived from NIST mds2-2960 by Asmar et al. NIST source terms apply to source data and derivatives; no NIST endorsement implied.",
        "modification_notice": "Dylan Stechmann project: sampled tiles and computed phase-image features and nuclear area fractions; source TIFFs unchanged.",
    }
    output.mkdir(parents=True, exist_ok=False)
    table = output / "features.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance["features_sha256"] = file_sha256(table)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export phase-image features and nuclear mask area from three NIST iPSC wells")
    parser.add_argument("--cache", required=True, help="directory containing training_{low,medium,high}.zip")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--download", action="store_true", help="download missing pinned archives (~501 MB)")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tiles-per-well", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = export_nist(args.cache, args.out, download=args.download, tile_size=args.tile_size,
                             tiles_per_well=args.tiles_per_well, seed=args.seed,
                             progress=lambda message: print(message, file=sys.stderr, flush=True))
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps({"n_rows": result["n_rows"], "features_sha256": result["features_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
