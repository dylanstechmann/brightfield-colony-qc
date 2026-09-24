"""Import annotated fields without losing sample grouping or source hashes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from colonyqc.features import FEATURE_NAMES, featurize
from colonyqc.synthetic import LABELS, read_pgm


def read_image(path):
    path = Path(path)
    if path.suffix.lower() == ".pgm":
        return read_pgm(str(path))
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("PNG/TIFF support requires pip install '.[images]'") from exc
    with Image.open(path) as image:
        if image.mode != "L" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("use a single-frame 8-bit grayscale image; document any upstream conversion")
        return np.asarray(image, dtype=np.float64) / 255.0


def export_features(manifest, output):
    manifest, output = Path(manifest), Path(output)
    provenance = output.with_suffix(output.suffix + ".provenance.json")
    if output.exists() or provenance.exists():
        raise ValueError("output or provenance already exists; use a new output path")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        required = {"sample_id", "image_path", "label", "group_id"}
        if len(header) != len(set(header)) or not required.issubset(header):
            raise ValueError("manifest needs unique sample_id,image_path,label,group_id columns")
        rows = list(reader)
    if not rows:
        raise ValueError("empty manifest")
    optional = [c for c in ["donor_id", "batch_id", "plate_id"] if c in header]
    records, images, seen_ids, seen_hashes = [], [], set(), set()
    for line, row in enumerate(rows, 2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"row {line}: ragged manifest")
        row = {k: v.strip() for k, v in row.items()}
        if any(not row[c] for c in required):
            raise ValueError(f"row {line}: blank required field")
        if row["sample_id"] in seen_ids:
            raise ValueError(f"row {line}: duplicate sample_id")
        if row["label"] not in LABELS:
            raise ValueError(f"row {line}: unsupported label {row['label']}")
        path = manifest.parent / row["image_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            raise ValueError(f"row {line}: duplicate image bytes; reconcile duplicate records first")
        seen_ids.add(row["sample_id"])
        seen_hashes.add(digest)
        features = featurize(read_image(path))
        records.append({"sample_id": row["sample_id"], "label": row["label"],
                        "group_id": row["group_id"], **{c: row[c] for c in optional},
                        "image_sha256": digest,
                        **{f"f_{key}": float(value) for key, value in zip(FEATURE_NAMES, features)}})
        images.append({"sample_id": row["sample_id"], "image_sha256": digest})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {"schema_version": 1, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "features_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "images": images, "feature_names": list(FEATURE_NAMES),
               "note": "User annotations, not model-confirmed cell states. Group related fields before evaluation."}
    with provenance.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return {"n_images": len(records), "features": str(output), "provenance": str(provenance)}
