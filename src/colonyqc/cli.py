"""Command line for the synthetic bake-off and PGM scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from colonyqc.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, featurize
from colonyqc.manifest import decode_image_bytes, export_features
from colonyqc.model import MODEL_SCHEMA_VERSION, SoftmaxQC, accuracy, majority_accuracy
from colonyqc.report import build_report, generate_html_report
from colonyqc.selective import (
    DEFAULT_ABSTAIN_THRESHOLD,
    confidence,
    risk_coverage_curve,
    selective_calls,
    selective_metrics,
)
from colonyqc.synthetic import box_blur, dataset, read_pgm


def demo(seed: int) -> dict:
    images, labels = dataset(n_per_class=60, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    split = int(0.75 * len(idx))
    train, test = idx[:split], idx[split:]
    x = np.vstack([featurize(images[i]) for i in range(len(images))])
    y = [labels[i] for i in range(len(labels))]
    model = SoftmaxQC().fit(x[train], [y[i] for i in train], seed=seed)
    pred = model.predict(x[test])
    y_test = [y[i] for i in test]
    clean = accuracy(y_test, pred)
    blurred = [box_blur(box_blur(images[i])) for i in test]
    blur_x = np.vstack([featurize(im) for im in blurred])
    blur_acc = accuracy(y_test, model.predict(blur_x))
    base = majority_accuracy(y_test, [y[i] for i in train])
    clean_proba = model.predict_proba(x[test])
    blur_proba = model.predict_proba(blur_x)
    blur_pred = model.predict(blur_x)
    return {
        "n_train": int(split),
        "n_test": int(len(test)),
        "majority_baseline": round(base, 4),
        "holdout_accuracy": round(clean, 4),
        "blurred_holdout_accuracy": round(blur_acc, 4),
        "selective_prediction": {
            "abstain_threshold": DEFAULT_ABSTAIN_THRESHOLD,
            "clean": selective_metrics(y_test, pred, clean_proba),
            "blurred": selective_metrics(y_test, blur_pred, blur_proba),
            "clean_risk_coverage": risk_coverage_curve(y_test, pred, clean_proba),
            "blurred_risk_coverage": risk_coverage_curve(y_test, blur_pred, blur_proba),
        },
        "note": (
            "Clean accuracy is on images drawn from the same generator as training. "
            "It is a software ceiling, not an iPSC result. The blurred number is the "
            "stress test. Retrain on real annotated fields before using any call. "
            "Selective accuracy is measured only on answered fields, so read it with its "
            "coverage; abstaining is a request for human review, not a culture verdict."
        ),
    }


def training_domain(images: list[np.ndarray]) -> dict:
    """Summarize fixed-scale image statistics seen during fit."""
    if not images:
        raise ValueError("training domain needs at least one image")
    shapes = {tuple(np.asarray(image).shape) for image in images}
    if len(shapes) != 1:
        raise ValueError("training images must share one image shape")
    means = np.asarray([np.asarray(image, dtype=np.float64).mean() for image in images])
    stds = np.asarray([np.asarray(image, dtype=np.float64).std() for image in images])
    adjacent_differences = np.asarray([
        (np.abs(np.diff(image, axis=0)).mean() + np.abs(np.diff(image, axis=1)).mean()) / 2.0
        for image in images
    ])
    return {
        "status": "recorded_training_summary",
        "scope": "image_only; microscope, objective, illumination, and exposure metadata are not recorded",
        "image_shape": list(next(iter(shapes))),
        "pixel_scale": [0.0, 1.0],
        "image_mean_min_max": [float(means.min()), float(means.max())],
        "image_std_min_max": [float(stds.min()), float(stds.max())],
        "comparison_tolerance": 0.01,
        "adjacent_pixel_difference_min_max": [
            float(adjacent_differences.min()), float(adjacent_differences.max())
        ],
        "detail_tolerance": 0.002,
        "summary_method": "min_max_across_training_images_with_fixed_tolerance",
        "image_count": len(images),
    }


def _domain_rejection_reasons(img: np.ndarray, domain: dict) -> list[str]:
    if domain.get("status") != "recorded_training_summary":
        return ["training_domain_not_recorded"]
    try:
        if list(img.shape) != domain["image_shape"]:
            return ["image_shape_outside_training_domain"]
        tol = float(domain["comparison_tolerance"])
        mean_min, mean_max = map(float, domain["image_mean_min_max"])
        std_min, std_max = map(float, domain["image_std_min_max"])
        detail_min, detail_max = map(float, domain["adjacent_pixel_difference_min_max"])
        detail_tol = float(domain["detail_tolerance"])
    except (KeyError, TypeError, ValueError):
        return ["training_domain_metadata_invalid"]
    values = [tol, mean_min, mean_max, std_min, std_max, detail_min, detail_max, detail_tol]
    if not np.isfinite(values).all() or tol < 0 or detail_tol < 0 or mean_min > mean_max or std_min > std_max or detail_min > detail_max:
        return ["training_domain_metadata_invalid"]
    detail = float((np.abs(np.diff(img, axis=0)).mean() + np.abs(np.diff(img, axis=1)).mean()) / 2.0)
    reasons = []
    if not mean_min - tol <= float(img.mean()) <= mean_max + tol:
        reasons.append("image_intensity_outside_training_domain")
    if not std_min - tol <= float(img.std()) <= std_max + tol:
        reasons.append("image_contrast_or_blur_outside_training_domain")
    if not detail_min - detail_tol <= detail <= detail_max + detail_tol:
        reasons.append("image_detail_outside_training_domain")
    return reasons


def score_path(
    model_path: str,
    image_path: str,
    html_path: str | None = None,
    input_domain: str = "unverified",
    abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
) -> dict:
    # Hash and decode exactly the byte snapshots used for the report.
    model_bytes = Path(model_path).read_bytes()
    image_bytes = Path(image_path).read_bytes()
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    input_sha = hashlib.sha256(image_bytes).hexdigest()
    model = SoftmaxQC()
    model_error = None
    try:
        model = SoftmaxQC.from_dict(json.loads(model_bytes))
    except (ValueError, TypeError, KeyError) as exc:
        model_error = f"model_artifact_invalid:{exc}"
    common = {
        "model_sha256": model_sha,
        "input_image_sha256": input_sha,
        "model_schema_version": model.model_schema_version,
        "feature_schema_version": model.feature_schema_version,
        "training_data_status": model.training_data_status,
        "training_domain": model.training_domain,
        "training_source_sha256": model.training_provenance.get("training_source_sha256"),
    }
    decoded_img = None

    def rejected(status: str, reason: str, img=None, features=None):
        report = build_report(
            status,
            {},
            [] if features is None else features.tolist(),
            training_provenance=model.training_provenance,
            synthetic_training=model.training_data_status == "synthetic",
        )
        report.update(common)
        report.update({"prediction_status": status, "rejection_reason": reason})
        if html_path:
            html_out = Path(html_path)
            html_out.parent.mkdir(parents=True, exist_ok=True)
            html_out.write_text(
                generate_html_report(report, image_array=img if img is not None else decoded_img, image_name=str(image_path)),
                encoding="utf-8",
            )
        return report

    try:
        decoded_img = decode_image_bytes(image_bytes, Path(image_path).suffix)
    except (ValueError, OSError) as exc:
        return rejected("unscorable", f"image_decode_failed:{exc}")

    if model_error is not None:
        return rejected("unscorable", model_error)
    if model.model_schema_version != MODEL_SCHEMA_VERSION:
        return rejected("unscorable", "unsupported_model_schema_version")
    if model.feature_schema_version != FEATURE_SCHEMA_VERSION:
        return rejected("unscorable", "unsupported_feature_schema_version")
    if model.training_data_status != "synthetic":
        return rejected("unscorable", "training_source_not_verified_for_scoring")
    if input_domain != "synthetic-demo":
        return rejected("unscorable", "synthetic_model_requires_explicit_synthetic_demo_domain")
    img = decoded_img
    try:
        feat = featurize(img)
    except ValueError as exc:
        return rejected("unscorable", f"image_features_unavailable:{exc}", img=img)
    reasons = _domain_rejection_reasons(img, model.training_domain)
    if reasons:
        return rejected("out_of_domain", ";".join(reasons), img=img, features=feat)

    proba = model.predict_proba(feat.reshape(1, -1))[0]
    if not np.isfinite(proba).all():
        return rejected("unscorable", "model_outputs_nonfinite", img=img, features=feat)
    mapping = {name: float(p) for name, p in zip(model.classes, proba)}
    demo_label = model.classes[int(np.argmax(proba))]
    top_confidence = float(confidence(proba.reshape(1, -1))[0])
    abstained = top_confidence < abstain_threshold
    report = build_report(
        "synthetic_demo_only",
        mapping,
        feat.tolist(),
        training_provenance=model.training_provenance,
        synthetic_training=True,
        demonstration_class=None if abstained else demo_label,
    )
    report.update(common)
    report["prediction_status"] = "synthetic_demo_only"
    report["selective_prediction"] = {
        "abstain_threshold": float(abstain_threshold),
        "top_label_confidence": top_confidence,
        "abstained": abstained,
        "withheld_demonstration_class": demo_label if abstained else None,
        "note": (
            "Top-label confidence is below the abstention threshold, so no demonstration class is "
            "presented. An abstention asks for human review; it is not a statement about a culture."
            if abstained else
            "Top-label confidence is at or above the abstention threshold. On blurred fields this "
            "confidence score does not rank correctness, so it is not a reliability estimate."
        ),
    }
    if abstained:
        report["flags"] = [*report.get("flags", []), "abstained_low_confidence"]
        report["next_human_step"] = (
            "Confidence is below the abstention threshold: review this field manually. "
            "No class is reported, and this remains a synthetic demonstration.")
    if html_path:
        html_out = Path(html_path)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_content = generate_html_report(report, image_array=img, image_name=str(image_path))
        html_out.write_text(html_content, encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Brightfield colony morphology QC")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="train on synthetic fields and print the bake-off")
    d.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("predict", help="score one brightfield image (PGM, PNG, TIFF)")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--html", default=None, help="save visual HTML triage report to path")
    p.add_argument(
        "--input-domain", choices=["unverified", "synthetic-demo"], default="unverified",
        help="synthetic-demo is for generator fixtures only; it does not permit culture decisions",
    )
    p.add_argument(
        "--abstain-threshold", type=float, default=DEFAULT_ABSTAIN_THRESHOLD,
        help=("withhold the demonstration class when top-label confidence falls below this value "
              f"(default {DEFAULT_ABSTAIN_THRESHOLD}); abstaining asks for human review"),
    )
    t = sub.add_parser("train", help="fit on the synthetic generator and save weights")
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("export-features", help="export an annotated image manifest for grouped evaluation")
    e.add_argument("manifest")
    e.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "export-features":
        try:
            result = export_features(args.manifest, args.out)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.cmd == "demo":
        json.dump(demo(args.seed), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.cmd == "train":
        images, labels = dataset(n_per_class=80, seed=args.seed)
        x = np.vstack([featurize(im) for im in images])
        training_digest = hashlib.sha256(
            x.astype("<f8", copy=False).tobytes()
            + json.dumps(labels, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        provenance = {
            "source": "synthetic_generator",
            "generator": "colonyqc.synthetic.dataset",
            "generator_seed": args.seed,
            "samples_per_class": 80,
            "sample_count": len(labels),
            "feature_names": list(FEATURE_NAMES),
            "training_matrix_sha256": training_digest,
            "training_source_sha256": hashlib.sha256(
                b"".join(np.asarray(image, dtype="<f8").tobytes() for image in images)
                + json.dumps(labels, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        SoftmaxQC().fit(
            x,
            labels,
            seed=args.seed,
            provenance=provenance,
            training_domain=training_domain(images),
        ).save(args.out)
        return 0
    try:
        if not 0 <= args.abstain_threshold <= 1:
            parser.error("--abstain-threshold must lie in [0, 1]")
        report = score_path(args.model, args.image, html_path=args.html,
                            input_domain=args.input_domain,
                            abstain_threshold=args.abstain_threshold)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
