"""Command line for the synthetic bake-off and PGM scoring."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from colonyqc.features import featurize
from colonyqc.model import SoftmaxQC, accuracy, majority_accuracy
from colonyqc.report import build_report
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
    return {
        "n_train": int(split),
        "n_test": int(len(test)),
        "majority_baseline": round(base, 4),
        "holdout_accuracy": round(clean, 4),
        "blurred_holdout_accuracy": round(blur_acc, 4),
        "note": (
            "Clean accuracy is on images drawn from the same generator as training. "
            "It is a software ceiling, not an iPSC result. The blurred number is the "
            "stress test. Retrain on real annotated fields before using any call."
        ),
    }


def score_path(model_path: str, image_path: str) -> dict:
    model = SoftmaxQC.load(model_path)
    img = read_pgm(image_path)
    feat = featurize(img)
    proba = model.predict_proba(feat.reshape(1, -1))[0]
    mapping = {name: float(p) for name, p in zip(model.classes, proba)}
    label = model.classes[int(np.argmax(proba))]
    return build_report(label, mapping, feat.tolist())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Brightfield colony morphology QC")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="train on synthetic fields and print the bake-off")
    d.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("predict", help="score one binary PGM image")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    t = sub.add_parser("train", help="fit on the synthetic generator and save weights")
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        json.dump(demo(args.seed), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.cmd == "train":
        images, labels = dataset(n_per_class=80, seed=args.seed)
        x = np.vstack([featurize(im) for im in images])
        SoftmaxQC().fit(x, labels, seed=args.seed).save(args.out)
        return 0
    json.dump(score_path(args.model, args.image), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
