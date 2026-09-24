"""Create a portable manifest fixture. All images and IDs are synthetic."""

import csv
from pathlib import Path

import numpy as np

from colonyqc.synthetic import LABELS, render, write_pgm


def main():
    root = Path("artifacts/manifest-demo")
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "image_path", "label", "group_id", "donor_id", "batch_id"])
        for group in range(8):
            for label in LABELS:
                sample = f"synthetic-{group}-{label}"
                write_pgm(str(root / f"{sample}.pgm"), render(rng, label))
                writer.writerow([sample, f"{sample}.pgm", label, f"run-{group}",
                                 f"donor-{group // 2}", f"batch-{group // 2}"])


if __name__ == "__main__":
    main()
