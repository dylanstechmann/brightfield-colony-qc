"""Synthetic brightfield fields with known morphology labels.

The generator exists so the QC stack can be tested without shipping
licensed cell images. It is not a claim about real iPSC appearance.
"""

from __future__ import annotations

import numpy as np

LABELS = (
    "undifferentiated",
    "differentiating",
    "debris",
    "contamination_suspect",
)


def render(rng: np.random.Generator, label: str, size: int = 96) -> np.ndarray:
    if label not in LABELS:
        raise ValueError(f"unknown label {label}")
    img = np.full((size, size), 0.64, dtype=np.float64)
    img += rng.normal(0.0, 0.012, img.shape)
    yy, xx = np.mgrid[0:size, 0:size]
    if label == "undifferentiated":
        _colonies(img, xx, yy, rng, n=int(rng.integers(1, 3)), radius=(11, 16),
                  interior=0.22, noise=0.018, irregular=0.0, halo=0.95)
    elif label == "differentiating":
        _colonies(img, xx, yy, rng, n=1, radius=(20, 30),
                  interior=0.40, noise=0.09, irregular=0.28, halo=0.72)
    elif label == "debris":
        _specks(img, xx, yy, rng, n=int(rng.integers(10, 22)), radius=(1.2, 2.6))
    else:
        _filaments(img, rng, n=int(rng.integers(5, 9)))
        if rng.random() < 0.5:
            _specks(img, xx, yy, rng, n=int(rng.integers(2, 6)), radius=(1.0, 2.0))
    return np.clip(img, 0.0, 1.0)


def dataset(n_per_class: int = 80, size: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for label in LABELS:
        for _ in range(n_per_class):
            images.append(render(rng, label, size=size))
            labels.append(label)
    return images, labels


def _colonies(img, xx, yy, rng, n, radius, interior, noise, irregular, halo):
    size = img.shape[0]
    for _ in range(n):
        r = float(rng.uniform(*radius))
        margin = int(r + 4)
        cx = float(rng.uniform(margin, size - margin))
        cy = float(rng.uniform(margin, size - margin))
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        theta = np.arctan2(yy - cy, xx - cx)
        edge = r * (1.0 + irregular * np.sin(3.0 * theta + rng.random())
                    + 0.45 * irregular * np.sin(7.0 * theta))
        mask = d <= np.maximum(edge, 1.0)
        img[mask] = interior + rng.normal(0.0, noise, int(mask.sum()))
        ring = (d > edge) & (d < edge + 2.4)
        img[ring] = np.maximum(img[ring], halo)


def _specks(img, xx, yy, rng, n, radius):
    size = img.shape[0]
    for _ in range(n):
        r = float(rng.uniform(*radius))
        cx = float(rng.uniform(2, size - 3))
        cy = float(rng.uniform(2, size - 3))
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = d <= r
        img[mask] = 0.12 + rng.normal(0.0, 0.02, int(mask.sum()))


def _filaments(img, rng, n: int):
    size = img.shape[0]
    for _ in range(n):
        x0, y0 = rng.uniform(0, size, size=2)
        x1, y1 = rng.uniform(0, size, size=2)
        amp = float(rng.uniform(2.0, 8.0))
        phase = float(rng.uniform(0, 6.28))
        samples = np.linspace(0.0, 1.0, 120)
        xs = x0 + (x1 - x0) * samples
        ys = y0 + (y1 - y0) * samples + amp * np.sin(2 * np.pi * samples * rng.uniform(0.6, 1.6) + phase)
        # perpendicular wobble so the trace is not a straight smear
        ortho = np.array([-(y1 - y0), x1 - x0])
        norm = np.linalg.norm(ortho) + 1e-6
        ortho = ortho / norm
        xs = xs + ortho[0] * amp * np.sin(4 * np.pi * samples + phase)
        ys = ys + ortho[1] * amp * np.sin(4 * np.pi * samples + phase)
        for x, y in zip(xs, ys):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < size and 0 <= ix < size:
                img[iy, ix] = 0.08
                if iy + 1 < size:
                    img[iy + 1, ix] = min(img[iy + 1, ix], 0.16)


def box_blur(img: np.ndarray) -> np.ndarray:
    """3x3 box blur. Used as a domain-shift stress test."""
    p = np.pad(img, 1, mode="edge")
    acc = np.zeros_like(img)
    for dy in range(3):
        for dx in range(3):
            acc += p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return acc / 9.0


def write_pgm(path: str, img: np.ndarray) -> None:
    data = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    h, w = data.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


def read_pgm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError("only binary PGM (P5) is supported")
        tokens = []
        while len(tokens) < 3:
            line = f.readline()
            if not line:
                raise ValueError("truncated PGM header")
            if line.startswith(b"#"):
                continue
            tokens.extend(line.split())
        w, h, maxv = int(tokens[0]), int(tokens[1]), int(tokens[2])
        if maxv != 255:
            raise ValueError("expected maxval 255")
        raw = f.read(w * h)
        if len(raw) != w * h:
            raise ValueError("truncated PGM raster")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w).astype(np.float64) / 255.0
