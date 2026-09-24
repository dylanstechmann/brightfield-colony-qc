"""Hand-built morphology features. No deep net, on purpose.

A PI can see why a field was flagged. That matters more than a leaderboard
number on synthetic blobs.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "fg_fraction",
    "n_components",
    "largest_area_frac",
    "largest_circularity",
    "halo",
    "interior_std",
    "thin_fraction",
    "small_component_frac",
    "bright_near_fg",
)


def featurize(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2 or not img.size:
        raise ValueError("expected a nonempty single grayscale image")
    if not np.isfinite(img).all() or np.any(img < 0) or np.any(img > 1):
        raise ValueError("image values must be finite and scaled to [0, 1]")
    fg = img < 0.48
    total = fg.size
    fg_fraction = float(fg.mean())
    labels, areas = _components(fg)
    n_components = len(areas)
    if n_components == 0:
        return np.array([
            fg_fraction, 0, 0, 0, 0, 0, 0, 0, 0,
        ], dtype=np.float64)
    largest = int(np.argmax(areas))
    largest_area = areas[largest]
    largest_mask = labels == (largest + 1)
    circ = _circularity(largest_mask)
    halo = _halo(img, largest_mask)
    interior = img[largest_mask]
    interior_std = float(interior.std()) if interior.size else 0.0
    thin = _thin_fraction(fg)
    small = sum(1 for a in areas if a < 12)
    bright = _bright_near(img, fg)
    return np.array([
        fg_fraction,
        float(n_components),
        largest_area / total,
        circ,
        halo,
        interior_std,
        thin,
        small / n_components,
        bright,
    ], dtype=np.float64)


def featurize_many(images) -> np.ndarray:
    return np.vstack([featurize(im) for im in images])


def _components(mask: np.ndarray):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    areas = []
    current = 0
    # row-major flood fill, 4-connected
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if not row[x] or labels[y, x]:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cy > 0 and mask[cy - 1, cx] and not labels[cy - 1, cx]:
                    labels[cy - 1, cx] = current
                    stack.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and not labels[cy + 1, cx]:
                    labels[cy + 1, cx] = current
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not labels[cy, cx - 1]:
                    labels[cy, cx - 1] = current
                    stack.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and not labels[cy, cx + 1]:
                    labels[cy, cx + 1] = current
                    stack.append((cy, cx + 1))
            areas.append(area)
    return labels, areas


def _circularity(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area < 1:
        return 0.0
    peri = _perimeter(mask)
    if peri < 1:
        return 0.0
    return float(min(1.0, 4.0 * np.pi * area / (peri * peri)))


def _perimeter(mask: np.ndarray) -> float:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:h + 1, 1:w + 1]
    neighbors = (
        padded[0:h, 1:w + 1]
        & padded[2:h + 2, 1:w + 1]
        & padded[1:h + 1, 0:w]
        & padded[1:h + 1, 2:w + 2]
    )
    return float(np.logical_and(center, np.logical_not(neighbors)).sum())


def _halo(img: np.ndarray, mask: np.ndarray) -> float:
    """Bright pixels just outside the object minus the object interior."""
    if mask.sum() < 4:
        return 0.0
    h, w = mask.shape
    padded = np.pad(mask, 2, constant_values=False)
    dil = np.zeros_like(mask)
    for dy in range(5):
        for dx in range(5):
            dil |= padded[dy:dy + h, dx:dx + w]
    ring = dil & np.logical_not(mask)
    if ring.sum() < 3:
        return 0.0
    return float(img[ring].mean() - img[mask].mean())


def _thin_fraction(mask: np.ndarray) -> float:
    """Foreground pixels with two or fewer foreground neighbors.

    Long 1-pixel traces (a stand-in for fungal hyphae) score high.
    Compact colonies score low. This is a triage feature, not a diagnosis.
    """
    n = int(mask.sum())
    if n == 0:
        return 0.0
    h, w = mask.shape
    padded = np.pad(mask.astype(np.uint8), 1, constant_values=0)
    acc = np.zeros((h, w), dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            acc += padded[dy:dy + h, dx:dx + w]
    thin = mask & (acc <= 2)
    return float(thin.sum() / n)


def _bright_near(img: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    h, w = mask.shape
    padded = np.pad(mask, 2, constant_values=False)
    near = np.zeros_like(mask)
    for dy in range(5):
        for dx in range(5):
            near |= padded[dy:dy + h, dx:dx + w]
    ring = near & np.logical_not(mask) & (img > 0.78)
    return float(ring.sum() / mask.sum())
