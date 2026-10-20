"""Where text can go, and what colors suit the picture.

Empty space is where the picture is calm: little edge energy (sky, a wall, a blurred background). Busyness is
measured on a coarse grid over the stage, averaged over every frame the text will be seen on, so a title is placed
where the picture stays calm for its whole duration.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GRID = (32, 18)   # rows, columns of the busyness map (9:16 friendly; any stage shape works)


def busyness(gray: np.ndarray, grid: tuple[int, int] = GRID) -> np.ndarray:
    """Edge energy per grid cell, 0 (flat) to about 1 (busy)."""
    g = cv2.GaussianBlur(gray, (0, 0), 1.2).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    cells = cv2.resize(mag, (grid[1], grid[0]), interpolation=cv2.INTER_AREA)
    return np.clip(cells / 60.0, 0, 1)


def luminance(bgr: np.ndarray, grid: tuple[int, int] = GRID) -> np.ndarray:
    """Mean brightness per grid cell, 0 to 255."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.resize(y, (grid[1], grid[0]), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class Placement:
    box: tuple[float, float, float, float]   # x, y, w, h as fractions of the stage
    calm: float                              # 1 − mean busyness under the box
    tone: str                                # "light" text on a dark area, or "dark" text on a bright one


class SpaceFinder:
    """The calmest place for a box of a given size, away from the stage's edges (the safe area)."""

    def __init__(self, margin: float = 0.06, step: float = 0.02, bright_at: float = 185.0):
        self.margin = margin
        self.step = step
        self.bright_at = bright_at

    def find(self, busy: np.ndarray, lum: np.ndarray, size: tuple[float, float]) -> Placement:
        rows, cols = busy.shape
        bw, bh = size
        integral = cv2.integral(busy.astype(np.float64))
        best = None
        for y in np.arange(self.margin, 1 - self.margin - bh + 1e-9, self.step):
            for x in np.arange(self.margin, 1 - self.margin - bw + 1e-9, self.step):
                r0, r1 = int(round(y * rows)), max(int(round((y + bh) * rows)), int(round(y * rows)) + 1)
                c0, c1 = int(round(x * cols)), max(int(round((x + bw) * cols)), int(round(x * cols)) + 1)
                total = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
                mean = total / ((r1 - r0) * (c1 - c0))
                if best is None or mean < best[0]:
                    best = (mean, x, y, (r0, r1, c0, c1))
        mean, x, y, (r0, r1, c0, c1) = best
        tone = "dark" if lum[r0:r1, c0:c1].mean() >= self.bright_at else "light"
        return Placement(box=(float(x), float(y), bw, bh), calm=float(1 - mean), tone=tone)


def palette(frames: list[np.ndarray], k: int = 5) -> list[tuple[int, int, int]]:
    """The k main colors of some BGR frames, as RGB, most common first."""
    px = np.vstack([cv2.resize(f, (48, 85), interpolation=cv2.INTER_AREA).reshape(-1, 3) for f in frames])
    px = px.astype(np.float32)
    _, labels, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                                    cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=k))
    return [tuple(int(c) for c in centers[i][::-1]) for i in order]


def accent(colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """The most vivid of the colors, lifted to a usable brightness: for lines, dots and highlights."""
    def hsv(c):
        return cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = max((hsv(c) for c in colors), key=lambda x: int(x[1]) * (0.5 + int(x[2]) / 510))
    rgb = cv2.cvtColor(np.uint8([[[h, max(s, 150), max(v, 200)]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(c) for c in rgb)
