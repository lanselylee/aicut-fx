"""Following an object through a clip: a box given at one moment, tracked forward and backward with optical flow."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aicut_analyzer.frames import Clip


@dataclass(frozen=True)
class Track:
    times: np.ndarray        # seconds in the source clip
    boxes: np.ndarray        # (n, 4): x, y, w, h as fractions of the frame
    confidence: np.ndarray   # share of the tracked points that survived each step, 0 to 1

    def at(self, t: float) -> np.ndarray:
        """The box at any time, linearly interpolated."""
        return np.array([np.interp(t, self.times, self.boxes[:, i]) for i in range(4)])


class BoxTracker:
    """Pyramidal Lucas-Kanade on corner points inside the box.

    Each step keeps the points that track forward and back to where they started (forward-backward check), moves
    the box by their median shift and scales it by the median change of their spread. When too few points are left
    (the object turned, or left the frame), it finds new corners inside the current box.
    """

    def __init__(self, max_points: int = 80, min_points: int = 8, fb_error: float = 1.0, smooth: int = 5):
        self.max_points = max_points
        self.min_points = min_points
        self.fb_error = fb_error     # px (at analysis size) a point may miss its way back by
        self.smooth = smooth         # frames in the moving average applied to the result

    def track(self, clip: Clip, at: float, box: tuple[float, float, float, float]) -> Track:
        frames = clip.frames
        h, w = frames[0].gray.shape
        start = int(np.argmin([abs(f.t - at) for f in frames]))
        px = np.array([box[0] * w, box[1] * h, box[2] * w, box[3] * h], float)

        fwd_boxes, fwd_conf = self._run([f.gray for f in frames[start:]], px)
        back_boxes, back_conf = self._run([f.gray for f in frames[start::-1]], px)
        boxes = np.vstack([back_boxes[:0:-1], fwd_boxes])
        conf = np.concatenate([back_conf[:0:-1], fwd_conf])
        boxes = self._smooth(boxes)
        norm = boxes / np.array([w, h, w, h])
        return Track(times=np.array([f.t for f in frames]), boxes=norm, confidence=conf)

    def _run(self, grays: list[np.ndarray], box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes, conf = [box.copy()], [1.0]
        pts = self._corners(grays[0], box)
        for prev, nxt in zip(grays, grays[1:]):
            if pts is None or len(pts) < self.min_points:
                pts = self._corners(prev, box)
            if pts is None or len(pts) == 0:
                boxes.append(box.copy()); conf.append(0.0)
                continue
            new, st, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, pts, None, winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, new, None, winSize=(21, 21), maxLevel=3)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pts - back, axis=2)[:, 0] < self.fb_error)
            n_before = len(pts)
            if ok.sum() < 3:
                boxes.append(box.copy()); conf.append(0.0)
                pts = None
                continue
            a, b = pts[ok][:, 0], new[ok][:, 0]
            shift = np.median(b - a, axis=0)
            spread_a = np.linalg.norm(a - a.mean(0), axis=1)
            spread_b = np.linalg.norm(b - b.mean(0), axis=1)
            keep = spread_a > 1e-3
            scale = float(np.clip(np.median(spread_b[keep] / spread_a[keep]), 0.9, 1.1)) if keep.sum() >= 3 else 1.0
            cx, cy = box[0] + box[2] / 2 + shift[0], box[1] + box[3] / 2 + shift[1]
            bw, bh = box[2] * scale, box[3] * scale
            box = np.array([cx - bw / 2, cy - bh / 2, bw, bh])
            boxes.append(box.copy()); conf.append(ok.sum() / n_before)
            pts = new[ok].reshape(-1, 1, 2)
        return np.array(boxes), np.array(conf)

    def _corners(self, gray: np.ndarray, box: np.ndarray):
        h, w = gray.shape
        x0, y0 = int(max(0, box[0])), int(max(0, box[1]))
        x1, y1 = int(min(w, box[0] + box[2])), int(min(h, box[1] + box[3]))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        mask = np.zeros_like(gray)
        mask[y0:y1, x0:x1] = 255
        return cv2.goodFeaturesToTrack(gray, self.max_points, 0.01, 4, mask=mask)

    def _smooth(self, boxes: np.ndarray) -> np.ndarray:
        k = self.smooth
        if k < 2 or len(boxes) < k:
            return boxes
        pad = np.pad(boxes, ((k // 2, k // 2), (0, 0)), mode="edge")
        return np.column_stack([np.convolve(pad[:, i], np.ones(k) / k, "valid") for i in range(4)])
