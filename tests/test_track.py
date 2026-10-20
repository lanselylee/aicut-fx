import numpy as np

from aicut_analyzer.frames import Clip, Frame
from aicut_fx.track import BoxTracker
from conftest import H, W, moving_object_frames

import cv2


def as_clip(frames, fps=15.0):
    from pathlib import Path
    return Clip(path=Path("x.mp4"), duration=len(frames) / fps, fps=fps,
                frames=[Frame(t=i / fps, gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), bgr=f) for i, f in enumerate(frames)])


def center_error(track, truth):
    got = (track.boxes[:, :2] + track.boxes[:, 2:] / 2) * [W, H]
    want = truth[:, :2] + truth[:, 2:] / 2
    return np.linalg.norm(got - want, axis=1)


def test_follows_a_moving_object_forward():
    frames, truth = moving_object_frames(n=45)
    box = truth[0] / [W, H, W, H]
    track = BoxTracker(smooth=1).track(as_clip(frames), at=0.0, box=tuple(box))
    assert center_error(track, truth).max() < 4      # px, at 360 × 640
    assert track.confidence.mean() > 0.8


def test_follows_it_backward_from_a_box_given_mid_clip():
    frames, truth = moving_object_frames(n=45)
    mid = 30
    track = BoxTracker(smooth=1).track(as_clip(frames), at=mid / 15, box=tuple(truth[mid] / [W, H, W, H]))
    assert center_error(track, truth)[:mid].max() < 4


def test_interpolates_between_frames():
    frames, truth = moving_object_frames(n=10)
    track = BoxTracker(smooth=1).track(as_clip(frames), at=0.0, box=tuple(truth[0] / [W, H, W, H]))
    halfway = track.at(0.5 / 15)
    np.testing.assert_allclose(halfway, (track.boxes[0] + track.boxes[1]) / 2, atol=1e-9)


def test_reports_low_confidence_when_the_object_vanishes():
    frames, truth = moving_object_frames(n=30)
    blank = [np.full_like(frames[0], 110) for _ in range(15)]
    track = BoxTracker(smooth=1).track(as_clip(frames + blank), at=0.0, box=tuple(truth[0] / [W, H, W, H]))
    assert track.confidence[:29].mean() > 0.8
    assert track.confidence[31:].max() < 0.3
