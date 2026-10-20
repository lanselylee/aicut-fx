from pathlib import Path

import cv2
import numpy as np
import pytest

W, H = 360, 640


def textured(rng, h, w):
    small = rng.random((h // 8, w // 8, 3)).astype(np.float32)
    return (cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * 200 + 25).astype(np.uint8)


def moving_object_frames(n=45, start=(80, 200), step=(3, 4), size=(90, 90), seed=0):
    """A flat gray background with a textured square moving `step` px per frame: (frames, true boxes in px)."""
    rng = np.random.default_rng(seed)
    patch = textured(rng, size[1], size[0])
    frames, boxes = [], []
    for i in range(n):
        f = np.full((H, W, 3), 110, np.uint8)
        x, y = start[0] + step[0] * i, start[1] + step[1] * i
        f[y:y + size[1], x:x + size[0]] = patch
        frames.append(f)
        boxes.append((x, y, size[0], size[1]))
    return frames, np.array(boxes, float)


def write_video(frames, path: Path, fps=30.0) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        out.write(f)
    out.release()
    return path


FILM = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1080, height=1920">
</head>
<body>
<section>
  <iframe id="card" src="card.html#t=0,1" at="3.5"></iframe>
</section>
<section>
  <video id="a" src="assets/a.mp4#t=0,3" volume="0.5"></video>
  <video id="b" src="assets/a.mp4#t=0.5,2" at="3"></video>
</section>
</body>
</html>
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """An OpenFilm project with one synthetic clip used twice, and a page a person made."""
    (tmp_path / "assets").mkdir()
    frames, _ = moving_object_frames(n=90, step=(1, 2))
    write_video(frames, tmp_path / "assets" / "a.mp4")
    (tmp_path / "film.html").write_text(FILM)
    (tmp_path / "card.html").write_text("<!doctype html><p>card</p>")
    return tmp_path
