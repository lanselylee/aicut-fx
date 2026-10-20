"""Effects: each type reads what it needs from the footage and returns OpenFilm pages placed in the film.

Effects are parametric templates, not free-form code: a plan names a type and gives its parameters, and the effect
grounds them in the picture (where the object is, where the picture is calm, which colors it has). A registry maps
type names to classes, so a planner (a person or an LLM) can only ask for effects that exist, with checked fields.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from aicut_analyzer.frames import Clip, FrameSampler

from .openfilm import FX_DIR, FilmClip, Project, VideoClip
from .space import SpaceFinder, accent, busyness, luminance, palette
from .track import BoxTracker


class Context:
    """What effects share: the project, its clips, and each clip's sampled frames (decoded once)."""

    def __init__(self, project: Project, sampler: FrameSampler | None = None):
        self.project = project
        self.stage = project.stage
        self.videos = project.videos()
        self.sampler = sampler or FrameSampler()
        self._samples: dict[str, Clip] = {}

    def video(self, clip_id: str) -> VideoClip:
        if clip_id not in self.videos:
            raise KeyError(f"no video clip {clip_id!r} in film.html (there are: {', '.join(self.videos)})")
        return self.videos[clip_id]

    def sample(self, v: VideoClip) -> Clip:
        if v.src not in self._samples:
            self._samples[v.src] = self.sampler.sample(self.project.root / v.src)
        return self._samples[v.src]

    def visible(self, film_t0: float, film_t1: float) -> list[tuple[VideoClip, list]]:
        """The sampled frames shown between two film times, clip by clip."""
        out = []
        for v in self.videos.values():
            a, b = max(film_t0, v.at), min(film_t1, v.at + v.length)
            if b <= a:
                continue
            s0, s1 = v.to_source(a), v.to_source(b)
            frames = [f for f in self.sample(v).frames if s0 <= f.t <= s1]
            if frames:
                out.append((v, frames))
        return out


@dataclass
class Built:
    clips: list[FilmClip] = field(default_factory=list)
    pages: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class Effect(ABC):
    type: ClassVar[str]
    REGISTRY: ClassVar[dict[str, type["Effect"]]] = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        Effect.REGISTRY[cls.type] = cls

    @classmethod
    def from_plan(cls, spec: dict) -> "Effect":
        spec = dict(spec)
        kind = spec.pop("type", None)
        if kind not in cls.REGISTRY:
            raise ValueError(f"unknown effect type {kind!r}; known: {', '.join(sorted(cls.REGISTRY))}")
        return cls.REGISTRY[kind](**spec)

    @abstractmethod
    def build(self, ctx: Context, n: int) -> Built:
        """Pages and their clips; `n` makes page names unique."""


def rgb(c) -> str:
    return f"rgb({c[0]}, {c[1]}, {c[2]})"


class TrackedLabel(Effect):
    """A label that stays on an object: a pill of text above it and a line down to a dot on it, following the object
    as it and the camera move. Fades while the object is lost."""
    type = "tracked_label"

    def __init__(self, clip: str, text: str, target: dict, color: str | None = None):
        self.clip, self.text, self.color = clip, text, color
        self.target_at = float(target["at"])
        self.target_box = tuple(float(x) for x in target["box"])
        if len(self.target_box) != 4 or not all(0 <= x <= 1 for x in self.target_box):
            raise ValueError("target.box is [x, y, w, h] as fractions of the frame, each 0 to 1")

    def build(self, ctx: Context, n: int) -> Built:
        v = ctx.video(self.clip)
        sample = ctx.sample(v)
        track = BoxTracker().track(sample, self.target_at, self.target_box)
        W, H = ctx.stage
        keys = []
        for t, (x, y, w, h), c in zip(track.times, track.boxes, track.confidence):
            if v.source_in - 0.1 <= t <= v.source_out + 0.1:
                keys.append([round(t - v.source_in, 3), round((x + w / 2) * W, 1), round((y + h / 2) * H, 1),
                             round(y * H, 1), round(float(c), 2)])
        color = self.color or rgb(accent(palette([f.bgr for f in sample.frames[::5]])))
        lost = float(np.mean([k[4] < 0.3 for k in keys])) if keys else 1.0
        name = f"label-{n}.html"
        built = Built(clips=[FilmClip(src=f"{FX_DIR}/{name}", at=v.at, length=round(v.length, 3), id=f"fx-label-{n}")],
                      pages={name: LABEL_PAGE.replace("__TEXT__", _html(self.text)).replace("__COLOR__", color)
                             .replace("__KEYS__", json.dumps(keys)).replace("__LENGTH__", f"{v.length:g}")
                             .replace("__W__", str(W)).replace("__H__", str(H))})
        if lost > 0.25:
            built.notes.append(f"{self.clip}: the object was lost {lost:.0%} of the time; check the label")
        return built


class Title(Effect):
    """A title placed where the picture stays calm for as long as it is on screen (often the sky), in a color that
    reads on that area, with an accent line in the footage's own most vivid color."""
    type = "title"

    def __init__(self, text: str, kicker: str = "", at: float | None = None, clip: str | None = None,
                 duration: float = 4.0, size: tuple[float, float] = (0.84, 0.15)):
        self.text, self.kicker, self.duration, self.size = text, kicker, float(duration), tuple(size)
        self.at, self.clip = at, clip

    def build(self, ctx: Context, n: int) -> Built:
        start = self.at if self.at is not None else (ctx.video(self.clip).at if self.clip else 0.0)
        seen = ctx.visible(start, start + self.duration)
        if not seen:
            raise ValueError(f"title at {start}s: no footage under it")
        frames = [f for _, fs in seen for f in fs]
        busy = np.mean([busyness(f.gray) for f in frames], axis=0)
        lum = np.mean([luminance(f.bgr) for f in frames], axis=0)
        place = SpaceFinder().find(busy, lum, self.size)
        W, H = ctx.stage
        x, y, w, h = place.box
        ink, shade = ("#ffffff", "rgba(0,0,0,.45)") if place.tone == "light" else ("#1d1d1f", "rgba(255,255,255,.35)")
        line = rgb(accent(palette([f.bgr for f in frames[::3]])))
        name = f"title-{n}.html"
        page = (TITLE_PAGE.replace("__TEXT__", _html(self.text)).replace("__KICKER__", _html(self.kicker))
                .replace("__LEFT__", f"{x * W:.0f}").replace("__TOP__", f"{y * H:.0f}")
                .replace("__WIDTH__", f"{w * W:.0f}").replace("__HEIGHT__", f"{h * H:.0f}")
                .replace("__INK__", ink).replace("__SHADE__", shade).replace("__LINE__", line)
                .replace("__DURATION__", f"{self.duration:g}").replace("__W__", str(W)).replace("__H__", str(H)))
        return Built(clips=[FilmClip(src=f"{FX_DIR}/{name}", at=start, length=self.duration, id=f"fx-title-{n}")],
                     pages={name: page},
                     notes=[f"title at {x:.2f},{y:.2f} (calm {place.calm:.2f}, {place.tone} text)"])


def _html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


LABEL_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; width: __W__px; height: __H__px; background: transparent; overflow: hidden; }
  #pill { position: absolute; left: 0; top: 0; padding: 18px 34px; border-radius: 999px; white-space: nowrap;
          background: rgba(255,255,255,.92); color: #1d1d1f; font: 600 46px/1.1 "PingFang SC", "Helvetica Neue", sans-serif;
          box-shadow: 0 10px 40px rgba(0,0,0,.25); transform-origin: 50% 100%; }
  #line { position: absolute; width: 4px; background: rgba(255,255,255,.92); transform-origin: 50% 0; }
  #dot { position: absolute; width: 26px; height: 26px; margin: -13px 0 0 -13px; border-radius: 50%;
         background: __COLOR__; border: 5px solid #fff; box-sizing: border-box; box-shadow: 0 0 0 10px rgba(255,255,255,.25); }
</style>
</head>
<body>
<div id="line"></div>
<div id="dot"></div>
<div id="pill">__TEXT__</div>
<script type="module">
// [t, center x, center y, top y, confidence] per tracked frame, in stage px, t from the clip's in point
const KEYS = __KEYS__;
const LENGTH = __LENGTH__, W = __W__, H = __H__, GAP = 150, MARGIN = 48;
const pill = document.getElementById('pill'), line = document.getElementById('line'), dot = document.getElementById('dot');
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const ease = x => 1 - Math.pow(1 - clamp(x, 0, 1), 3);
const back = x => { x = clamp(x, 0, 1); const c = 1.7; return 1 + (c + 1) * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); };
function at(t) {
  if (!KEYS.length) return [W / 2, H / 2, H / 2, 0];
  if (t <= KEYS[0][0]) return KEYS[0].slice(1);
  for (let i = 1; i < KEYS.length; i++) {
    if (t <= KEYS[i][0]) {
      const a = KEYS[i - 1], b = KEYS[i], k = (t - a[0]) / (b[0] - a[0] || 1);
      return a.slice(1).map((v, j) => v + (b[j + 1] - v) * k);
    }
  }
  return KEYS[KEYS.length - 1].slice(1);
}
window.film = {
  duration: LENGTH,
  ready: document.fonts.ready,
  frame(t) {
    const [cx, cy, top, conf] = at(t);
    const seen = clamp((conf - 0.15) / 0.25, 0, 1);
    const show = back(t / 0.45) * (1 - ease((t - (LENGTH - 0.35)) / 0.35));
    const pw = pill.offsetWidth, ph = pill.offsetHeight;
    let py = top - GAP - ph;
    const below = py < MARGIN;
    if (below) py = cy + GAP;
    const px = clamp(cx - pw / 2, MARGIN, W - MARGIN - pw);
    pill.style.transform = `translate(${px}px, ${py}px) scale(${Math.max(show, 0)})`;
    pill.style.opacity = clamp(show, 0, 1) * seen;
    dot.style.left = cx + 'px'; dot.style.top = cy + 'px';
    dot.style.opacity = clamp(show, 0, 1) * seen;
    const y0 = below ? cy : py + ph, y1 = below ? py : cy;
    line.style.left = (cx - 2) + 'px'; line.style.top = y0 + 'px';
    line.style.height = Math.max(0, y1 - y0) * ease((t - 0.1) / 0.4) + 'px';
    line.style.opacity = clamp(show, 0, 1) * seen;
  },
};
</script>
</body>
</html>
"""

TITLE_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; width: __W__px; height: __H__px; background: transparent; overflow: hidden; }
  .wrap { position: absolute; left: __LEFT__px; top: __TOP__px; width: __WIDTH__px; height: __HEIGHT__px;
          display: flex; flex-direction: column; justify-content: center; color: __INK__;
          font-family: "PingFang SC", "Helvetica Neue", sans-serif; text-shadow: 0 2px 24px __SHADE__; }
  #kicker { font-size: 30px; letter-spacing: .35em; font-weight: 500; opacity: .9; }
  #title { font-size: 104px; font-weight: 700; margin-top: 10px; letter-spacing: .04em; line-height: 1.1; }
  #line { height: 6px; border-radius: 3px; background: __LINE__; margin-top: 24px; width: 0; }
</style>
</head>
<body>
<div class="wrap">
  <div id="kicker">__KICKER__</div>
  <div id="title">__TEXT__</div>
  <div id="line"></div>
</div>
<script type="module">
const ease = x => 1 - Math.pow(1 - Math.min(Math.max(x, 0), 1), 3);
const kicker = document.getElementById('kicker'), title = document.getElementById('title'), line = document.getElementById('line');
const DURATION = __DURATION__;
window.film = {
  duration: DURATION,
  ready: document.fonts.ready,
  frame(t) {
    const out = 1 - ease((t - (DURATION - 0.6)) / 0.6);
    const k = ease(t / 0.8), h = ease((t - 0.3) / 0.9);
    kicker.style.opacity = k * out;
    kicker.style.transform = `translateY(${(1 - k) * 20}px)`;
    title.style.opacity = h * out;
    title.style.transform = `translateY(${(1 - h) * 30}px)`;
    line.style.width = `${ease((t - 0.8) / 1.2) * 320}px`;
    line.style.opacity = out;
  },
};
</script>
</body>
</html>
"""
