"""Reading and writing an OpenFilm project: the video clips of film.html, and the one track aicut-fx owns.

Only the track marked `data-aicut-fx` and the `fx/` folder are ever rewritten; every other track, and every change
the person made in Studio, stays as it is.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

MARK = 'data-aicut-fx="1"'
FX_DIR = "fx"
_ATTR = re.compile(r'([\w-]+)="([^"]*)"')
_OUR_TRACK = re.compile(r'\n?<section[^>]*\bdata-aicut-fx="1"[^>]*>.*?</section>', re.S)


@dataclass(frozen=True)
class VideoClip:
    id: str
    src: str            # path relative to film.html, without its fragment
    source_in: float    # seconds into the file where the clip starts
    source_out: float
    at: float           # where it starts in the film

    @property
    def length(self) -> float:
        return self.source_out - self.source_in

    def to_source(self, film_t: float) -> float:
        return self.source_in + (film_t - self.at)


@dataclass(frozen=True)
class FilmClip:
    """A clip aicut-fx adds: a page at a time in the film."""
    src: str
    at: float
    length: float
    id: str

    def tag(self) -> str:
        return f'  <iframe id="{self.id}" src="{self.src}#t=0,{self.length:g}" at="{self.at:g}"></iframe>'


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.film_path = root / "film.html"
        self.text = self.film_path.read_text()

    @property
    def stage(self) -> tuple[int, int]:
        m = re.search(r'<meta name="viewport" content="width=(\d+), height=(\d+)"', self.text)
        return (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)

    def videos(self) -> dict[str, VideoClip]:
        out = {}
        for tag in re.findall(r"<video\b[^>]*>", self.text):
            a = dict(_ATTR.findall(tag))
            src, _, frag = a["src"].partition("#t=")
            if frag:
                lo, _, hi = frag.partition(",")
                source_in, source_out = float(lo), float(hi) if hi else None
            else:
                source_in, source_out = 0.0, None
            if source_out is None:
                raise ValueError(f"{a['src']}: give the clip an out point (#t=in,out)")
            cid = a.get("id", Path(src).name)
            out[cid] = VideoClip(id=cid, src=src, source_in=source_in, source_out=source_out, at=float(a.get("at", 0)))
        return out

    def other_pages(self) -> list[tuple[str, float, float]]:
        """(src, start, end) of the pages already in the film that are not aicut-fx's: titles and cards a person
        made, which effects should not cover."""
        out = []
        for tag in re.findall(r"<iframe\b[^>]*>", _OUR_TRACK.sub("", self.text)):
            a = dict(_ATTR.findall(tag))
            src, _, frag = a["src"].partition("#t=")
            lo, _, hi = frag.partition(",")
            if hi:
                start = float(a.get("at", 0))
                out.append((src, start, start + float(hi) - float(lo or 0)))
        return out

    def write_fx(self, clips: list[FilmClip], pages: dict[str, str]) -> None:
        """Replaces aicut-fx's tracks (the top ones) and its pages."""
        fx = self.root / FX_DIR
        if fx.exists():
            shutil.rmtree(fx)
        fx.mkdir()
        for name, html in pages.items():
            (fx / name).write_text(html)
        text = _OUR_TRACK.sub("", self.text)
        tracks = "".join(f"<section {MARK}>\n" + "\n".join(c.tag() for c in lane) + "\n</section>\n"
                         for lane in lanes(clips))
        text = text.replace("<body>\n", f"<body>\n{tracks}", 1)
        self.film_path.write_text(text)
        self.text = text


def lanes(clips: list[FilmClip]) -> list[list[FilmClip]]:
    """Clips split into tracks where none overlap (a track holds clips that do not overlap): each clip goes on the
    first track that is free by its start."""
    out: list[list[FilmClip]] = []
    for c in sorted(clips, key=lambda c: c.at):
        for lane in out:
            if lane[-1].at + lane[-1].length <= c.at + 1e-6:
                lane.append(c)
                break
        else:
            out.append([c])
    return out
