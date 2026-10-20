import json
import re

import pytest

from aicut_fx.cli import apply
from aicut_fx.effects import Effect


def test_unknown_types_are_refused_with_the_known_ones_listed():
    with pytest.raises(ValueError, match="title"):
        Effect.from_plan({"type": "explosion"})


def test_a_target_box_has_to_be_fractions():
    with pytest.raises(ValueError):
        Effect.from_plan({"type": "tracked_label", "clip": "a", "text": "x", "target": {"at": 0, "box": [10, 10, 50, 50]}})


def test_unknown_clip_is_named(project):
    with pytest.raises(KeyError, match="nope"):
        apply(project, {"effects": [{"type": "title", "clip": "nope", "text": "T"}]})


def test_end_to_end(project):
    plan = {"effects": [
        {"type": "title", "text": "Hello", "kicker": "K", "at": 0, "duration": 2},
        {"type": "tracked_label", "clip": "b", "text": "Box <3", "target": {"at": 0.5, "box": [0.22, 0.31, 0.25, 0.14]}},
    ]}
    out = apply(project, plan)
    film = (project / "film.html").read_text()
    assert 'src="fx/title-1.html#t=0,2" at="0"' in film
    assert 'src="fx/label-2.html#t=0,1.5" at="3"' in film

    label = (project / "fx" / "label-2.html").read_text()
    assert "Box &lt;3" in label
    keys = json.loads(re.search(r"const KEYS = (\[.*?\]);", label).group(1))
    assert keys[0][0] == pytest.approx(0, abs=0.1) and keys[-1][0] == pytest.approx(1.5, abs=0.1)
    # the square moves right and down: so does the label
    assert keys[-1][1] > keys[0][1] and keys[-1][2] > keys[0][2]

    assert "window.film" in (project / "fx" / "title-1.html").read_text()
    warned = [n for n in out.notes if "card.html" in n]
    assert len(warned) == 1 and warned[0].startswith("fx-label-2")   # the title (0–2 s) is clear of the card


def test_warns_when_an_effect_shares_the_screen_with_a_persons_page(project):
    out = apply(project, {"effects": [{"type": "title", "text": "T", "at": 3.8, "duration": 0.5}]})
    assert any("card.html" in n for n in out.notes)
