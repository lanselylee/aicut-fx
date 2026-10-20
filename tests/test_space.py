import numpy as np

from aicut_fx.space import GRID, SpaceFinder, accent, busyness, palette


def test_flat_areas_are_calm_and_textured_ones_busy():
    rng = np.random.default_rng(0)
    img = np.full((640, 360), 128, np.uint8)
    img[320:] = rng.integers(0, 255, (320, 360))
    b = busyness(img)
    assert b[: GRID[0] // 2 - 1].max() < 0.05 and b[GRID[0] // 2 + 1:].mean() > 0.5


def test_places_the_box_in_the_calm_part():
    busy = np.ones(GRID)
    busy[20:26, :] = 0.0                       # a calm band from 62% to 81% of the height
    place = SpaceFinder().find(busy, np.full(GRID, 100.0), (0.8, 0.15))
    x, y, w, h = place.box
    assert 20 / 32 - 0.02 <= y and y + h <= 26 / 32 + 0.02
    assert place.calm > 0.9


def test_stays_out_of_the_margins():
    place = SpaceFinder(margin=0.06).find(np.zeros(GRID), np.zeros(GRID), (0.5, 0.1))
    x, y, w, h = place.box
    assert x >= 0.06 and y >= 0.06 and x + w <= 0.94 + 1e-9 and y + h <= 0.94 + 1e-9


def test_dark_text_on_bright_areas():
    assert SpaceFinder().find(np.zeros(GRID), np.full(GRID, 230.0), (0.5, 0.1)).tone == "dark"
    assert SpaceFinder().find(np.zeros(GRID), np.full(GRID, 90.0), (0.5, 0.1)).tone == "light"


def test_palette_and_accent_pick_the_vivid_color():
    frame = np.zeros((170, 96, 3), np.uint8)
    frame[:, :] = (120, 120, 120)              # gray, BGR
    frame[:40, :] = (20, 180, 230)             # golden yellow, BGR
    colors = palette([frame], k=2)
    assert colors[0] == (120, 120, 120)
    r, g, b = accent(colors)
    assert r > 200 and g > 150 and b < 120
