from aicut_fx.openfilm import FilmClip, Project, lanes


def clip(at, length, i=0):
    return FilmClip(src=f"fx/{i}.html", at=at, length=length, id=f"c{i}")


def test_reads_stage_and_video_clips(project):
    p = Project(project)
    assert p.stage == (1080, 1920)
    v = p.videos()
    assert set(v) == {"a", "b"}
    assert (v["b"].source_in, v["b"].source_out, v["b"].at) == (0.5, 2.0, 3.0)
    assert v["b"].to_source(4.0) == 1.5


def test_finds_the_pages_a_person_made(project):
    assert Project(project).other_pages() == [("card.html", 3.5, 4.5)]


def test_overlapping_clips_go_on_separate_tracks():
    out = lanes([clip(0, 4, 1), clip(2, 1, 2), clip(4, 1, 3), clip(3.5, 2, 4)])
    assert [[c.id for c in lane] for lane in out] == [["c1", "c3"], ["c2", "c4"]]


def test_writing_twice_replaces_only_its_own_tracks(project):
    p = Project(project)
    p.write_fx([clip(0, 4, 1), clip(1, 1, 2)], {"1.html": "x", "2.html": "y"})
    first = (project / "film.html").read_text()
    assert first.count('data-aicut-fx="1"') == 2
    assert 'id="card"' in first and 'id="a"' in first

    p.write_fx([clip(0, 1, 3)], {"3.html": "z"})
    text = (project / "film.html").read_text()
    assert text.count('data-aicut-fx="1"') == 1
    assert sorted(f.name for f in (project / "fx").iterdir()) == ["3.html"]
    assert 'id="card"' in text and 'id="b"' in text


def test_writing_nothing_restores_the_film(project):
    before = (project / "film.html").read_text()
    p = Project(project)
    p.write_fx([clip(0, 2)], {"0.html": "x"})
    p.write_fx([], {})
    assert (project / "film.html").read_text() == before
