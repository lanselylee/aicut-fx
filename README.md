# aicut-fx

Footage-aware effects for vlogs. A label that stays on the object it names, a title set where the picture is calm, an
accent color taken from the footage itself: effects that know what is in the frame, written as an editable
[OpenFilm](https://github.com/openfilm/openfilm) project.

Part of **AICUT**, an AI video editor. It builds on
[aicut-analyzer](https://github.com/lanselylee/aicut-analyzer), which samples and scores the footage.

## Why

An LLM can write a motion graphic as code, but it does not see the footage: text lands on a face, a sticker stays put
while the cup it points at moves away, an animation ignores the camera. aicut-fx measures the picture first and
grounds every effect in it:

| Effect | Grounded in | How |
|---|---|---|
| `tracked_label` | where the object is, frame by frame | Lucas-Kanade optical flow on corner points, forward-backward checked, tracked both ways from the given box |
| `title` | where the picture stays calm for the title's whole duration | edge energy (Sobel) on a grid, averaged over every frame under the title; the calmest box inside the safe area |
| both | the footage's colors | k-means palette; the most vivid color becomes the accent |

Text color follows the brightness under it; a label fades while its object is lost.

```
plan.json ─┐
           ├─▶ effects (Strategy + registry) ─▶ analysis (track, space, palette) ─▶ pages with frame(t) ─▶ film.html
film.html ─┘                                                                                             (OpenFilm)
```

The result is ordinary OpenFilm: every label and title is a page on a track of its own, which a person can move,
retime or restyle in OpenFilm Studio. Running aicut-fx again replaces only its own tracks (`data-aicut-fx`), never the
person's.

## Usage

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and an OpenFilm project whose `film.html` cuts the footage.

```sh
uv sync
uv run aicut-fx types                                  # the effect types and their parameters
uv run aicut-fx apply path/to/project examples/autumn-vlog.json
npx openfilm open path/to/project                      # see and edit it in Studio
```

A plan names effects by type, and refers to the film's clips by id:

```json
{
  "effects": [
    {"type": "title", "clip": "street-tree", "text": "秋天的一个下午", "kicker": "SEATTLE · OCTOBER", "duration": 4},
    {"type": "tracked_label", "clip": "elevator", "text": "Matcha 🍵",
     "target": {"at": 0.3, "box": [0.22, 0.37, 0.56, 0.27]}}
  ]
}
```

`target` is where the object is at one moment (`at`, seconds into the source file; `box`, x/y/w/h as fractions of the
frame). The tracker follows it from there in both directions.

aicut-fx warns when an effect shares the screen with a page the person made (a title card, an end card), since it
cannot yet see where that page draws.

## Design

- **Effects are parametric templates, not generated code.** A planner (today a person, next an LLM) can only ask for
  types in the registry, with checked parameters. The same plan always gives the same film, and the measured data
  (track points, placement) goes straight into the page.
- **Strategy + registry.** Each effect subclasses `Effect` and registers itself by `type`; `Effect.from_plan` builds
  one from a plan entry and lists the known types when asked for an unknown one.
- **Shared context.** `Context` decodes each clip once and answers "which frames are on screen between film times
  a and b", across cuts.
- **Owns its tracks, nothing else.** `Project.write_fx` rewrites only sections marked `data-aicut-fx` and the `fx/`
  folder; overlapping effects are spread over as many tracks as needed (interval partitioning).
- **Pages obey OpenFilm's contract**: the picture depends only on t, so preview, scrubbing and export agree.

## Tests

```sh
uv run pytest
```

19 tests on synthetic footage: a textured square moving on a known path (the tracker must stay within 4 px of it,
forward and backward, and report low confidence when it vanishes), calm and busy areas for placement, palettes,
the film.html round trip (writing twice replaces; writing nothing restores the file byte for byte), and an end-to-end
run on an encoded MP4.

## Roadmap

- [ ] LLM planner: a prompt and the footage analysis in, a validated plan out
- [ ] Find targets by name with a vision model, instead of giving boxes by hand
- [ ] Keep text off faces (face detection as a no-go area for placement)
- [ ] Beat-synced effects: zoom punches and cuts on the music's beats
- [ ] Transitions that follow the camera's motion (whip pans)
- [ ] Color grade presets
- [ ] Know where a person's own pages draw, not only when

## License

MIT
