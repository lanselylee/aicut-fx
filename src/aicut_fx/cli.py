"""aicut-fx apply <openfilm project> <plan.json>

Adds the effects of a plan to an OpenFilm project: reads its film.html, grounds each effect in the footage (tracking,
empty space, colors), and writes the effects as pages on tracks of their own. Running it again replaces them; the
rest of the film is left as it is.

A plan is {"effects": [{"type": ..., ...}, ...]}; `aicut-fx types` lists the types.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from .effects import Built, Context, Effect
from .openfilm import Project


def apply(root: Path, plan: dict) -> Built:
    effects = [Effect.from_plan(spec) for spec in plan.get("effects", [])]
    ctx = Context(Project(root))
    out = Built()
    for i, fx in enumerate(effects, 1):
        b = fx.build(ctx, i)
        out.clips += b.clips
        out.pages.update(b.pages)
        out.notes += b.notes
    for c in out.clips:
        for src, a, b in ctx.project.other_pages():
            if c.at < b and a < c.at + c.length:
                out.notes.append(f"{c.id} is on screen with {src} ({max(a, c.at):g}–{min(b, c.at + c.length):g}s): "
                                 "check they do not cover each other")
    ctx.project.write_fx(out.clips, out.pages)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply", help="add a plan's effects to a project")
    a.add_argument("project", type=Path)
    a.add_argument("plan", type=Path)
    sub.add_parser("types", help="list the effect types and their parameters")
    args = ap.parse_args()

    if args.cmd == "types":
        for name, cls in sorted(Effect.REGISTRY.items()):
            params = [p for p in inspect.signature(cls.__init__).parameters.values() if p.name != "self"]
            print(f"{name}: " + ", ".join(p.name if p.default is p.empty else f"{p.name}={p.default!r}" for p in params))
            print(f"    {inspect.getdoc(cls).splitlines()[0]}")
        return

    out = apply(args.project, json.loads(args.plan.read_text()))
    for c in out.clips:
        print(f"+ {c.id:<14} {c.at:>6.2f}s  {c.length:>5.2f}s  {c.src}")
    for note in out.notes:
        print(f"  · {note}")


if __name__ == "__main__":
    main()
