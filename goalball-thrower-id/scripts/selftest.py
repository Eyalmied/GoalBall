#!/usr/bin/env python3
"""
Check the installation and the geometry, without touching a video.

Run this first, and any time something behaves oddly. It answers two
questions: are the packages there, and is the maths right?

    python scripts/selftest.py
    python scripts/selftest.py --config config/mygame.json   # also check the files
"""

from __future__ import annotations

import importlib
import pathlib
import sys

from _common import base_parser, resolve
from goalball import logs

CORE = [
    ("cv2", "opencv-python", "reading video, drawing the windows"),
    ("numpy", "numpy", "all the arithmetic"),
    ("ultralytics", "ultralytics", "finding people, poses and the ball"),
]
APPEARANCE = [
    ("torchreid", "torchreid", "the OSNet appearance model (--appearance osnet)"),
    ("transformers", "transformers", "the DINOv2 appearance model (--appearance dinov2)"),
]


def check_packages() -> bool:
    ok = True
    rows = []
    for module, package, why in CORE:
        try:
            found = importlib.import_module(module)
            rows.append(["yes", package,
                         getattr(found, "__version__", "?"), why])
        except Exception:
            rows.append(["NO", package, "-", why])
            ok = False
    logs.table(["installed", "package", "version", "what it is for"], rows)

    available = []
    rows = []
    for module, package, why in APPEARANCE:
        try:
            with logs.muted():
                found = importlib.import_module(module)
            rows.append(["yes", package, getattr(found, "__version__", "?"), why])
            available.append(package)
        except Exception:
            rows.append(["no", package, "-", why])
    logs.blank()
    logs.table(["installed", "package", "version", "what it is for"], rows)

    if not ok:
        logs.blank()
        logs.error("a core package is missing")
        logs.hint("python -m pip install -r requirements.txt")
    if not available:
        logs.blank()
        logs.error("no appearance model is installed")
        logs.hint("python -m pip install torch torchreid tensorboard   (recommended)")
        logs.hint("python -m pip install torch transformers pillow     (alternative)")
        ok = False
    return ok


def check_geometry() -> bool:
    """The court mapping and the change-of-ends flip, on a made-up court."""
    import numpy as np

    from goalball import halves
    from goalball.court import Court, order_corners

    ok = True
    # A perspective-looking trapezoid: narrow at the top, wide at the bottom.
    corners = [[0.15, 0.37], [0.86, 0.37], [0.98, 0.82], [0.03, 0.82]]
    court = Court(order_corners(corners))

    named = {"top-left": (0.15, 0.37), "top-right": (0.86, 0.37),
             "bottom-right": (0.98, 0.82), "bottom-left": (0.03, 0.82)}
    for name, point in named.items():
        u, v = court.to_plane(*point)
        if not (abs(u - round(u)) < 1e-3 and abs(v - round(v)) < 1e-3):
            logs.error(f"the {name} corner should map to a corner of the flat "
                       f"court, got ({u:.3f}, {v:.3f})")
            ok = False
    if ok:
        logs.ok("the four corners map onto the corners of the flat court")

    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(400):
        x = float(rng.uniform(0.1, 0.9))
        y = float(rng.uniform(0.4, 0.8))
        back = court.mirror_norm(*court.mirror_norm(x, y))
        worst = max(worst, abs(back[0] - x), abs(back[1] - y))
    if worst < 1e-4:
        logs.ok(f"changing ends twice returns every point to where it started "
                f"(worst error {worst:.1e} of a frame)")
    else:
        logs.error(f"changing ends twice does not return the original point "
                   f"(worst error {worst:.1e})")
        ok = False

    # Three players a side, symmetric about the centre: half 2 must swap them.
    roster = {
        1: {"name": "A1", "team": "A", "home_zone": [0.35, 0.44]},
        2: {"name": "A2", "team": "A", "home_zone": [0.50, 0.42]},
        3: {"name": "A3", "team": "A", "home_zone": [0.66, 0.44]},
        4: {"name": "B1", "team": "B", "home_zone": [0.28, 0.76]},
        5: {"name": "B2", "team": "B", "home_zone": [0.50, 0.78]},
        6: {"name": "B3", "team": "B", "home_zone": [0.74, 0.76]},
    }
    plan = halves.build(1, "2", None, 25.0)
    first = plan.team_ends(roster, court, 1)
    second = plan.team_ends(roster, court, 2)
    if first and second and first["A"] != second["A"] and first["B"] != second["B"]:
        logs.ok(f"half 1: A at the {first['A']}, B at the {first['B']}  "
                f"{logs.sym('arrow')}  half 2: A at the {second['A']}, "
                f"B at the {second['B']}")
    else:
        logs.error(f"the teams did not change ends between halves "
                   f"(half 1 {first}, half 2 {second})")
        ok = False

    zones_one = plan.home_zones(roster, court, 1)
    zones_two = plan.home_zones(roster, court, 2)
    # left/right must swap too: player 1 sits left of 3 in half 1, right in half 2
    if (zones_one[1][0] < zones_one[3][0]) and (zones_two[1][0] > zones_two[3][0]):
        logs.ok("left and right swap with the ends, as they do on court")
    else:
        logs.error("left and right did not swap with the ends")
        ok = False
    return ok


def check_files(cfg) -> bool:
    ok = True
    rows = []
    for label, path, needed in [
        ("video", cfg.video, True),
        ("throws CSV", cfg.events, True),
        ("gallery", pathlib.Path(cfg.gallery_dir) / "manifest.json", False),
        ("court", cfg.court_file, False),
        ("pose model", cfg.pose_model, False),
        ("ball model", cfg.ball_model, False),
        ("labels", cfg.labels, False),
    ]:
        if not path:
            rows.append(["not set", label, "-"])
            if needed:
                ok = False
            continue
        if label.endswith("model") and "/" not in str(path):
            # a bare model name: ultralytics fetches it the first time it runs
            rows.append(["downloads", label, str(path)])
            continue
        exists = pathlib.Path(path).exists()
        rows.append(["yes" if exists else ("NO" if needed else "not yet"),
                     label, str(path)])
        if needed and not exists:
            ok = False
    logs.table(["there?", "what", "path"], rows)
    if any(r[0] == "not yet" for r in rows):
        logs.hint("'not yet' just means that step has not been run yet")
    if not ok:
        logs.blank()
        logs.error("something required is missing - check the paths in your config")
    return ok


def main() -> None:
    parser = base_parser(__doc__)
    args = parser.parse_args()

    logs.title("self test", "packages and geometry")

    total = 3 if args.config else 2
    logs.step(1, total, "Checking the packages")
    packages_ok = check_packages()
    logs.step_done()

    logs.step(2, total, "Checking the court mapping and the change of ends")
    geometry_ok = check_geometry()
    logs.step_done()

    files_ok = True
    if args.config:
        cfg = resolve(args)
        logs.step(3, total, "Checking the files in your config")
        files_ok = check_files(cfg)
        logs.step_done()

    logs.section("RESULT")
    if packages_ok and geometry_ok and files_ok:
        logs.ok("everything checks out")
        logs.blank()
        logs.next_steps([
            "run the whole pipeline:  python scripts/run_pipeline.py "
            "--config config/mygame.json --half 1",
            "or read docs/USAGE.md",
        ])
        return
    logs.error("something above needs fixing first")
    logs.blank()
    sys.exit(1)


if __name__ == "__main__":
    main()
