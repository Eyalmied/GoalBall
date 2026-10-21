#!/usr/bin/env python3
"""
STEP 1 of 4 - build the reference gallery. The only manual step.

You tag all the players in a handful of frames. Each tag records two things:
what that player looks like, and where they stand on court - their "home zone".
The home zone is what lets the pipeline tell two teammates apart later, since
identical uniforms and blackout eyeshades defeat every appearance model.

    python scripts/build_gallery.py --config config/mygame.json --half 1

TAG FROM EITHER HALF
    Press `h` during tagging to flip between a frame from the first half and
    one from the second, as often as you like, for any player. A player who is
    blocked or facing away for the whole first half is often perfectly clear in
    the second.

    Every position is stored with the half it came from, so this is safe:
    medians are taken per half, never across them. A player tagged in both ends
    up with a MEASURED home zone for each, which beats measuring one and
    mirroring it into the other.

    --half only says which half to START in, and which one the gallery reports
    as its reference. Giving --half2-start (or half2_start in the config) is
    what makes the second half available at all.
"""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np

from _common import base_parser, parse_timestamp, resolve
from goalball import court as court_module
from goalball import gallery as gallery_module
from goalball import halves, logs, video

EXAMPLE_ROSTER = {
    "1": {"name": "Player A", "team": "ISR"},
    "2": {"name": "Player B", "team": "ISR"},
    "3": {"name": "Player C", "team": "ISR"},
    "4": {"name": "Player D", "team": "CHN"},
    "5": {"name": "Player E", "team": "CHN"},
    "6": {"name": "Player F", "team": "CHN"},
}


def load_roster(cfg, roster_path) -> dict:
    """The roster comes from the config file or a small JSON file - never code."""
    if roster_path:
        file = pathlib.Path(roster_path)
        if not file.exists():
            logs.fail(f"roster file not found: {roster_path}")
        try:
            raw = json.loads(file.read_text())
        except json.JSONDecodeError as e:
            logs.fail(f"roster file is not valid JSON ({e})")
        raw = raw.get("roster", raw)
        return {int(k): dict(v) for k, v in raw.items()}

    roster = cfg.player_config()
    if not roster:
        logs.error("no roster given, so the players have no names or teams")
        logs.info("")
        logs.info("Add a 'roster' block to your config file, or pass "
                  "--roster roster.json, shaped like this:")
        logs.info("")
        for line in json.dumps({"roster": EXAMPLE_ROSTER}, indent=2).splitlines():
            logs.info("    " + line)
        logs.info("")
        logs.info("The team labels are yours to choose; they are what the "
                  "predictions will report.")
        raise SystemExit(1)
    return roster


def check_roster(roster: dict) -> None:
    problems = []
    for pid, info in sorted(roster.items()):
        if not info.get("name"):
            problems.append(f"player {pid} has no name")
        if not info.get("team"):
            problems.append(f"player {pid} has no team")
    teams = {info.get("team") for info in roster.values() if info.get("team")}
    if len(teams) != 2:
        problems.append(f"expected exactly 2 teams, found {len(teams)}: "
                        f"{', '.join(sorted(t for t in teams if t))}")
    sizes = {}
    for info in roster.values():
        sizes[info.get("team")] = sizes.get(info.get("team"), 0) + 1
    for team, count in sizes.items():
        if count != 3:
            problems.append(f"team {team} has {count} players (goalball plays 3)")
    if problems:
        for problem in problems:
            logs.warn(problem)
        logs.hint("the pipeline still runs, but 'which team' and 'which of the "
                  "three' assume the normal 3-a-side line-up")


def half_ranges(info, half2_start) -> tuple:
    """
    ({half: (low, high)}, boundary_frame) - the frames belonging to each half.

    Without a half-2 start there is only one range, covering the whole video,
    and it is reported as half 1: nothing tells us where to split it.
    """
    last = max(1, info.frame_count - 1)
    if not half2_start:
        return {1: (0, last)}, None
    from goalball.events import mmss_to_seconds
    boundary = int(round(mmss_to_seconds(half2_start) * info.fps))
    boundary = max(0, min(boundary, last))
    ranges = {}
    if boundary > 0:
        ranges[1] = (0, boundary)
    if boundary < last:
        ranges[2] = (boundary, last)
    return ranges, boundary


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--half", default="1", choices=["1", "2"],
                        help="which half to START in, and the one the gallery "
                             "reports as its reference (default: 1). You can "
                             "switch halves freely while tagging with h.")
    parser.add_argument("--half2-start", dest="half2_start", default=None,
                        help="video timestamp where the second half begins, "
                             "e.g. 24:00. Without it the second half cannot be "
                             "offered and h does nothing.")
    parser.add_argument("--rounds", type=int, default=5,
                        help="reference images per player (default: 5). More is "
                             "better, but 5 is plenty for the position signal.")
    parser.add_argument("--roster", default=None,
                        help="JSON file with the players' names and teams "
                             "(defaults to the 'roster' block in --config)")
    parser.add_argument("--from", dest="from_time", default=None,
                        help="only offer frames after this moment (12:34 / 400s / frame)")
    parser.add_argument("--to", dest="to_time", default=None,
                        help="only offer frames before this moment")
    args = parser.parse_args()
    cfg = resolve(args, "video")

    logs.title("build the reference gallery",
               f"tagging {args.rounds} reference image(s) per player, "
               f"in half {args.half}")

    logs.step(1, 4, "Checking the roster")
    roster = load_roster(cfg, args.roster)
    check_roster(roster)
    logs.table(["id", "name", "team"],
               [[pid, roster[pid].get("name", "?"), roster[pid].get("team", "?")]
                for pid in sorted(roster)])
    logs.step_done(f"{len(roster)} players")

    logs.step(2, 4, "Opening the video")
    cap, info = video.open_video(cfg.video)
    info.describe()
    logs.step_done()

    logs.step(3, 4, "Working out which frames to offer")
    ranges, _boundary = half_ranges(info, args.half2_start or cfg.half2_start)
    clip_low = parse_timestamp(args.from_time, info.fps, 0) if args.from_time else None
    clip_high = parse_timestamp(args.to_time, info.fps, info.frame_count) \
        if args.to_time else None

    pools = {}
    for half, (low, high) in sorted(ranges.items()):
        if clip_low is not None:
            low = max(low, clip_low)
        if clip_high is not None:
            high = min(high, clip_high)
        if high <= low:
            continue
        pools[half] = [int(v) for v in
                       np.linspace(low, high, max(args.rounds * 4, 8))]
        logs.kv(f"half {half}", f"{logs.mmss(info.seconds_of(low))} to "
                                f"{logs.mmss(info.seconds_of(high))}  "
                                f"({len(pools[half])} frames on offer)")
    if not pools:
        logs.fail("the chosen time range is empty",
                  "check --from / --to and --half2-start")
    if len(pools) < 2:
        logs.warn("only one half can be offered, so you cannot press h to "
                  "switch during tagging",
                  "pass --half2-start MM:SS (or add half2_start to your "
                  "config) to make both halves available")
    start_half = int(args.half) if int(args.half) in pools else sorted(pools)[0]
    logs.kv("starting in", f"half {start_half}")
    logs.step_done()

    logs.step(4, 4, "Tagging")
    crops, samples, frame_size, counts = gallery_module.tag_interactive(
        cap, info, roster, args.rounds, pools, start_half=start_half)
    cap.release()
    cv2.destroyAllWindows()
    logs.blank()
    gallery_module.save(cfg.gallery_dir, crops, roster, samples,
                        gallery_half=start_half, frame_size=frame_size)
    logs.step_done(f"{sum(counts.values())} reference image(s)")

    # Read it straight back and show what the home zones imply. A silent
    # gallery is a gallery you cannot trust.
    logs.section("CHECK")
    reloaded_crops, player_config, meta = gallery_module.load(cfg.gallery_dir)
    court = court_module.load(cfg.court_file)
    if meta["has_home_zones"]:
        plan = halves.build(meta["gallery_half"], "auto",
                            args.half2_start or cfg.half2_start, info.fps)
        for half in sorted(pools):
            ends = plan.team_ends(player_config, court, half)
            provenance = plan.zone_provenance(player_config, half)
            measured = sum(1 for v in provenance.values() if v == "measured")
            where = ",  ".join(f"{t} at the {side}" for t, side in sorted(ends.items()))
            logs.kv(f"half {half}", f"{where}   "
                                    f"({measured}/{len(player_config)} zones "
                                    f"measured here)")
        logs.hint("if those ends look swapped against the video, re-run and "
                  "start in the other half (--half)")

    sheet = gallery_module.contact_sheet(reloaded_crops, player_config)
    sheet_path = str(pathlib.Path(cfg.gallery_dir) / "_contact_sheet.jpg")
    if sheet is not None:
        cv2.imwrite(sheet_path, sheet)

    logs.closing([("gallery", f"{cfg.gallery_dir}/"),
                  ("all references in one image", sheet_path)])
    logs.next_steps([
        "mark the court (if you have not):  python scripts/set_court.py --config ...",
        f"run the prediction:  python scripts/predict_throwers.py --config ... "
        f"--half {args.half}",
    ])


if __name__ == "__main__":
    main()
