#!/usr/bin/env python3
"""
STEP 1 of 4 - build the reference gallery. The only manual step.

You tag all the players in a handful of frames. Each tag records two things:
what that player looks like, and where they stand on court - their "home zone".
The home zone is what lets the pipeline tell two teammates apart later, since
identical uniforms and blackout eyeshades defeat every appearance model.

    python scripts/build_gallery.py --config config/mygame.json --half 1

WHY --half IS ASKED FOR HERE
    A home zone is only meaningful together with the half it was measured in,
    because the teams change ends at half-time. Tag inside ONE half, tell the
    script which one, and every later run can mirror the zones correctly for
    the other half. The half you tag in is written into the gallery, so you
    only state it once.
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


def half_frame_range(info, half: str, half2_start: str) -> tuple:
    """Restrict tagging to the half being tagged, when we know where it starts."""
    last = max(1, info.frame_count - 1)
    if not half2_start:
        return 0, last, None
    from goalball.events import mmss_to_seconds
    boundary = int(round(mmss_to_seconds(half2_start) * info.fps))
    boundary = max(0, min(boundary, last))
    if str(half) == "1":
        return 0, boundary, f"before {half2_start}"
    return boundary, last, f"from {half2_start} onwards"


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--half", default="1", choices=["1", "2"],
                        help="which half of the match you are tagging in "
                             "(default: 1). Tag inside ONE half only.")
    parser.add_argument("--half2-start", dest="half2_start", default=None,
                        help="video timestamp where the second half begins, "
                             "e.g. 24:00. Given this, the script only offers "
                             "you frames from the half you are tagging.")
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

    logs.step(3, 4, "Choosing which part of the video to tag in")
    low, high, why = half_frame_range(info, args.half,
                                      args.half2_start or cfg.half2_start)
    if args.from_time:
        low = max(low, parse_timestamp(args.from_time, info.fps, low))
    if args.to_time:
        high = min(high, parse_timestamp(args.to_time, info.fps, high))
    if high <= low:
        logs.fail("the chosen time range is empty",
                  "check --from / --to and --half2-start")
    logs.kv("tagging half", args.half + (f"  ({why})" if why else ""))
    logs.kv("frames offered from", f"{logs.mmss(info.seconds_of(low))} to "
                                   f"{logs.mmss(info.seconds_of(high))}")
    if not why and not (args.from_time or args.to_time):
        logs.warn("nothing tells this script where half 2 starts, so frames "
                  "from the whole video may be offered",
                  "pass --half2-start MM:SS (or add half2_start to your config) "
                  "so the home zones all come from one half - mixing halves "
                  "makes them meaningless")
    pool = [int(v) for v in np.linspace(low, high, max(args.rounds * 4, 8))]
    logs.step_done()

    logs.step(4, 4, "Tagging")
    crops, zones, frame_size, saved = gallery_module.tag_interactive(
        cap, info, roster, args.rounds, candidate_frames=pool)
    cap.release()
    cv2.destroyAllWindows()
    logs.blank()
    gallery_module.save(cfg.gallery_dir, crops, roster, zones,
                        gallery_half=int(args.half), frame_size=frame_size)
    logs.step_done(f"{saved} complete round(s)")

    # Read it straight back and show what the home zones imply. A silent
    # gallery is a gallery you cannot trust.
    logs.section("CHECK")
    reloaded_crops, player_config, meta = gallery_module.load(cfg.gallery_dir)
    court = court_module.load(cfg.court_file)
    if meta["has_home_zones"]:
        plan = halves.build(meta["gallery_half"], str(args.half), None, info.fps)
        ends = plan.team_ends(player_config, court, int(args.half))
        for team, side in sorted(ends.items()):
            logs.kv(f"{team} home end", f"the {side} of the frame in half {args.half}")
        logs.hint("if those ends are swapped, you tagged in the other half - "
                  "re-run with the other --half")

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
