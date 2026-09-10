#!/usr/bin/env python3
"""
STEP 2 of 4 - mark the court, once per camera angle.

Click the four corners of the playing surface. From then on, anyone whose feet
are outside that shape is not a player - crowd, referees, coaches and
substitutes are all removed before the pipeline reasons about anything.

The same four corners also define the flat "court plane" used to flip home
zones end-for-end at half-time, so this step is what makes --half exact rather
than approximate.

    python scripts/set_court.py --config config/mygame.json
    python scripts/set_court.py --video game.mp4 --at 07:30
"""

from __future__ import annotations

import cv2

from _common import add_half_flags, base_parser, parse_timestamp, resolve
from goalball import court as court_module
from goalball import gallery as gallery_module
from goalball import halves, logs, video


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--at", default=None,
                        help="which moment to click on: 12:34, 400s, or a frame "
                             "number (default: about a fifth into the video, "
                             "where play is usually under way)")
    parser.add_argument("--replace", action="store_true",
                        help="overwrite an existing court file without asking")
    add_half_flags(parser)
    args = parser.parse_args()
    cfg = resolve(args, "video")

    logs.title("mark the court", "one-off setup, takes about ten seconds")

    logs.step(1, 3, "Opening the video")
    cap, info = video.open_video(cfg.video)
    info.describe()
    logs.step_done()

    logs.step(2, 3, "Choosing a frame to click on")
    frame_index = parse_timestamp(args.at, info.fps,
                                  default=int(info.frame_count * 0.2))
    frame = video.read_frame(cap, frame_index)
    if frame is None:
        logs.fail(f"could not read frame {frame_index} from the video",
                  "pick a different moment with --at, e.g. --at 05:00")
    logs.ok(f"showing {logs.mmss(info.seconds_of(frame_index))} "
            f"(frame {frame_index})")
    logs.hint("pick a moment where the whole court is visible and play has "
              "started; use --at to choose another")
    logs.step_done()

    import pathlib
    if pathlib.Path(cfg.court_file).exists() and not args.replace:
        logs.warn(f"{cfg.court_file} already exists - it will be replaced if "
                  f"you confirm new corners")

    logs.step(3, 3, "Clicking the corners")
    points = court_module.pick_corners(frame)
    if points is None:
        logs.info("no court file was written")
        cap.release()
        return
    height, width = frame.shape[:2]
    court_module.save(cfg.court_file, points, width, height)
    logs.step_done()

    # Show what the court now implies, so a bad click is obvious immediately.
    court = court_module.load(cfg.court_file)
    court.describe()
    try:
        crops, player_config, meta = gallery_module.load(cfg.gallery_dir)
    except SystemExit:
        player_config, meta = {}, None
    if meta and meta["has_home_zones"]:
        plan = halves.build(args.gallery_half or meta["gallery_half"],
                            args.half, args.half2_start, info.fps,
                            mirror_about=args.mirror_about)
        logs.section("WHAT THIS MEANS FOR THE TEAMS")
        plan.describe(player_config, court)
        logs.hint("if those ends look swapped, either the court corners or the "
                  "--half setting is wrong")

    cap.release()
    cv2.destroyAllWindows()
    logs.closing([("court corners", cfg.court_file)])
    logs.next_steps([
        "build the reference gallery:  python scripts/build_gallery.py --config ...",
        "or, if the gallery already exists, run the prediction:  "
        "python scripts/predict_throwers.py --config ...",
    ])


if __name__ == "__main__":
    main()
