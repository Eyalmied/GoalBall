#!/usr/bin/env python3
"""
OPTIONAL - record who really threw each ball, so accuracy can be measured.

Predictions run fine without this. Do it if you want a number you can trust:
label a few dozen throws, then run scripts/evaluate.py.

Each throw plays on a loop with the ball's release point marked and the
reference gallery beside it. Press the thrower's number. Progress is saved
after every single label, so you can stop and come back whenever.

    python scripts/label_throwers.py --config config/mygame.json --half auto

    SPACE   play / pause
    a  d    step one frame back / forward
    1-6     this player threw it   (then it moves on)
    0  u    not sure  (left out of the accuracy figures)
    b  n    previous throw / next throw without labelling
    q       stop (everything so far is already saved)
"""

from __future__ import annotations

import pathlib

import cv2

from _common import add_half_flags, base_parser, resolve
from goalball import court as court_module
from goalball import events as events_module
from goalball import gallery as gallery_module
from goalball import halves, labels as labels_module, logs, video

DISPLAY_WIDTH = 1180


def ball_marker(throw, width, height, native_width, native_height):
    point = throw.release_point()
    if not point:
        return None
    return (int(point[0] * width / native_width),
            int(point[1] * height / native_height))


def main() -> None:
    parser = base_parser(__doc__)
    add_half_flags(parser)
    parser.add_argument("--limit", type=int, default=None,
                        help="only offer the first N throws")
    parser.add_argument("--skip-labelled", dest="skip_labelled",
                        action="store_true",
                        help="jump straight past throws you already labelled")
    args = parser.parse_args()
    cfg = resolve(args, "video", "events")

    logs.title("label the throwers", "ground truth, so accuracy can be measured")

    logs.step(1, 3, "Opening the video and the throw list")
    cap, info = video.open_video(cfg.video)
    info.describe()
    throws = events_module.load(cfg.events, info.fps)
    if args.limit:
        throws = throws[:args.limit]
    logs.step_done(f"{len(throws)} throws")

    logs.step(2, 3, "Loading the gallery so you can see who is who")
    crops, player_config, meta = gallery_module.load(cfg.gallery_dir)
    court = court_module.load(cfg.court_file)
    plan = halves.build(args.gallery_half or meta["gallery_half"], args.half,
                        args.half2_start or cfg.half2_start, info.fps,
                        mirror_about=args.mirror_about)
    plan.describe(player_config, court)
    player_count = len(player_config)
    sheet = gallery_module.contact_sheet(crops, player_config)
    logs.step_done()

    logs.step(3, 3, "Labelling")
    existing = labels_module.load(cfg.labels)
    rows = {t.throw_id: existing.get(t.throw_id, {}) for t in throws}
    order = [t.throw_id for t in throws]
    logs.info(f"  {sum(1 for r in rows.values() if labels_module.is_labelled(r))} "
              f"of {len(throws)} throws are already labelled")
    logs.info("  keys:  1-%d thrower   0/u not sure   a/d step   SPACE play   "
              "b back   n skip   q stop" % player_count)
    logs.blank()

    window = "Label the thrower"
    cv2.namedWindow(window)
    if sheet is not None:
        preview_h = 260
        cv2.imshow("Who is who",
                   cv2.resize(sheet, (sheet.shape[1] * preview_h // sheet.shape[0],
                                      preview_h)))

    index = 0
    while 0 <= index < len(throws):
        throw = throws[index]
        if args.skip_labelled and labels_module.is_labelled(rows[throw.throw_id]):
            index += 1
            continue
        half = plan.half_of_frame(throw.start_frame)
        ends = plan.team_ends(player_config, court, half)
        frames = video.read_span(cap, throw.start_frame, throw.end_frame)
        if not frames:
            logs.warn(f"{throw.throw_id}: no frames could be read, skipping")
            index += 1
            continue

        position, playing = 0, True
        while True:
            _, frame = frames[position]
            display = cv2.resize(
                frame, (DISPLAY_WIDTH,
                        int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])))
            height, width = display.shape[:2]
            marker = ball_marker(throw, width, height, info.width, info.height)
            if marker:
                cv2.circle(display, marker, 13, (0, 255, 255), 2)
                cv2.putText(display, "ball released here",
                            (marker[0] + 16, marker[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            current = rows[throw.throw_id].get("thrower_player_id", "")
            shown = current or ("not sure"
                                if rows[throw.throw_id].get("thrower_name") == "unsure"
                                else "-")
            cv2.rectangle(display, (0, 0), (width, 80), (0, 0, 0), -1)
            cv2.putText(display, f"[{index + 1}/{len(throws)}] {throw.throw_id}"
                                 f"   at {throw.start_mmss}   half {half}",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            if ends:
                cv2.putText(display, "this half:  " + ",   ".join(
                    f"{team} at the {side}" for team, side in sorted(ends.items())),
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1)
            cv2.putText(display, f"press 1-{player_count} for the thrower   "
                                 f"0/u not sure   a/d step   SPACE   b back   "
                                 f"n skip   q stop      current: {shown}",
                        (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
            cv2.imshow(window, display)

            key = cv2.waitKey(33 if playing else 0) & 0xFF
            if playing and key == 255:
                position = (position + 1) % len(frames)
                continue
            if key == ord(" "):
                playing = not playing
            elif key == ord("d"):
                playing, position = False, (position + 1) % len(frames)
            elif key == ord("a"):
                playing, position = False, (position - 1) % len(frames)
            elif key in (ord("u"), ord("0")):
                _record(rows, throw, None, player_config, half)
                labels_module.save(cfg.labels, rows, order)
                index += 1
                break
            elif ord("1") <= key <= ord(str(min(9, player_count))):
                _record(rows, throw, key - ord("0"), player_config, half)
                labels_module.save(cfg.labels, rows, order)
                logs.info(f"  {throw.throw_id} at {throw.start_mmss}  "
                          f"{logs.sym('arrow')} "
                          f"{player_config[key - ord('0')]['name']}")
                index += 1
                break
            elif key == ord("n"):
                index += 1
                break
            elif key == ord("b"):
                index = max(0, index - 1)
                break
            elif key == ord("q"):
                labels_module.save(cfg.labels, rows, order)
                _finish(cfg, rows, throws)
                cap.release()
                cv2.destroyAllWindows()
                return

    labels_module.save(cfg.labels, rows, order)
    cap.release()
    cv2.destroyAllWindows()
    _finish(cfg, rows, throws)


def _record(rows, throw, player_id, player_config, half) -> None:
    rows[throw.throw_id] = {
        "throw_id": throw.throw_id,
        "thrower_player_id": player_id if player_id else "",
        "thrower_name": (player_config.get(player_id, {}).get("name", "")
                         if player_id else "unsure"),
        "half": half,
        "start_time": throw.start_mmss,
        "csv_team_hint": throw.csv_team_hint,
    }


def _finish(cfg, rows, throws) -> None:
    logs.blank()
    logs.step_done()
    labelled = [r for r in rows.values() if labels_module.is_labelled(r)]
    unsure = [r for r in rows.values() if r.get("thrower_name") == "unsure"]
    logs.section("SUMMARY")
    logs.kv("labelled", f"{len(labelled)} of {len(throws)} throws")
    logs.kv("marked not sure", str(len(unsure)))
    per_half: dict = {}
    for row in labelled:
        key = row.get("half", "?")
        per_half[key] = per_half.get(key, 0) + 1
    if len(per_half) > 1:
        logs.kv("per half", "   ".join(f"half {h}: {n}"
                                       for h, n in sorted(per_half.items(),
                                                          key=lambda kv: str(kv[0]))))
    if len(labelled) < 30:
        logs.hint("about 30+ labelled throws makes the accuracy figures "
                  "meaningful; below that they swing a lot")
    logs.closing([("labels", cfg.labels)])
    logs.next_steps([
        "measure the accuracy:  python scripts/evaluate.py --config ... --holdout 0.3",
        "see the mistakes:  python scripts/review_predictions.py --config ... --only-wrong",
    ])


if __name__ == "__main__":
    main()
