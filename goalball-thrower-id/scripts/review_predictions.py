#!/usr/bin/env python3
"""
Look at the results - the fastest way to understand a mistake.

Each throw plays with two reference panels beside it: on the left the player the
pipeline chose, on the right the player you labelled. Green means right, red
means wrong. One glance tells you which half failed - wrong team, or the wrong
teammate.

    python scripts/review_predictions.py --config config/mygame.json
    python scripts/review_predictions.py --config config/mygame.json --only-wrong
    python scripts/review_predictions.py --config config/mygame.json --export out/

    SPACE   play / pause          n  b   next / previous throw
    a  d    step one frame        q      quit
"""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np

from _common import base_parser, resolve
from goalball import events as events_module
from goalball import gallery as gallery_module
from goalball import labels as labels_module
from goalball import logs, video

MAIN_WIDTH = 900
PANEL_WIDTH = 250
GREEN, RED, GREY = (0, 200, 0), (0, 0, 255), (120, 120, 120)


def player_panel(player_id, crops, player_config, width, height, heading,
                 border, footnote=""):
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.putText(canvas, heading, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    if player_id and int(player_id) in crops and crops[int(player_id)]:
        images = crops[int(player_id)][:2]
        cell = (height - 90) // len(images)
        y = 36
        for image in images:
            fitted = gallery_module.letterbox(image, width - 16, cell - 6)
            canvas[y:y + fitted.shape[0], 8:8 + fitted.shape[1]] = fitted
            y += cell
        info = player_config[int(player_id)]
        caption = f"{player_id}: {info['name']} ({info['team']})"
    else:
        cv2.putText(canvas, "nobody", (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREY, 2)
        caption = "no player chosen"
    cv2.putText(canvas, caption, (10, height - 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    if footnote:
        cv2.putText(canvas, footnote, (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), border, 4)
    return canvas


def compose(frame, throw_id, prediction, label, crops, player_config,
            index, total):
    prediction = prediction or {}
    predicted_id = prediction.get("thrower_id")
    truth_id = labels_module.player_id(label or {})
    correct = (predicted_id is not None and truth_id is not None
               and int(predicted_id) == int(truth_id))
    colour = GREEN if correct else (GREY if truth_id is None else RED)

    main = cv2.resize(frame, (MAIN_WIDTH,
                              int(frame.shape[0] * MAIN_WIDTH / frame.shape[1])))
    height = main.shape[0]

    banner = np.zeros((76, main.shape[1] + 2 * PANEL_WIDTH, 3), dtype=np.uint8)
    verdict = "CORRECT" if correct else ("NOT LABELLED" if truth_id is None
                                         else "WRONG")
    cv2.putText(banner, f"[{index + 1}/{total}] {throw_id}   at "
                        f"{prediction.get('start_time', '?')}   "
                        f"half {prediction.get('half', '?')}   {verdict}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
    confidence = prediction.get("confidence", "?")
    ball = "ball seen" if prediction.get("ball_seen") else "ball not seen"
    cv2.putText(banner, f"chose {prediction.get('thrower_name')} "
                        f"(confidence {confidence}, {ball}, by "
                        f"{prediction.get('decided_by', '?')})    "
                        f"you said {(label or {}).get('thrower_name', '-')}",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    left = player_panel(predicted_id, crops, player_config, PANEL_WIDTH, height,
                        "PIPELINE CHOSE", colour, f"confidence {confidence}")
    right = player_panel(truth_id, crops, player_config, PANEL_WIDTH, height,
                         "YOU LABELLED", GREEN if truth_id else GREY)
    cv2.rectangle(main, (0, 0), (main.shape[1] - 1, height - 1), colour, 3)
    return cv2.vconcat([banner, cv2.hconcat([left, main, right])]), correct


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--only-wrong", dest="only_wrong", action="store_true",
                        help="show only the throws the pipeline got wrong")
    parser.add_argument("--only-uncertain", dest="only_uncertain", type=float,
                        default=None,
                        help="show only throws below this confidence, e.g. 0.4")
    parser.add_argument("--include-unlabelled", dest="include_unlabelled",
                        action="store_true",
                        help="also show throws you never labelled")
    parser.add_argument("--export", default=None,
                        help="write one image per throw into this folder "
                             "instead of opening a window")
    args = parser.parse_args()
    cfg = resolve(args, "video", "events")

    logs.title("review the predictions", "predicted next to the truth")

    logs.step(1, 2, "Loading everything")
    path = pathlib.Path(cfg.predictions)
    if not path.exists():
        logs.fail(f"no predictions at {cfg.predictions}",
                  "run scripts/predict_throwers.py first")
    predictions = {}
    for record in json.loads(path.read_text()):
        key = record.get("throw_id") or record.get("event_id")
        row = dict(record)
        row.setdefault("thrower_id", record.get("predicted_thrower_id"))
        row.setdefault("thrower_name", record.get("predicted_thrower_name"))
        predictions[key] = row
    logs.ok(f"{len(predictions)} predictions loaded")
    label_rows = labels_module.load(cfg.labels)
    crops, player_config, meta = gallery_module.load(cfg.gallery_dir)
    cap, info = video.open_video(cfg.video)
    throws = {t.throw_id: t for t in events_module.load(cfg.events, info.fps)}
    logs.step_done()

    logs.step(2, 2, "Choosing which throws to show")

    def wanted(throw_id):
        label = label_rows.get(throw_id)
        labelled = labels_module.is_labelled(label or {})
        if not labelled and not args.include_unlabelled:
            return False
        prediction = predictions.get(throw_id, {})
        if args.only_wrong:
            if not labelled:
                return False
            if str(prediction.get("thrower_id")) == \
                    str(label.get("thrower_player_id")):
                return False
        if args.only_uncertain is not None:
            if float(prediction.get("confidence") or 0) >= args.only_uncertain:
                return False
        return True

    selection = [tid for tid in throws if tid in predictions and wanted(tid)]
    if not selection:
        logs.fail("no throw matches those filters",
                  "drop --only-wrong / --only-uncertain, or add "
                  "--include-unlabelled to see throws you have not labelled")
    logs.ok(f"{len(selection)} throws to review"
            + (" (only the wrong ones)" if args.only_wrong else ""))
    logs.step_done()

    if args.export:
        folder = pathlib.Path(args.export)
        folder.mkdir(parents=True, exist_ok=True)
        right = 0
        for position, throw_id in enumerate(selection):
            throw = throws[throw_id]
            middle = (throw.start_frame + throw.end_frame) // 2
            frame = video.read_frame(cap, middle)
            if frame is None:
                continue
            image, correct = compose(frame, throw_id, predictions.get(throw_id),
                                     label_rows.get(throw_id), crops,
                                     player_config, position, len(selection))
            right += int(correct)
            tag = "correct" if correct else "WRONG"
            cv2.imwrite(str(folder / f"{throw_id}_{tag}.jpg"), image)
            logs.progress(position + 1, len(selection), throw_id)
        cap.release()
        logs.ok(f"{len(selection)} images written, {right} of them correct")
        logs.closing([("review images", f"{folder}/")])
        return

    logs.info("  SPACE play/pause   a/d step   n/b next/previous   q quit")
    logs.blank()
    window = "Review"
    cv2.namedWindow(window)
    index = 0
    while 0 <= index < len(selection):
        throw = throws[selection[index]]
        frames = video.read_span(cap, throw.start_frame, throw.end_frame)
        if not frames:
            index += 1
            continue
        position, playing = 0, True
        while True:
            image, _ = compose(frames[position][1], throw.throw_id,
                               predictions.get(throw.throw_id),
                               label_rows.get(throw.throw_id), crops,
                               player_config, index, len(selection))
            cv2.imshow(window, image)
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
            elif key == ord("n"):
                index += 1
                break
            elif key == ord("b"):
                index = max(0, index - 1)
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                logs.blank()
                return

    cap.release()
    cv2.destroyAllWindows()
    logs.blank()


if __name__ == "__main__":
    main()
