#!/usr/bin/env python3
"""
STEP 4 of 4 - measure the accuracy against the labels you recorded.

    python scripts/evaluate.py --config config/mygame.json --holdout 0.3

The three numbers that matter:

    CORRECT THROWER      right team AND right player. The headline.
    correct team         solved by posture; expect around 89%.
    right player given
    the right team       the hard part, and the real ceiling (~81% at best).

Because the whole is the product of the last two, and both are capped, roughly
72% correct thrower is the ceiling with a good ball release point. See
docs/RESULTS.md before chasing a higher number.

WHY --holdout
    Tuning against a number and then quoting that same number measures
    nothing. --holdout 0.3 splits the throws the same way every run: tune
    against the "tune" block, and only read the "held out" block at the end.
"""

from __future__ import annotations

import json
import pathlib

from _common import base_parser, resolve
from goalball import evaluate as evaluate_module
from goalball import gallery as gallery_module
from goalball import labels as labels_module
from goalball import logs


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--holdout", type=float, default=0.0,
                        help="fraction of throws to hold back and report "
                             "separately, e.g. 0.3 (default: 0, use everything)")
    parser.add_argument("--seed", type=int, default=0,
                        help="which split to use; the same seed always gives "
                             "the same split (default: 0)")
    parser.add_argument("--team-size", dest="team_size", type=int, default=3,
                        help="players per team, for the guessing baseline "
                             "(default: 3)")
    args = parser.parse_args()
    cfg = resolve(args)

    logs.title("measure the accuracy", "predictions against your labels")

    logs.step(1, 3, "Loading the predictions and the labels")
    predictions_path = pathlib.Path(cfg.predictions)
    if not predictions_path.exists():
        logs.fail(f"no predictions at {cfg.predictions}",
                  "run scripts/predict_throwers.py first")
    try:
        raw = json.loads(predictions_path.read_text())
    except Exception as e:
        logs.fail(f"{cfg.predictions} could not be read ({e})")
    # "throw_id" is the current key; "event_id" is what the earlier scripts wrote
    predictions = {}
    for record in raw:
        key = record.get("throw_id") or record.get("event_id")
        if key:
            row = dict(record)
            row.setdefault("thrower_id", record.get("predicted_thrower_id"))
            row.setdefault("thrower_name", record.get("predicted_thrower_name"))
            row.setdefault("thrower_team", record.get("predicted_team"))
            predictions[key] = row
    logs.ok(f"{len(predictions)} predictions read from {cfg.predictions}")

    label_rows = labels_module.load(cfg.labels)
    if not label_rows:
        logs.fail(f"no labels at {cfg.labels}",
                  "run scripts/label_throwers.py to record who really threw; "
                  "without labels there is nothing to measure against")
    ground_truth = {k: v for k, v in label_rows.items()
                    if labels_module.is_labelled(v)}
    if not ground_truth:
        logs.fail("every label is marked 'not sure', so nothing can be measured",
                  "re-run scripts/label_throwers.py and label some throws")
    logs.step_done()

    logs.step(2, 3, "Matching them up")
    crops, player_config, meta = gallery_module.load(cfg.gallery_dir)
    items, missing = [], []
    for throw_id, label in ground_truth.items():
        if throw_id in predictions:
            items.append((throw_id, label, predictions[throw_id]))
        else:
            missing.append(throw_id)
    logs.kv("labelled throws", str(len(ground_truth)))
    logs.kv("with a prediction", str(len(items)))
    if missing:
        logs.warn(f"{len(missing)} labelled throw(s) have no prediction and are "
                  f"left out",
                  "finish the prediction run (it supports --resume) so the two "
                  "files cover the same throws")
    if not items:
        logs.fail("no throw has both a label and a prediction",
                  "check that both files are for the same game")
    logs.step_done()

    logs.step(3, 3, "Working out the numbers")
    if args.holdout > 0:
        groups: dict = {"tune": [], "held out": []}
        for item in items:
            groups[evaluate_module.bucket_of(item[0], args.holdout, args.seed)]\
                .append(item)
        logs.kv("split", f"{len(groups['tune'])} to tune against, "
                         f"{len(groups['held out'])} held out "
                         f"(holdout={args.holdout}, seed={args.seed})")
        evaluate_module.report("TUNE BLOCK - do not quote this number",
                               groups["tune"], player_config, args.team_size)
        evaluate_module.report("HELD OUT - this is the number that counts",
                               groups["held out"], player_config, args.team_size)
    else:
        evaluate_module.report("ALL LABELLED THROWS", items, player_config,
                               args.team_size)
        logs.blank()
        logs.hint("add --holdout 0.3 to see a held-out number, which is the "
                  "honest one if you have been changing settings")
    logs.blank()
    logs.next_steps([
        "see exactly what went wrong:  python scripts/review_predictions.py "
        "--config ... --only-wrong",
        "read docs/RESULTS.md for what each number can realistically reach",
    ])


if __name__ == "__main__":
    main()
