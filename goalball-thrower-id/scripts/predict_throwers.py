#!/usr/bin/env python3
"""
STEP 3 of 4 - predict who threw each ball.

Reads the throw timestamps, watches the video around each one, and writes a
named thrower plus a confidence for every throw.

    # best accuracy (uses the CSV's ball release point for POSITION only)
    python scripts/predict_throwers.py --config config/mygame.json --half 1

    # a full-match video that changes ends part-way through
    python scripts/predict_throwers.py --config config/mygame.json \
           --half auto --half2-start 22:00

    # strictly video-only: nothing from the CSV but the timestamps
    python scripts/predict_throwers.py --config config/mygame.json --half 1 \
           --release-source thrower

Stop it at any time; --resume picks up where it left off, and the output file
is rewritten after every throw.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

from _common import add_half_flags, base_parser, resolve, resolve_halves
from goalball import court as court_module
from goalball import gallery as gallery_module
from goalball import detect, halves, logs, predict, video

TOTAL_STEPS = 6


def write_atomically(path: str, records: list) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w") as f:
        json.dump(records, f, indent=2, default=str)
    os.replace(temporary, path)


def main() -> None:
    parser = base_parser(__doc__)
    add_half_flags(parser)

    tuning = parser.add_argument_group(
        "how the thrower is chosen",
        "The defaults are the values measured to work best; change them only "
        "with a way to measure the effect (scripts/evaluate.py).")
    tuning.add_argument("--appearance", default="osnet",
                        choices=["osnet", "dinov2"],
                        help="appearance model: osnet is stronger, dinov2 "
                             "installs more easily (default: osnet)")
    tuning.add_argument("--release-source", dest="release_source",
                        default="csv", choices=["csv", "thrower"],
                        help="where the thrower's POSITION comes from. "
                             "csv = the upstream ball release point, most "
                             "accurate (~74%% correct thrower). thrower = the "
                             "video-localised player, strictly video-only "
                             "(~50%%). (default: csv)")
    tuning.add_argument("--frames", dest="frames_per_throw", type=int, default=5,
                        help="frames examined and voted per throw (default: 5)")
    tuning.add_argument("--window-fraction", dest="window_fraction",
                        type=float, default=0.6,
                        help="how much of each throw window to look at, from "
                             "its start (default: 0.6)")
    tuning.add_argument("--processing-width", dest="processing_width",
                        type=int, default=1280,
                        help="frames are scaled to this width before detection "
                             "(default: 1280)")
    tuning.add_argument("--position-weight", dest="weight_position",
                        type=float, default=0.75,
                        help="how much court position counts against "
                             "appearance when naming the player. 0.75 is the "
                             "measured best; below about 0.65 appearance "
                             "starts doing more harm than good (default: 0.75)")
    tuning.add_argument("--position-spread", dest="position_spread",
                        type=float, default=0.12,
                        help="how far a player may stray from their home zone "
                             "and still be recognised, as a fraction of the "
                             "frame (default: 0.12)")
    tuning.add_argument("--thrower-model", dest="thrower_model", default=None,
                        help="a goalball-trained detector whose class 0 is "
                             "'throwing_player'. This is the default way the "
                             "thrower is picked (69%% correct against 46%% for "
                             "the motion rule, which covers the throws it "
                             "misses). Defaults to the project's "
                             "Model Weights/best.pt; pass 'none' to turn it "
                             "off")
    tuning.add_argument("--motion-weight", dest="weight_motion", type=float,
                        default=0.4,
                        help="how much 'this player is moving' counts when "
                             "picking the thrower. The thrower runs up and "
                             "swings; defenders are already down and "
                             "team-mates stand still (default: 0.4)")
    tuning.add_argument("--upright-weight", dest="weight_upright", type=float,
                        default=0.6,
                        help="how much 'this player is on their feet' counts "
                             "(default: 0.6). Anything from 0.4 to 0.7 scored "
                             "the same on the reference match")
    tuning.add_argument("--ball-weight", dest="weight_ball", type=float,
                        default=0.0,
                        help="how much nearness to the detected ball counts "
                             "(default: 0, off). The ball is detected most "
                             "confidently at the BLOCKER's end, so this used "
                             "to point at the wrong team")
    tuning.add_argument("--pose-weight", dest="weight_pose", type=float,
                        default=0.0,
                        help="how much arm extension and stance width count "
                             "(default: 0, off - measured to make things worse)")
    tuning.add_argument("--localise-among", dest="localise_among",
                        default="everyone", choices=["everyone", "attackers"],
                        help="who can be the thrower: everyone on court "
                             "(default), or only the posture-derived attacking "
                             "team. Restricting it was measured to lose "
                             "accuracy, because a wrong team removes the right "
                             "answer")
    tuning.add_argument("--team-source", dest="team_source",
                        default="position", choices=["position", "posture"],
                        help="where the team in the answer comes from. "
                             "position = name the thrower against every "
                             "player's home zone and report the winner's own "
                             "team, which needs no team step and was measured "
                             "best; posture = decide the team first and only "
                             "consider its three players, which is required "
                             "when the gallery has no home zones "
                             "(default: position)")
    tuning.add_argument("--team-grouping", dest="team_grouping",
                        default="relative", choices=["relative", "absolute"],
                        help="how the players are split into two teams by kit "
                             "colour. relative = rank them within each frame, "
                             "which survives the change of ends; absolute = "
                             "match each crop on its own, which was measured to "
                             "invert in the half the gallery was not tagged in "
                             "(default: relative)")
    tuning.add_argument("--no-ball", dest="use_ball", action="store_false",
                        default=True,
                        help="never look for the ball, even if --ball-weight "
                             "is set")
    tuning.add_argument("--no-flip-augment", dest="flip_augment",
                        action="store_false", default=True,
                        help="skip mirrored copies when embedding crops (faster, "
                             "slightly worse)")
    tuning.add_argument("--pose-model", dest="pose_model", default=None,
                        help="YOLO pose model file (default: yolov8m-pose.pt)")
    tuning.add_argument("--ball-model", dest="ball_model", default=None,
                        help="YOLO detection model file (default: yolov8m.pt)")

    run = parser.add_argument_group("run control")
    run.add_argument("--limit", type=int, default=None,
                     help="only do the first N throws (handy for a quick check)")
    run.add_argument("--resume", action="store_true",
                     help="skip throws already present in the output file")
    args = parser.parse_args()
    cfg = resolve(args, "video", "events")

    logs.title("predict the throwers",
               "one named player and one confidence per throw")

    # ---- 1. video -----------------------------------------------------
    logs.step(1, TOTAL_STEPS, "Opening the video")
    cap, info = video.open_video(cfg.video)
    info.describe()
    logs.step_done()

    # ---- 2. throws ----------------------------------------------------
    logs.step(2, TOTAL_STEPS, "Reading the throw list")
    from goalball import events as events_module
    throws = events_module.load(cfg.events, info.fps)
    if args.limit:
        throws = throws[:args.limit]
        logs.info(f"  limited to the first {len(throws)} throws (--limit)")
    logs.step_done(f"{len(throws)} throws")

    # ---- 3. gallery + court -------------------------------------------
    logs.step(3, TOTAL_STEPS, "Loading the reference gallery and the court")
    crops, player_config, meta = gallery_module.load(cfg.gallery_dir)
    court = court_module.load(cfg.court_file)
    court.describe()
    logs.step_done()

    # ---- 4. the half --------------------------------------------------
    logs.step(4, TOTAL_STEPS, "Working out which half is being analysed")
    gallery_half = args.gallery_half or meta["gallery_half"]
    plan = resolve_halves(args, cfg, info.fps, gallery_half)
    plan.describe(player_config, court)
    per_half = halves.counts_by_half(plan, throws)
    logs.kv("throws in each half", "   ".join(f"half {h}: {n}"
                                              for h, n in sorted(per_half.items())))
    if not meta["has_home_zones"] and args.team_source == "position":
        logs.info("  no home zones, so the team has to come from posture "
                  "instead of position")
        args.team_source = "posture"
    if not meta["has_home_zones"]:
        logs.warn("without home zones the --half setting has nothing to mirror, "
                  "so teammates will be separated by appearance alone "
                  "(measured at ~44%, against ~81% for position)",
                  "rebuild the gallery: python scripts/build_gallery.py "
                  "--config ... --half 1")
    elif plan.mode != "auto" and len(per_half) == 1 and \
            info.duration_s > 40 * 60 and not plan.half2_start_text:
        logs.warn(f"the video is {logs.human_duration(info.duration_s)} long, "
                  f"which usually means it holds both halves, but every throw "
                  f"is being treated as half {plan.mode}",
                  "if it really covers both, add half2_start to your config - "
                  "every step then splits the match the same way")
    logs.step_done()

    # ---- 5. models ----------------------------------------------------
    logs.step(5, TOTAL_STEPS, "Loading the models")
    settings = predict.Settings(
        frames_per_throw=args.frames_per_throw,
        window_fraction=args.window_fraction,
        processing_width=args.processing_width,
        use_ball=args.use_ball and args.weight_ball > 0,
        weight_motion=args.weight_motion,
        weight_upright=args.weight_upright,
        weight_ball=args.weight_ball,
        weight_pose=args.weight_pose,
        localise_among=args.localise_among,
        # "none" turns it off; an unset flag falls through to the config.
        thrower_model=(None
                       if (args.thrower_model or "").lower() == "none"
                       else (args.thrower_model or cfg.thrower_model)),
        team_grouping=args.team_grouping,
        team_source=args.team_source,
        release_source=args.release_source,
        weight_position=args.weight_position,
        position_spread=args.position_spread,
        flip_augment=args.flip_augment,
        pose_model=args.pose_model or cfg.pose_model,
        ball_model=args.ball_model or cfg.ball_model,
        appearance_model=args.appearance,
    )
    if settings.release_source == "csv" and \
            not any(t.release_point() for t in throws):
        logs.warn("--release-source csv was asked for, but no throw in the CSV "
                  "has a ball release point",
                  "the run will fall back to video-localised positions, which "
                  "is what --release-source thrower does")
    if settings.thrower_model and not detect.has_model(settings.thrower_model):
        logs.warn(f"no goalball detector at {settings.thrower_model}",
                  "the thrower will be picked by movement and posture "
                  "instead, which is measured at 46% against 69% - point "
                  "--thrower-model at the project's Model Weights/best.pt to "
                  "get the better one")
        settings.thrower_model = None
    detect.warm_up(settings.pose_model,
                   settings.ball_model if settings.use_ball else None)
    predictor = predict.Predictor(cap, info, court, plan, player_config,
                                  crops, settings,
                                  crops_by_half=meta.get("crops_by_half"))
    logs.blank()
    settings.describe()
    logs.step_done()

    # ---- 6. the run ---------------------------------------------------
    logs.step(6, TOTAL_STEPS, f"Working through {len(throws)} throws")
    records, already_done = [], set()
    if args.resume and pathlib.Path(cfg.predictions).exists():
        try:
            records = json.loads(pathlib.Path(cfg.predictions).read_text())
            already_done = {r.get("throw_id") for r in records}
            logs.ok(f"resuming: {len(already_done)} throws are already done")
        except Exception:
            logs.warn(f"{cfg.predictions} could not be read, starting fresh")
            records = []
    elif pathlib.Path(cfg.predictions).exists():
        logs.info(f"  {cfg.predictions} will be overwritten "
                  f"(pass --resume to add to it instead)")

    logs.blank()
    started = time.time()
    todo = [t for t in throws if t.throw_id not in already_done]
    for position, throw in enumerate(todo, start=1):
        record = predictor.predict(throw)
        records.append(record)
        predict.log_one(record, position, len(todo))
        write_atomically(cfg.predictions, records)
    elapsed = time.time() - started

    cap.release()
    logs.blank()
    logs.step_done(f"{len(todo)} throws in {logs.human_duration(elapsed)}"
                   + (f", {elapsed / len(todo):.1f} s each" if todo else ""))

    predict.log_summary(predictor.totals, cfg.predictions)
    logs.closing([("predictions", cfg.predictions)])

    steps = []
    if pathlib.Path(cfg.labels).exists():
        steps.append("measure the accuracy:  python scripts/evaluate.py "
                     "--config ... --holdout 0.3")
    else:
        steps.append("to measure accuracy, label some throws first:  "
                     "python scripts/label_throwers.py --config ...")
    steps.append("look at the results:  python scripts/review_predictions.py "
                 "--config ...")
    steps.append("review the least certain calls first - sort by confidence")
    logs.next_steps(steps)


if __name__ == "__main__":
    main()
