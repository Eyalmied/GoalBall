#!/usr/bin/env python3
"""
Run the whole thing, start to finish.

    python scripts/run_pipeline.py --config config/mygame.json --half 1

This is the one command to remember. It checks what is already done, does what
is missing, and stops with a plain explanation if it needs you:

    1. gallery       tag the players once (opens a window; skipped if it exists)
    2. court         click the four corners once (opens a window; skipped if set)
    3. predict       name the thrower for every throw           <- the real work
    4. evaluate      accuracy, but only if you have labelled some throws

Nothing is redone unless you ask:

    --redo gallery,court     re-run those setup steps from scratch
    --resume                 continue an interrupted prediction run
    --skip-setup             fail instead of opening a window (for servers/CI)
    --dry-run                say what it would do, and do nothing

THE HALF IS NOT OPTIONAL
    Teams change ends at half-time, which moves every player's home zone to the
    opposite end. Tell the pipeline which half it is looking at:

        --half 1            the video is the half the gallery was tagged in
        --half 2            the video is the other half
        --half auto --half2-start 22:00
                            the video covers both; ends change at 22:00

    Getting this wrong does not make the result slightly worse - it inverts the
    identity of every player. See docs/HALVES.md.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from _common import add_half_flags, base_parser, resolve
from goalball import gallery as gallery_module
from goalball import logs

HERE = pathlib.Path(__file__).resolve().parent
STAGES = ("gallery", "court", "predict", "evaluate")


def shell(command: list, dry_run: bool) -> int:
    """Run one stage as its own process, so a crash cannot poison the rest."""
    printable = " ".join(
        f'"{c}"' if " " in str(c) else str(c) for c in command[1:])
    logs.info(f"  {logs.dim('running:')} python {printable}")
    if dry_run:
        return 0
    logs.blank()
    return subprocess.call([sys.executable] + command[1:])


def passthrough(args) -> list:
    """The flags every stage should inherit from this run."""
    flags = []
    if args.config:
        flags += ["--config", args.config]
    for name in ("video", "events", "gallery_dir", "court_file", "labels",
                 "predictions"):
        value = getattr(args, name, None)
        if value:
            flags += ["--" + name.replace("_", "-"), value]
    if args.quiet:
        flags.append("--quiet")
    if args.verbose:
        flags.append("--verbose")
    return flags


def main() -> None:
    parser = base_parser(__doc__)
    add_half_flags(parser)
    parser.add_argument("--rounds", type=int, default=5,
                        help="reference images per player, if the gallery has "
                             "to be built (default: 5)")
    parser.add_argument("--appearance", default="osnet",
                        choices=["osnet", "dinov2"],
                        help="appearance model (default: osnet)")
    parser.add_argument("--release-source", dest="release_source",
                        default="csv", choices=["csv", "thrower"],
                        help="csv = use the upstream ball release point for the "
                             "thrower's position, most accurate; thrower = "
                             "strictly video-only (default: csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only predict the first N throws (a quick check)")
    parser.add_argument("--holdout", type=float, default=0.3,
                        help="held-out fraction for the accuracy report "
                             "(default: 0.3)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted prediction run")
    parser.add_argument("--redo", default="",
                        help="comma-separated stages to force: "
                             "gallery, court, predict")
    parser.add_argument("--skip-setup", dest="skip_setup", action="store_true",
                        help="never open an interactive window; stop with an "
                             "explanation instead")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="print the plan and change nothing")
    parser.add_argument("--stop-after", dest="stop_after", default=None,
                        choices=list(STAGES),
                        help="finish after this stage")
    args = parser.parse_args()
    cfg = resolve(args, "video", "events")

    redo = {s.strip() for s in args.redo.split(",") if s.strip()}
    unknown = redo - set(STAGES)
    if unknown:
        logs.fail(f"--redo does not know the stage(s): {', '.join(sorted(unknown))}",
                  f"choose from: {', '.join(STAGES)}")

    logs.title("run the whole pipeline", cfg.name)
    cfg.describe()
    logs.kv("half", args.half + (f", ends change at {args.half2_start or cfg.half2_start}"
                                 if args.half == "auto" else ""))
    if args.dry_run:
        logs.blank()
        logs.info(logs.yellow("  dry run - nothing will be changed"))

    # ---- work out the plan --------------------------------------------
    gallery_manifest = pathlib.Path(cfg.gallery_dir) / gallery_module.MANIFEST_NAME
    have_gallery = gallery_manifest.exists()
    have_court = pathlib.Path(cfg.court_file).exists()
    have_predictions = pathlib.Path(cfg.predictions).exists()
    have_labels = pathlib.Path(cfg.labels).exists()

    plan = []
    plan.append(("gallery", "build the reference gallery",
                 "gallery" in redo or not have_gallery))
    plan.append(("court", "mark the court",
                 "court" in redo or not have_court))
    plan.append(("predict", "predict the throwers",
                 "predict" in redo or not have_predictions or args.resume))
    plan.append(("evaluate", "measure the accuracy", have_labels))

    logs.section("PLAN")
    for name, description, will_run in plan:
        if will_run:
            state = logs.green("will run")
        elif name == "evaluate":
            state = logs.dim("skipped - no labels file, so there is nothing "
                             "to measure against")
        else:
            state = logs.dim("already done - use --redo %s to repeat it" % name)
        logs.kv(description, state, width=30)
    if args.stop_after:
        logs.kv("stopping after", args.stop_after, width=30)
    logs.blank()

    inherited = passthrough(args)
    failures = []

    for name, description, will_run in plan:
        if not will_run:
            if args.stop_after == name:
                break
            continue

        logs.rule()
        logs.info(bold_stage(name, description))
        logs.rule()

        if name in ("gallery", "court") and args.skip_setup:
            logs.fail(f"{description} still needs doing, but --skip-setup was "
                      f"given and it needs a window",
                      "run it once on a machine with a display, then copy "
                      f"{cfg.gallery_dir}/ and {cfg.court_file} across")

        if name == "gallery":
            command = ["python", str(HERE / "build_gallery.py")] + inherited + [
                "--rounds", str(args.rounds),
                "--half", "1" if args.half == "auto" else args.half,
            ]
            if args.half2_start or cfg.half2_start:
                command += ["--half2-start", args.half2_start or cfg.half2_start]
        elif name == "court":
            command = ["python", str(HERE / "set_court.py")] + inherited + [
                "--half", args.half]
            if args.half2_start:
                command += ["--half2-start", args.half2_start]
        elif name == "predict":
            command = ["python", str(HERE / "predict_throwers.py")] + inherited + [
                "--half", args.half,
                "--appearance", args.appearance,
                "--release-source", args.release_source,
            ]
            if args.half2_start:
                command += ["--half2-start", args.half2_start]
            if args.gallery_half:
                command += ["--gallery-half", str(args.gallery_half)]
            command += ["--mirror-about", args.mirror_about]
            if args.limit:
                command += ["--limit", str(args.limit)]
            if args.resume:
                command.append("--resume")
        else:
            command = ["python", str(HERE / "evaluate.py")] + inherited + [
                "--holdout", str(args.holdout)]

        code = shell(command, args.dry_run)
        if code != 0:
            failures.append((description, code))
            logs.blank()
            logs.error(f"'{description}' stopped with exit code {code}")
            if name in ("gallery", "court", "predict"):
                logs.info("      the stages after this one need it, so the run "
                          "stops here")
                break
            continue
        if args.stop_after == name:
            logs.blank()
            logs.info(f"  stopping after '{description}' as asked")
            break

    logs.blank()
    logs.rule()
    if failures:
        logs.section("FINISHED WITH PROBLEMS")
        for description, code in failures:
            logs.bullet(f"{description} (exit code {code})")
        logs.info("")
        logs.info("  Scroll up to that stage's own output - it says what went "
                  "wrong and what to do.")
        sys.exit(1)

    logs.section("DONE")
    logs.kv("predictions", cfg.predictions)
    if have_labels:
        logs.kv("labels used", cfg.labels)
    else:
        logs.kv("accuracy", "not measured (no labels yet)")
    logs.next_steps([
        "look through the results:  python scripts/review_predictions.py "
        "--config ... --only-wrong",
        "check the least certain calls:  python scripts/review_predictions.py "
        "--config ... --only-uncertain 0.4",
    ] + ([] if have_labels else [
        "to get an accuracy figure, label some throws:  "
        "python scripts/label_throwers.py --config ...",
    ]))


def bold_stage(name: str, description: str) -> str:
    return logs.bold(f"  STAGE: {description}") + logs.dim(f"   ({name})")


if __name__ == "__main__":
    main()
