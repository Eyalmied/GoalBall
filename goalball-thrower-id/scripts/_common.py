"""
Shared plumbing for the command-line scripts.

Keeps the flags identical across every script, so anything you learn about one
transfers to the others.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goalball import config, logs        # noqa: E402  (after the path fix)


def base_parser(description: str) -> argparse.ArgumentParser:
    """The flags every script understands."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    where = parser.add_argument_group(
        "which game",
        "Point these at your match. Easiest is to put them in a config file "
        "once (see config/example.json) and pass --config.")
    where.add_argument("--config", default=None,
                       help="JSON file describing the game (video, CSV, roster, ...)")
    where.add_argument("--video", default=None, help="path to the match video")
    where.add_argument("--events", default=None,
                       help="the upstream detector's throws CSV")
    where.add_argument("--gallery-dir", dest="gallery_dir", default=None,
                       help="folder holding the reference gallery")
    where.add_argument("--court-file", dest="court_file", default=None,
                       help="JSON file holding the court corners")
    where.add_argument("--labels", default=None,
                       help="CSV of hand-labelled throwers (ground truth)")
    where.add_argument("--predictions", default=None,
                       help="JSON file of predictions")

    output = parser.add_argument_group("output")
    output.add_argument("--quiet", action="store_true",
                        help="print almost nothing")
    output.add_argument("--verbose", action="store_true",
                        help="print the internals too, useful when debugging")
    return parser


def add_half_flags(parser: argparse.ArgumentParser, *,
                   include_auto: bool = True) -> None:
    """
    The half of the match being analysed.

    This matters because the teams change ends at half-time, so every recorded
    home zone moves to the opposite end of the court. See docs/HALVES.md.

    `half2_start` in the config file is the source of truth for WHEN the ends
    change. These flags only override it, and say so when they do.
    """
    group = parser.add_argument_group(
        "which half",
        "Teams change ends at half-time, so the half decides where each "
        "player's home zone is. Getting it wrong does not degrade the result "
        "gently - it inverts it. Put half2_start in your config file once and "
        "none of these flags are needed.")
    choices = ["1", "2"] + (["auto"] if include_auto else [])
    group.add_argument(
        "--half", default=None, choices=choices,
        help="1 or 2 = treat the whole run as that half"
             + ("; auto = the video covers both. Default: auto when the config "
                "gives half2_start, otherwise 1."
                if include_auto else "  (default: 1)"))
    if include_auto:
        group.add_argument(
            "--half2-start", dest="half2_start", default=None,
            help="override the config's half2_start for this run only "
                 "(video timestamp where the second half begins, e.g. 24:00)")
    group.add_argument(
        "--gallery-half", dest="gallery_half", type=int, default=None,
        choices=[1, 2],
        help="override which half the gallery's home zones were tagged in "
             "(normally read from the gallery itself)")
    group.add_argument(
        "--mirror-about", dest="mirror_about", default="zones",
        choices=["zones", "court"],
        help="what the change of ends turns the home zones about: zones = the "
             "middle of the zones you tagged (default, and the more robust "
             "choice); court = the middle of the court polygon")


def resolve(args, *required: str):
    """Apply the config file plus CLI overrides, and report what will be used."""
    logs.configure(quiet=args.quiet, verbose=args.verbose)
    cfg = config.load(args.config, vars(args))
    config.require(cfg, *required)
    return cfg


def half_boundary(args, cfg) -> tuple:
    """
    (timestamp_text, where_it_came_from) for the change of ends.

    The config file is the source of truth. A command-line --half2-start
    overrides it for one run, and the override is reported rather than applied
    quietly, because the two disagreeing is almost always a mistake.
    """
    # cfg already carries the command line layered over the file, so cfg is
    # the answer. file_values is only consulted to say where it came from.
    resolved = getattr(cfg, "half2_start", None)
    in_file = (getattr(cfg, "file_values", None) or {}).get("half2_start")
    from_cli = getattr(args, "half2_start", None)

    if not resolved:
        return None, None
    if from_cli and in_file and str(from_cli) != str(in_file):
        logs.warn(f"--half2-start {from_cli} overrides half2_start {in_file} "
                  f"from the config, for this run only",
                  f"the config is what every other step reads, so if {from_cli} "
                  f"is right put it there - otherwise the steps will disagree "
                  f"about where the match splits")
        return resolved, "the command line, overriding the config"
    if from_cli:
        return resolved, "the command line"
    return resolved, "the config file"


def resolve_halves(args, cfg, fps: float, gallery_half: int = 1,
                   announce: bool = True):
    """
    Build the run's HalfPlan from one place, so every script agrees.

    `--half` is left unset by default: when the config knows where the second
    half starts, the video evidently covers both and `auto` is the only answer
    that can be right. Naming a half explicitly still wins.
    """
    from goalball import halves

    boundary, source = half_boundary(args, cfg)
    mode = getattr(args, "half", None)
    if mode is None:
        mode = "auto" if boundary else "1"
        why = ("the config gives a half-time, so both halves are in play"
               if boundary else "no half-time known, so the run is one half")
    else:
        why = "you asked for it with --half"

    if announce:
        logs.kv("half", f"{mode}   ({why})")
        if boundary:
            logs.kv("ends change at", f"{boundary}   (from {source})")
        elif mode == "auto":
            logs.kv("ends change at", logs.yellow("not set"))

    return halves.build(gallery_half, mode, boundary, fps,
                        mirror_about=getattr(args, "mirror_about", "zones"))


def parse_timestamp(text, fps: float, default: int = 0) -> int:
    """Accept '12:34', '754' seconds with an s suffix, or a raw frame number."""
    if text is None:
        return default
    text = str(text).strip()
    from goalball.events import mmss_to_seconds
    if ":" in text:
        return int(round(mmss_to_seconds(text) * fps))
    if text.endswith("s"):
        return int(round(float(text[:-1]) * fps))
    return int(float(text))
