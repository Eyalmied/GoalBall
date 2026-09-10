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
    """
    group = parser.add_argument_group(
        "which half",
        "Teams change ends at half-time, so the half decides where each "
        "player's home zone is. Getting it wrong does not degrade the result "
        "gently - it inverts it.")
    choices = ["1", "2"] + (["auto"] if include_auto else [])
    group.add_argument(
        "--half", default="1", choices=choices,
        help="1 or 2 = the whole run is that half"
             + ("; auto = the video covers both, switch at --half2-start"
                if include_auto else "")
             + "  (default: 1)")
    if include_auto:
        group.add_argument(
            "--half2-start", dest="half2_start", default=None,
            help="video timestamp where the second half begins, e.g. 24:00 "
                 "(only needed with --half auto)")
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
