"""
Run settings: one config file per game, overridable on the command line.

Nothing about a specific match belongs in the source. A game is described by a
small JSON file (see ../config/example.json):

    {
      "name":        "Paralympics 2024 SF - Israel vs China",
      "video":       "../Paralympics24_SF_Israel2_China1.MP4",
      "events":      "../Paralympics24..._Throws_data_predicted.csv",
      "gallery_dir": "gallery",
      "court_file":  "court.json",
      "labels":      "thrower_labels.csv",
      "predictions": "thrower_predictions.json",
      "pose_model":  "../yolov8m-pose.pt",
      "ball_model":  "../yolov8m.pt",
      "half2_start": "24:00",
      "roster": {
        "1": {"name": "Ada",  "team": "ISR"},
        ...
      }
    }

Paths inside the file are resolved relative to THE REPOSITORY FOLDER (the one
holding `goalball/` and `scripts/`), so a config reads the same no matter which
directory you run from:

    "gallery"                 -> <repo>/gallery
    "../my_match.mp4"         -> the folder above the repo

A bare model name is left alone, because it is a name and not a path:

    "yolov8m-pose.pt"         -> ultralytics downloads it on first use
    "Model Weights/best.pt"   -> <repo>/Model Weights/best.pt

Paths passed on the COMMAND LINE are ordinary shell paths, relative to wherever
you happen to be standing. Any config value can be overridden by its matching
command-line flag.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional

from . import logs

# The folder holding goalball/ and scripts/ - the anchor for config paths.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Keys whose value is a path, anchored at the repo folder when it is relative.
PATH_KEYS = ("video", "events", "gallery_dir", "court_file", "labels",
             "predictions", "pose_model", "ball_model")

# Model keys are special: a BARE file name like "yolov8m-pose.pt" is a model
# name, which ultralytics downloads on first use. Only a value that actually
# looks like a path ("weights/best.pt", "../best.pt") is anchored.
MODEL_KEYS = ("pose_model", "ball_model")

DEFAULTS: dict = {
    "name": "unnamed game",
    "video": None,
    "events": None,
    "gallery_dir": "gallery",
    "court_file": "court.json",
    "labels": "thrower_labels.csv",
    "predictions": "thrower_predictions.json",
    "pose_model": "yolov8m-pose.pt",
    "ball_model": "yolov8m.pt",
    "half2_start": None,
    "roster": {},
    "source": None,
}


@dataclass
class GameConfig:
    """Everything a script needs to know about which game it is working on."""

    name: str = DEFAULTS["name"]
    video: Optional[str] = None
    events: Optional[str] = None
    gallery_dir: str = DEFAULTS["gallery_dir"]
    court_file: str = DEFAULTS["court_file"]
    labels: str = DEFAULTS["labels"]
    predictions: str = DEFAULTS["predictions"]
    pose_model: str = DEFAULTS["pose_model"]
    ball_model: str = DEFAULTS["ball_model"]
    half2_start: Optional[str] = None
    roster: dict = field(default_factory=dict)
    source: Optional[str] = None       # which config file this came from

    # -- roster ---------------------------------------------------------
    def player_config(self) -> dict:
        """{player_id:int -> {"name", "team"}} from the config's roster."""
        return {int(k): dict(v) for k, v in (self.roster or {}).items()}

    def teams(self) -> list:
        seen = []
        for v in self.player_config().values():
            if v.get("team") and v["team"] not in seen:
                seen.append(v["team"])
        return seen

    # -- reporting ------------------------------------------------------
    def describe(self) -> None:
        logs.kv("game", self.name)
        if self.source:
            logs.kv("settings from", self.source)
        logs.kv("video", self.video or logs.yellow("not set"))
        logs.kv("throws CSV", self.events or logs.yellow("not set"))
        logs.kv("gallery folder", self.gallery_dir)
        logs.kv("court file", self.court_file)


def _resolve(base: pathlib.Path, value: Any, key: str = "") -> Any:
    """Make a path from a config file absolute, anchored at the repo folder."""
    if not isinstance(value, str) or not value:
        return value
    if key in MODEL_KEYS and "/" not in value and "\\" not in value:
        return value                      # a model name, not a path
    p = pathlib.Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def load(path: Optional[str], overrides: Optional[dict] = None,
         base: Optional[pathlib.Path] = None) -> GameConfig:
    """
    Read a game config file (optional) and layer CLI overrides on top.

    `overrides` is normally `vars(args)`; only keys the config understands and
    whose value is not None are applied, so unset flags never clobber the file.
    """
    data = dict(DEFAULTS)
    base = pathlib.Path(base).resolve() if base else REPO_ROOT

    if path:
        f = pathlib.Path(path).expanduser()
        if not f.exists():
            logs.fail(f"config file not found: {path}",
                      "check the path, or drop --config and pass --video/--events directly")
        try:
            raw = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            logs.fail(f"config file {path} is not valid JSON ({e})",
                      "open it in an editor - a missing comma or quote is the usual cause")
        for key, value in raw.items():
            data[key] = (_resolve(base, value, key) if key in PATH_KEYS
                         else value)
        data["source"] = str(f)

    for key, value in (overrides or {}).items():
        if key in data and value is not None:
            data[key] = value

    known = {k: v for k, v in data.items()
             if k in GameConfig.__dataclass_fields__}
    cfg = GameConfig(**known)

    unknown = sorted(set(data) - set(GameConfig.__dataclass_fields__))
    if unknown:
        logs.warn(f"config keys ignored (not understood): {', '.join(unknown)}",
                  "remove them, or check the spelling against config/example.json")
    return cfg


def require(cfg: GameConfig, *keys: str) -> None:
    """Stop with a clear message if a script cannot run without a setting."""
    missing = [k for k in keys if not getattr(cfg, k, None)]
    if missing:
        flags = ", ".join(f"--{k}" for k in missing)
        logs.fail(f"missing required setting(s): {', '.join(missing)}",
                  f"pass {flags} on the command line, or add them to your --config file")
