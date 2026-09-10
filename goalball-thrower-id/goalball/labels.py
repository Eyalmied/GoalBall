"""
The ground-truth file: who actually threw each ball, according to a human.

Only needed if you want to MEASURE accuracy. Predictions run fine without it.

The file is a small CSV so it can be opened, sorted and corrected in any
spreadsheet:

    throw_id, thrower_player_id, thrower_name, half, start_time, csv_team_hint

`thrower_player_id` empty means "not sure" - those throws are excluded from
accuracy rather than counted as wrong, because a label you do not trust is
worse than no label.
"""

from __future__ import annotations

import csv
import os
import pathlib

from . import logs

COLUMNS = ["throw_id", "thrower_player_id", "thrower_name", "half",
           "start_time", "csv_team_hint"]

# Column names used by the earlier scripts, mapped onto the current ones.
LEGACY = {"event_id": "throw_id", "throwing_team": "csv_team_hint",
          "start_time_mmss": "start_time"}


def load(path: str, quiet: bool = False) -> dict:
    """{throw_id -> row}. Missing file is not an error; it just means no labels."""
    file = pathlib.Path(path)
    if not file.exists():
        return {}
    rows = {}
    with open(file, newline="") as f:
        for raw in csv.DictReader(f):
            row = {LEGACY.get(k, k): v for k, v in raw.items() if k}
            key = row.get("throw_id")
            if key:
                rows[key] = row
    if not quiet:
        sure = sum(1 for r in rows.values() if is_labelled(r))
        logs.ok(f"{sure} labelled throws read from {path}"
                + (f" ({len(rows) - sure} marked unsure)" if len(rows) > sure else ""))
    return rows


def save(path: str, rows: dict, order: list = None) -> None:
    """Write atomically, so a crash mid-write can never lose your labels."""
    keys = order or list(rows)
    temporary = f"{path}.tmp"
    with open(temporary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for key in keys:
            if key in rows:
                writer.writerow({c: rows[key].get(c, "") for c in COLUMNS})
    os.replace(temporary, path)


def is_labelled(row: dict) -> bool:
    return str((row or {}).get("thrower_player_id", "")).strip() not in ("", "0", "None")


def player_id(row: dict):
    return int(row["thrower_player_id"]) if is_labelled(row) else None
