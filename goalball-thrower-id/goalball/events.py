"""
The throws CSV -> a list of Throw objects.

WHAT WE TAKE FROM THE CSV
    The throw TIMESTAMP and the throw LENGTH. That is what defines the window
    of video to look at, and it is all the pipeline needs to decide a thrower.

WHAT WE DELIBERATELY DO NOT TAKE
    "Throwing Team" / "Defending Team" - the team is decided from the video by
    posture, so the answer never inherits the upstream detector's team call.
    They are kept on the record as `csv_team_hint` for comparison only.

THE ONE OPTIONAL EXTRA
    "From Coord" - the ball's release point. It is used only as the *position*
    of the thrower when the run asks for it (`--release_source csv`), never to
    decide the team. Turning it on is the difference between ~71% and ~50%
    correct-thrower; see docs/RESULTS.md.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
from typing import Optional

from . import logs


@dataclass
class Throw:
    throw_id: str
    start_frame: int
    end_frame: int
    start_seconds: float
    start_mmss: str
    meta: dict = field(default_factory=dict)

    @property
    def span(self) -> int:
        return max(1, self.end_frame - self.start_frame)

    def window(self, fraction: float) -> tuple:
        """The first `fraction` of the throw window - where the release is."""
        hi = self.start_frame + max(1, int(self.span * fraction))
        return self.start_frame, hi

    def release_point(self) -> Optional[tuple]:
        """The CSV ball release point in native pixels, or None."""
        raw = (self.meta or {}).get("from_coord") or ""
        try:
            x, y = ast.literal_eval(raw)
            return float(x), float(y)
        except Exception:
            return None

    @property
    def csv_team_hint(self) -> str:
        return (self.meta or {}).get("throwing_team", "")


def mmss_to_seconds(text: str) -> float:
    """'12:34' -> 754.0 ; also accepts 'h:mm:ss' and bare seconds."""
    parts = [float(p) for p in str(text).strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def load(csv_path: str, fps: float) -> list:
    """
    Read the throws CSV. Two schemas are recognised automatically:

      * the upstream detector's output  (has 'Start Time (mm:ss)')
      * a minimal frame-based CSV       (event_id, start_frame, end_frame)
    """
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            if "Start Time (mm:ss)" in fields:
                throws = _load_detector_csv(reader, fps)
                schema = "upstream throw detector"
            elif {"start_frame", "end_frame"} <= set(fields):
                throws = _load_frame_csv(reader, fps)
                schema = "simple frame list"
            else:
                logs.fail(
                    f"{csv_path} has no recognised columns",
                    "expected either 'Start Time (mm:ss)' (detector output) or "
                    "'event_id,start_frame,end_frame'",
                )
    except FileNotFoundError:
        logs.fail(f"throws CSV not found: {csv_path}",
                  "point --events at the upstream detector's ..._Throws_data_predicted.csv")

    if not throws:
        logs.fail(f"{csv_path} contained no usable throws",
                  "check that the file has rows and a start time on each row")

    with_release = sum(1 for t in throws if t.release_point())
    logs.ok(f"{len(throws)} throws read from the {schema}")
    logs.kv("first throw at", throws[0].start_mmss)
    logs.kv("last throw at", throws[-1].start_mmss)
    logs.kv("ball release point", f"present on {with_release}/{len(throws)} throws")
    if with_release == 0:
        logs.warn("no ball release points in the CSV",
                  "the run will fall back to video-only positioning "
                  "(--release_source thrower); accuracy is lower")
    return throws


def _load_detector_csv(reader: csv.DictReader, fps: float) -> list:
    throws, skipped = [], 0
    for row in reader:
        start_text = (row.get("Start Time (mm:ss)") or "").strip()
        if not start_text:
            skipped += 1
            continue
        start_seconds = mmss_to_seconds(start_text)
        start_frame = int(round(start_seconds * fps))
        try:
            length = int(float((row.get("Throw Length (frames)") or "").strip()))
        except ValueError:
            length = int(round(2.0 * fps))     # a throw is ~2 s if unstated
        number = (row.get("Throw Number") or "").strip()
        throws.append(Throw(
            throw_id=f"throw_{number}" if number else f"t_{start_frame}",
            start_frame=start_frame,
            end_frame=start_frame + max(1, length),
            start_seconds=start_seconds,
            start_mmss=start_text,
            meta={
                "throw_number": number,
                "throwing_team": (row.get("Throwing Team") or "").strip(),
                "defending_team": (row.get("Defending Team") or "").strip(),
                "from_zone": (row.get("From Zone") or "").strip(),
                "to_zone": (row.get("To Zone") or "").strip(),
                "from_coord": (row.get("From Coord") or "").strip(),
                "to_coord": (row.get("To Coord") or "").strip(),
                "predicted_outcome": (row.get("Predicted Outcome") or "").strip(),
                "throw_length_frames": length,
            },
        ))
    if skipped:
        logs.warn(f"{skipped} CSV row(s) had no start time and were skipped")
    return throws


def _load_frame_csv(reader: csv.DictReader, fps: float) -> list:
    throws = []
    for row in reader:
        start_frame = int(row["start_frame"])
        throws.append(Throw(
            throw_id=row.get("event_id") or f"t_{start_frame}",
            start_frame=start_frame,
            end_frame=int(row["end_frame"]),
            start_seconds=start_frame / fps if fps else 0.0,
            start_mmss=logs.mmss(start_frame / fps if fps else 0.0),
            meta={},
        ))
    return throws
