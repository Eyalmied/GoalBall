"""
The court: where the game is, and how to flip it end-for-end.

TWO JOBS
--------
1. FENCE. The four clicked corners form a polygon. Any detected person whose
   feet fall outside it is not a player - it is crowd, a referee, a coach or a
   substitute. Dropping them before any reasoning happens removes the single
   biggest source of wrong answers.

2. MAP. Those same four corners define a perspective mapping (a homography)
   between the camera image and a flat, rectangular "court plane" whose
   coordinates run 0..1 along each axis:

        (0,0) ------------ (1,0)      u = across the court
          |                  |        v = along the court, 0 = the end at the
          |     court        |            TOP of the image, 1 = the end at
          |                  |            the BOTTOM of the image
        (0,1) ------------ (1,1)

   In that flat space, swapping ends at half-time is a half-turn about a
   centre point:

        (u, v)  ->  (2*cu - u, 2*cv - v)

   which is what physically happens to a player's home zone when the teams
   change ends. Doing the flip in the court plane (rather than just flipping
   the image) is what makes it correct under the camera's perspective: the far
   end of the court occupies far fewer pixels than the near end, so a naive
   image flip would put a player metres away from where they actually stand.

   The centre defaults to the middle of the court, (0.5, 0.5). It can be
   overridden with the centre of the recorded home zones instead, which is
   what the pipeline actually does - see goalball/halves.py for why that is
   the more robust choice in practice.

All positions in this package are stored as NORMALISED FRAME COORDINATES -
(x/frame_width, y/frame_height), both in 0..1 - so they are independent of the
processing resolution. Court methods take and return that same space.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import logs


@dataclass
class Court:
    """Four court corners in normalised frame coordinates, plus the mapping."""

    corners: np.ndarray            # (4, 2) float, normalised, TL TR BR BL
    source: Optional[str] = None   # the file it was loaded from

    # -- construction ---------------------------------------------------
    def __post_init__(self):
        self.corners = np.asarray(self.corners, dtype=np.float64).reshape(4, 2)
        unit = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
        self._to_plane = cv2.getPerspectiveTransform(
            self.corners.astype(np.float32), unit.astype(np.float32))
        self._to_frame = cv2.getPerspectiveTransform(
            unit.astype(np.float32), self.corners.astype(np.float32))

    # -- fencing --------------------------------------------------------
    def polygon_px(self, width: int, height: int) -> np.ndarray:
        """The polygon in pixels, ready for cv2.pointPolygonTest / drawing."""
        pts = self.corners * np.array([width, height], dtype=np.float64)
        return pts.astype(np.int32).reshape(-1, 1, 2)

    def contains_norm(self, x: float, y: float) -> bool:
        poly = self.corners.astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    @property
    def centre_norm(self) -> tuple:
        return (float(self.corners[:, 0].mean()), float(self.corners[:, 1].mean()))

    # -- mapping between the image and the flat court -------------------
    def to_plane(self, x: float, y: float) -> tuple:
        """Normalised frame point -> court-plane (u, v), both 0..1 on court."""
        p = np.array([[[float(x), float(y)]]], dtype=np.float32)
        u, v = cv2.perspectiveTransform(p, self._to_plane)[0][0]
        return float(u), float(v)

    def to_frame(self, u: float, v: float) -> tuple:
        """Court-plane (u, v) -> normalised frame point."""
        p = np.array([[[float(u), float(v)]]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(p, self._to_frame)[0][0]
        return float(x), float(y)

    def mirror_norm(self, x: float, y: float, centre=(0.5, 0.5)) -> tuple:
        """
        Where this point ends up after the teams change ends.

        Correct under perspective, because the half-turn is applied on the flat
        court and then mapped back into the image. `centre` is the point turned
        about, in court-plane coordinates.
        """
        u, v = self.to_plane(x, y)
        return self.to_frame(2.0 * centre[0] - u, 2.0 * centre[1] - v)

    def end_of(self, x: float, y: float) -> str:
        """'top' or 'bottom' - which half of the court this point is in."""
        _, v = self.to_plane(x, y)
        return "top" if v < 0.5 else "bottom"

    def describe(self) -> None:
        cx, cy = self.centre_norm
        logs.kv("court corners", "4 corners loaded"
                                 + (f" from {self.source}" if self.source else ""))
        logs.kv("centre spot", f"x={cx:.2f}, y={cy:.2f} (fraction of the frame)")


# ----------------------------------------------------------------------
# Fallback when no court has been marked
# ----------------------------------------------------------------------

class WholeFrameCourt(Court):
    """
    Stand-in used when court.json is missing: the whole frame counts as court
    and mirroring is a plain flip about the frame centre.

    It keeps the pipeline runnable, but crowd and referees are no longer
    filtered out, and the end-swap flip is only approximate. Mark the court.
    """

    def __init__(self):
        super().__init__(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                         source=None)
        self.is_fallback = True

    def polygon_px(self, width: int, height: int):
        return None                      # None means "do not filter"

    def contains_norm(self, x: float, y: float) -> bool:
        return True

    def mirror_norm(self, x: float, y: float, centre=(0.5, 0.5)) -> tuple:
        return (2.0 * centre[0] - float(x), 2.0 * centre[1] - float(y))

    def describe(self) -> None:
        logs.kv("court", "not marked - using the whole frame")


# ----------------------------------------------------------------------
# Ordering the four clicked corners
# ----------------------------------------------------------------------

def order_corners(points) -> np.ndarray:
    """
    Put four clicked points into the order top-left, top-right, bottom-right,
    bottom-left, whatever order the user clicked them in. Being click-order
    independent means the court file can never silently produce a mirrored map.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    ordered = pts[np.argsort(angles)]              # counter-clockwise in image y-down
    start = int(np.argmin(ordered.sum(axis=1)))    # top-left has the smallest x+y
    ordered = np.roll(ordered, -start, axis=0)
    # arctan2 with y pointing down traverses clockwise on screen; make sure the
    # second point is the top-right one (larger x, similar y) and not bottom-left
    if ordered[1][0] < ordered[3][0]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def save(path: str, points_px, width: int, height: int) -> None:
    """Store the corners with the frame size they were clicked at."""
    ordered = order_corners(points_px)
    data = {
        "width": int(width),
        "height": int(height),
        "points": [[float(p[0]), float(p[1])] for p in ordered],
        "note": "clicked court corners, ordered top-left, top-right, "
                "bottom-right, bottom-left",
    }
    pathlib.Path(path).write_text(json.dumps(data, indent=2))
    logs.ok(f"court saved to {path}")


def load(path: str, required: bool = False) -> Court:
    """
    Load a court file. Returns a WholeFrameCourt (with a loud warning) when the
    file is absent, unless `required` is set.
    """
    f = pathlib.Path(path)
    if not f.exists():
        if required:
            logs.fail(f"no court file at {path}",
                      "run scripts/set_court.py once to click the four court corners")
        logs.warn(f"no court file at {path} - crowd and referees will not be filtered out",
                  "run scripts/set_court.py once; it takes about ten seconds")
        return WholeFrameCourt()
    try:
        data = json.loads(f.read_text())
        pts = data["points"]
        w = float(data.get("width") or 1)
        h = float(data.get("height") or 1)
        if len(pts) != 4:
            raise ValueError(f"expected 4 corners, found {len(pts)}")
        norm = [[float(p[0]) / w, float(p[1]) / h] for p in pts]
    except Exception as e:
        logs.fail(f"court file {path} could not be read ({e})",
                  "delete it and re-run scripts/set_court.py")
    court = Court(order_corners(norm), source=str(f))
    logs.ok(f"court loaded from {path}")
    return court


# ----------------------------------------------------------------------
# Interactive corner picker
# ----------------------------------------------------------------------

def pick_corners(frame: np.ndarray) -> Optional[list]:
    """
    Ask the user to click the four court corners on one frame.

    Returns the four points in this frame's pixel space, or None if the user
    chose to skip the court fence.
    """
    points: list = []
    window = "Mark the court - click the 4 corners"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    logs.section("CLICK THE FOUR COURT CORNERS")
    logs.info("Click the corners of the PLAYING SURFACE, in any order:")
    logs.bullet("the wooden area between the two goal lines")
    logs.bullet("NOT the aprons at the sides where referees and coaches stand")
    logs.info("")
    logs.info("Keys:  u = undo the last point   c or Enter = confirm   "
              "s = skip the fence   q = quit")

    while True:
        display = frame.copy()
        height, width = display.shape[:2]
        cv2.rectangle(display, (0, 0), (width, 64), (0, 0, 0), -1)
        cv2.putText(display, f"Click the 4 court corners  ({len(points)}/4)",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "u undo   c/Enter confirm   s skip   q quit",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for i, (px, py) in enumerate(points):
            cv2.circle(display, (px, py), 7, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(points) >= 2:
            cv2.polylines(display, [np.array(points, np.int32)],
                          isClosed=(len(points) == 4), color=(0, 255, 255),
                          thickness=2)
        if len(points) == 4:
            cv2.putText(display, "looks right? press c to save",
                        (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
        cv2.imshow(window, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("s"):
            cv2.destroyWindow(window)
            logs.warn("court fence skipped - every detected person will be considered")
            return None
        elif key == ord("q"):
            cv2.destroyWindow(window)
            logs.info("cancelled - nothing was saved")
            sys.exit(0)
        elif key in (ord("c"), 13, 10) and len(points) == 4:
            cv2.destroyWindow(window)
            return points
