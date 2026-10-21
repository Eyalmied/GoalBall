"""
Finding people in a frame.

PEOPLE
    One pass of YOLO-pose gives a box per person, and the box shape alone
    says whether they are upright or already down - which is what decides
    the attacking team.

    Every detection is then checked against the court fence using the FEET,
    not the middle of the box. A spectator in the front row can easily have
    their head and torso overlapping the court in the image; their feet never
    do.

THE THROWER
    A detector fine-tuned on goalball names the thrower outright. The
    movement-and-posture rule below is only the fallback for the throws it
    misses; see docs/RESULTS.md.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

from . import logs

PERSON_CLASS = 0

# A detector fine-tuned on goalball labels the thrower directly. Class 0 of
# the project's own weights is "throwing_player" - a model trained to answer
# the exact question this pipeline asks by hand. Measured on 181 labelled
# throws it finds the right player 73% of the time against 46% for the
# motion-and-posture rule, so it is used first whenever it is available.
THROWING_PLAYER_CLASS = 0

_models: dict = {}


def _load(path: str, what: str):
    """Load a YOLO model once and reuse it for the rest of the run."""
    if path in _models:
        return _models[path]
    try:
        from ultralytics import YOLO
    except ImportError:
        logs.fail("the ultralytics package is not installed",
                  "run:  python -m pip install ultralytics")
    logs.info(f"loading the {what} model ({path}) ...")
    try:
        with logs.muted():
            model = YOLO(path)
    except Exception as e:
        logs.fail(f"could not load the {what} model from {path} ({e})",
                  "check the path, or let ultralytics download it by passing "
                  "just the model name, e.g. --pose-model yolov8m-pose.pt")
    _models[path] = model
    logs.ok(f"{what} model ready")
    return model


def warm_up(pose_model: str) -> None:
    """Load the model up front so the first throw is not mysteriously slow."""
    _load(pose_model, "body-pose")


def people(frame: np.ndarray, court, pose_model: str,
           confidence: float = 0.25) -> list:
    """
    Every on-court person in this frame.

    Each entry:
        box         (x1, y1, x2, y2) in this frame's pixels
        crop        the pixels inside the box
        keypoints   the 17-point skeleton (or None)
        feet        (x, y) bottom-centre of the box, in pixels
        feet_norm   the same point as a fraction of the frame
        flatness    box width / height - high means lying down
    """
    model = _load(pose_model, "body-pose")
    height, width = frame.shape[:2]
    result = model(frame, classes=[PERSON_CLASS], conf=confidence, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    if result.keypoints is not None:
        skeletons = result.keypoints.data.cpu().numpy()
    else:
        skeletons = [None] * len(boxes)

    polygon = court.polygon_px(width, height) if court is not None else None
    found = []
    for box, skeleton in zip(boxes, skeletons):
        x1, y1, x2, y2 = (int(v) for v in box)
        feet = (float((x1 + x2) / 2), float(y2))
        if polygon is not None and cv2.pointPolygonTest(polygon, feet, False) < 0:
            continue
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        found.append({
            "box": (x1, y1, x2, y2),
            "crop": crop,
            "keypoints": skeleton,
            "feet": feet,
            "feet_norm": (feet[0] / width, feet[1] / height),
            "flatness": (x2 - x1) / max(1.0, y2 - y1),
        })
    return found


def throwing_players(frame: np.ndarray, court, model_path: str,
                     confidence: float = 0.15,
                     class_index: int = THROWING_PLAYER_CLASS) -> list:
    """
    Every on-court player this frame's detector calls a thrower.

    Returns the same shape as `people()` minus the skeleton, so the caller can
    treat both the same way. An empty list means the detector saw no thrower
    in this frame, and the caller should fall back to the motion rule.
    """
    model = _load(model_path, "goalball thrower")
    height, width = frame.shape[:2]
    result = model(frame, classes=[class_index], conf=confidence,
                   verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    polygon = court.polygon_px(width, height) if court is not None else None
    found = []
    for score, box in zip(result.boxes.conf.cpu().numpy(),
                          result.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = (int(v) for v in box)
        feet = (float((x1 + x2) / 2), float(y2))
        if polygon is not None and \
                cv2.pointPolygonTest(polygon, feet, False) < 0:
            continue
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        found.append({
            "box": (x1, y1, x2, y2), "crop": crop, "keypoints": None,
            "feet": feet, "feet_norm": (feet[0] / width, feet[1] / height),
            "flatness": (x2 - x1) / max(1.0, y2 - y1),
            "confidence": float(score),
        })
    return found


def has_model(path) -> bool:
    """True when a model file is actually there (a bare name counts: it downloads)."""
    if not path:
        return False
    return "/" not in str(path) or pathlib.Path(path).exists()


def motion_between(cap, first, proc_w: int, gap: int = 2):
    """
    How much each pixel changed over `gap` frames, starting at `frame_index`.

    The thrower is the player who MOVES: a run-up, a wind-up, an arm through
    the ball. The defenders are already down and waiting, and the thrower's own
    team-mates stand still. So frame-to-frame change separates the thrower from
    everybody else far better than where the ball happens to be detected -
    measured on 181 labelled throws, 58% correct team against 93%.

    `first` is the frame the caller has already read and is analysing, so this
    reads FORWARD from wherever the capture is now rather than seeking back to
    it. Seeking into a long video is by far the slowest thing here, and the
    caller has just paid for one.

    Returns a float32 image of absolute differences, or None if the later
    frame could not be read.
    """
    from .video import downscale

    second = None
    for _ in range(max(1, gap)):
        ok, raw = cap.read()
        if not ok:
            return None
        second = raw
    second = downscale(second, proc_w)
    if second is None or second.shape != first.shape:
        return None
    return np.abs(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float32)
                  - cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float32))


def motion_in_box(motion, box) -> float:
    """Mean change inside one detection's box, 0.0 when unavailable."""
    if motion is None:
        return 0.0
    x1, y1, x2, y2 = box
    patch = motion[max(0, y1):y2, max(0, x1):x2]
    return float(patch.mean()) if patch.size else 0.0
