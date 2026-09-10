"""
Finding people and the ball in a frame.

PEOPLE
    One pass of YOLO-pose gives both a box and a 17-point skeleton per person,
    so posture (which team is attacking) and throwing shape (which attacker
    threw) come free with the detection - no second model.

    Every detection is then checked against the court fence using the FEET,
    not the middle of the box. A spectator in the front row can easily have
    their head and torso overlapping the court in the image; their feet never
    do.

THE BALL
    Read with the generic COCO "sports ball" class. It is good enough to find
    roughly where the ball was released and no better - measured at ~55%
    when used to place the thrower, versus ~81% for the upstream detector's
    own release point. That gap is the single biggest limit on accuracy today;
    see docs/RESULTS.md.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import logs
from .pose import throw_score

PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32

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


def warm_up(pose_model: str, ball_model: Optional[str] = None) -> None:
    """Load the models up front so the first throw is not mysteriously slow."""
    _load(pose_model, "body-pose")
    if ball_model:
        _load(ball_model, "ball")


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
        throw_pose  0..1 "looks like a release" score
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
            "throw_pose": throw_score(skeleton),
        })
    return found


def ball_release_point(cap, throw, court, ball_model: str, proc_w: int,
                       window_fraction: float, samples: int = 4,
                       confidence: float = 0.10) -> Optional[tuple]:
    """
    Best guess at where the ball was released, found in the video alone.

    Looks at a few frames right after the throw starts and keeps the most
    confident on-court ball. Returns (x, y) in this frame's pixels, or None
    when the ball was never confidently seen - in which case the caller falls
    back to posture and pose only, rather than trusting a bad point.
    """
    from .video import read_frame

    model = _load(ball_model, "ball")
    start, end = throw.window(window_fraction)
    best = None                                    # (confidence, x, y)

    for index in np.linspace(start, end, max(1, samples), dtype=int):
        frame = read_frame(cap, int(index), proc_w)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        polygon = court.polygon_px(width, height) if court is not None else None
        result = model(frame, classes=[SPORTS_BALL_CLASS], conf=confidence,
                       verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for score, box in zip(result.boxes.conf.cpu().numpy(),
                              result.boxes.xyxy.cpu().numpy()):
            cx = float((box[0] + box[2]) / 2)
            cy = float((box[1] + box[3]) / 2)
            if polygon is not None and \
                    cv2.pointPolygonTest(polygon, (cx, cy), False) < 0:
                continue        # a ball in the crowd is a false positive
            if best is None or float(score) > best[0]:
                best = (float(score), cx, cy)

    return (best[1], best[2]) if best else None
