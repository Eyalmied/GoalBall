"""
"Does this body look like it is throwing?"

The thrower is in a distinctive shape at release: one arm swung out well away
from the shoulder, and a wide, low lunge. Both are measured from the pose
model's skeleton and then DIVIDED BY THE PLAYER'S OWN TORSO LENGTH.

That normalisation is the whole point. A raw pixel distance means nothing - a
player at the far end of the court is half the size of one near the camera, and
players differ in height. Expressed as a multiple of their own torso, the same
number means the same posture for everybody, in any video, at any zoom.
"""

from __future__ import annotations

import numpy as np

# COCO-17 keypoint order, which is what YOLO-pose returns.
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16

MIN_CONFIDENCE = 0.3      # ignore keypoints the model is unsure about
MIN_TORSO_PX = 5.0        # below this the skeleton is too small to trust
ARM_FULL_EXTENSION = 1.5  # wrist 1.5 torsos from the shoulder counts as "full"
STANCE_FULL_WIDTH = 1.2   # feet 1.2 torsos apart counts as a full lunge
ARM_WEIGHT = 0.6          # the arm is the more telling of the two


def throw_score(keypoints) -> float:
    """
    A 0..1 score: 0 = nothing like a throw, 1 = textbook release posture.

    Returns 0.0 whenever the skeleton is missing or too small to measure, so a
    bad detection can never masquerade as a confident throw.
    """
    if keypoints is None:
        return 0.0

    def point(index):
        x, y, confidence = keypoints[index]
        return (float(x), float(y)) if confidence > MIN_CONFIDENCE else None

    def distance(a, b):
        if a is None or b is None:
            return None
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    torso_lengths = [d for d in (distance(point(L_SHOULDER), point(L_HIP)),
                                 distance(point(R_SHOULDER), point(R_HIP))) if d]
    if not torso_lengths:
        return 0.0
    torso = float(np.mean(torso_lengths))
    if torso < MIN_TORSO_PX:
        return 0.0

    arms = [d for d in (distance(point(L_SHOULDER), point(L_WRIST)),
                        distance(point(R_SHOULDER), point(R_WRIST))) if d]
    stance = distance(point(L_ANKLE), point(R_ANKLE))

    arm_part = min((max(arms) / torso) / ARM_FULL_EXTENSION, 1.0) if arms else 0.0
    stance_part = min((stance / torso) / STANCE_FULL_WIDTH, 1.0) if stance else 0.0
    return ARM_WEIGHT * arm_part + (1.0 - ARM_WEIGHT) * stance_part
