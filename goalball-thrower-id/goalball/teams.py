"""
Which team is attacking?

THE OBVIOUS ANSWER IS WRONG
    "Whoever is closest to the ball threw it" scored 32% on 85 labelled
    throws - worse than a coin toss. The reason is simple once you look at the
    frames: the ball is easiest to detect where it is largest and slowest, and
    that is at the BLOCKER's end. Proximity to the ball therefore picks the
    defending team almost every time.

THE ANSWER THAT WORKS: POSTURE
    Goalball defenders throw themselves flat across their goal line to block.
    Attackers stay on their feet to throw. So a detection box tells you the
    role: lying players are wide and short, standing players are tall and
    narrow. Average that "flatness" per team, and the more upright team is the
    attacking one. This took team accuracy from 32% to 89%.

    It also generalises, which the ball-based rule never could: it depends on
    how goalball is played, not on which end a team happens to be defending -
    so it survives the half-time change of ends untouched.

GROUPING THE PLAYERS INTO TWO TEAMS
    By kit colour, from a hue/saturation histogram of each player's torso
    matched against prototypes built from the gallery. The two kits are fully
    separable this way, so colour is reliable for "same team or not" even
    though it is useless for telling teammates apart.

    But NOT by asking each crop, on its own, which kit it is closer to. A
    player at the far end of the court is a fraction of the size of one near
    the camera, and their torso crop is mostly not jersey - so the colour
    washes out and the crop drifts toward whichever kit the gallery happened to
    record at that scale. Measured on the reference match: absolute matching is
    right 97% of the time in the half the gallery was tagged in, and 22% in the
    other half, where the far end belongs to the other team. It is not noisy,
    it inverts - which would quietly ruin every second-half prediction.

    So the two groups are decided RELATIVELY, inside each frame. Every
    detection gets a "lean": how much more it looks like kit A than kit B. The
    detections are then ranked by that lean and split according to how many
    players each side has (three and three, in goalball). Only the ORDER of the
    leans has to be right, not their absolute values, so a scale or lighting
    shift that moves every crop the same way cancels out. Same match, same
    prototypes: 97% and 91%.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import logs

# The torso region of a crop: skip the head and the legs, keep the shirt.
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.55
TORSO_LEFT, TORSO_RIGHT = 0.15, 0.85
HUE_BINS, SAT_BINS = 30, 32


def torso_histogram(crop: np.ndarray) -> Optional[np.ndarray]:
    """Hue/saturation histogram of a player's shirt, or None if the crop is tiny."""
    height, width = crop.shape[:2]
    if height < 12 or width < 8:
        return None
    region = crop[int(height * TORSO_TOP):int(height * TORSO_BOTTOM),
                  int(width * TORSO_LEFT):int(width * TORSO_RIGHT)]
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [HUE_BINS, SAT_BINS],
                        [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def build_prototypes(crops: dict, player_config: dict) -> dict:
    """{team -> average shirt histogram} from the gallery."""
    per_team: dict = {}
    for pid, images in crops.items():
        team = player_config.get(pid, {}).get("team")
        if not team:
            continue
        for image in images:
            hist = torso_histogram(image)
            if hist is not None:
                per_team.setdefault(team, []).append(hist)
    return {team: np.mean(hists, axis=0) for team, hists in per_team.items()}


def _similarities(crop: np.ndarray, prototypes: dict) -> Optional[dict]:
    """{team -> histogram correlation} for one crop."""
    hist = torso_histogram(crop)
    if hist is None or not prototypes:
        return None
    return {team: float(cv2.compareHist(hist.astype("float32"),
                                        prototype.astype("float32"),
                                        cv2.HISTCMP_CORREL))
            for team, prototype in prototypes.items()}


def team_of_crop(crop: np.ndarray, prototypes: dict) -> tuple:
    """(team, similarity) for one detection, or (None, 0.0) if unusable."""
    scores = _similarities(crop, prototypes)
    if not scores:
        return None, 0.0
    best_team = max(scores, key=scores.get)
    return best_team, scores[best_team]


def group_relatively(detections: list, prototypes: dict,
                     team_sizes: dict) -> None:
    """
    Split the on-court detections between the two teams by their colour lean,
    writing 'team' and 'team_lean' onto each detection in place.

    Only the ranking of the leans is trusted, never the raw value, so a
    lighting or scale shift that pushes every crop the same way cancels. See
    the note at the top of this module for why that matters at half-time.

    Falls back to per-crop nearest-kit matching when there are not two kits to
    compare, and leaves detections whose crop is unusable with team None.
    """
    names = sorted(prototypes)
    if len(names) != 2:
        for detection in detections:
            team, score = team_of_crop(detection["crop"], prototypes)
            detection["team"], detection["team_lean"] = team, score
        return

    first, second = names
    usable = []
    for detection in detections:
        scores = _similarities(detection["crop"], prototypes)
        if scores is None:
            detection["team"], detection["team_lean"] = None, 0.0
            continue
        detection["team_lean"] = scores[first] - scores[second]
        detection["team"] = None
        usable.append(detection)

    if not usable:
        return
    if len(usable) == 1:
        # Nothing to compare against; the absolute answer is all there is.
        usable[0]["team"] = (first if usable[0]["team_lean"] >= 0 else second)
        return

    share = team_sizes.get(first, 3) / max(
        1, team_sizes.get(first, 3) + team_sizes.get(second, 3))
    take = max(1, min(len(usable) - 1, int(round(len(usable) * share))))
    ranked = sorted(usable, key=lambda d: -d["team_lean"])
    for index, detection in enumerate(ranked):
        detection["team"] = first if index < take else second


def check_prototypes(crops: dict, player_config: dict, prototypes: dict) -> None:
    """
    Sanity check the two kits really are separable, and say so plainly.

    Run once at start-up: if the kits are not separable in this video the whole
    team decision is unreliable, and the user should know before waiting for a
    full run.
    """
    total = correct = 0
    for pid, images in crops.items():
        truth = player_config.get(pid, {}).get("team")
        for image in images:
            guessed, _ = team_of_crop(image, prototypes)
            total += 1
            correct += int(guessed == truth)
    if not total:
        logs.warn("no reference images to check the kit colours against")
        return
    rate = 100.0 * correct / total
    line = f"{correct}/{total} reference images ({rate:.0f}%)"
    if rate >= 95:
        logs.ok(f"the two kits are clearly different colours - {line}")
    elif rate >= 80:
        logs.warn(f"the two kits are only partly separable by colour - {line}",
                  "add gallery images with better lighting; team calls will be "
                  "noisier than usual")
    else:
        logs.warn(f"the two kits look alike to the colour model - {line}",
                  "re-tag the gallery with clearer, brighter crops - team "
                  "grouping is unreliable at this level")


def attacking_team(detections: list) -> tuple:
    """
    Decide which team is attacking from posture.

    Returns (team, flatness_by_team, separation) where `separation` is 0..1 -
    how differently the two teams are standing. A separation near 0 means both
    teams look alike (nobody is down yet) and the call is a guess; it is fed
    into the confidence score rather than hidden.
    """
    flatness: dict = {}
    for detection in detections:
        team = detection.get("team")
        if team:
            flatness.setdefault(team, []).append(detection["flatness"])
    if len(flatness) < 2:
        return None, {team: float(np.mean(v)) for team, v in flatness.items()}, 0.0

    means = {team: float(np.mean(values)) for team, values in flatness.items()}
    # lower flatness = more upright = the team on its feet = the attackers
    attacker = min(means, key=means.get)
    low, high = min(means.values()), max(means.values())
    separation = float(min((high - low) / (high + low + 1e-9) / 0.25, 1.0))
    return attacker, means, separation
