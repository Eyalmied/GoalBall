"""
How good are the predictions? Measured against the human labels.

WHAT IS REPORTED AND WHY
    correct thrower      the end-to-end number: right team AND right player
    correct team         the easy half, solved by posture (~89%)
    right player given
      the right team     the hard half, and the real ceiling (~81% at best)
    per half             a big gap between halves almost always means the
                         --half setting was wrong for one of them
    confusions           which player gets mistaken for which; overlapping
                         home zones show up here immediately
    by confidence        confidence is only useful if high-confidence calls
                         really are more often right - this checks that

WHY THERE IS A HELD-OUT SPLIT
    Tuning weights while looking at the accuracy over all throws, then quoting
    that accuracy, measures nothing. `--holdout` splits the throws by a hash of
    their id (so the split is identical every run) into a part you may tune
    against and a part you only read at the end.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import labels as labels_module
from . import logs


def bucket_of(throw_id: str, holdout: float, seed: int) -> str:
    """Deterministic 'tune' / 'held out' assignment for one throw."""
    if holdout <= 0:
        return "held out"
    digest = hashlib.md5(f"{seed}:{throw_id}".encode()).hexdigest()
    return "held out" if int(digest[:8], 16) / 0xFFFFFFFF < holdout else "tune"


def _team_of(player_config: dict, pid):
    """The team a player id belongs to, according to the gallery roster."""
    if pid is None:
        return None
    return player_config.get(int(pid), {}).get("team")


def report(title: str, items: list, player_config: dict, team_size: int) -> dict:
    """
    Print one accuracy block. `items` is a list of (throw_id, label_row, prediction).
    Returns the headline numbers so a caller can compare blocks.
    """
    logs.section(title)
    if not items:
        logs.info("  no throws in this group")
        return {}

    total = len(items)
    with_call = 0
    thrower_right = team_right = 0
    within_team_right = within_team_total = 0
    per_half: dict = {}
    confusions: dict = {}
    confident_right = confident_total = 0
    unsure_right = unsure_total = 0

    for throw_id, label, prediction in items:
        truth_id = labels_module.player_id(label)
        truth_team = _team_of(player_config, truth_id)
        predicted_id = prediction.get("thrower_id")
        predicted_team = prediction.get("thrower_team") \
            or _team_of(player_config, predicted_id)
        half = prediction.get("half", "?")
        half_row = per_half.setdefault(half, [0, 0])
        half_row[1] += 1

        if predicted_id is None:
            continue
        with_call += 1
        team_ok = (predicted_team == truth_team)
        player_ok = (int(predicted_id) == int(truth_id))
        team_right += int(team_ok)
        thrower_right += int(player_ok)
        half_row[0] += int(player_ok)
        if team_ok:
            within_team_total += 1
            within_team_right += int(player_ok)
        if not player_ok:
            key = (label.get("thrower_name") or str(truth_id),
                   prediction.get("thrower_name") or str(predicted_id))
            confusions[key] = confusions.get(key, 0) + 1
        if float(prediction.get("confidence") or 0) >= 0.7:
            confident_total += 1
            confident_right += int(player_ok)
        else:
            unsure_total += 1
            unsure_right += int(player_ok)

    logs.kv("labelled throws", str(total))
    logs.kv("of those, predicted", f"{with_call} ({logs.percent(with_call, total)})")
    logs.blank()
    logs.kv("CORRECT THROWER", f"{thrower_right}/{total} = "
                               f"{logs.percent(thrower_right, total)}")
    logs.kv("correct team", f"{team_right}/{with_call} = "
                            f"{logs.percent(team_right, with_call)}")
    logs.kv("right player, given", f"{within_team_right}/{within_team_total} = "
                                   f"{logs.percent(within_team_right, within_team_total)}")
    logs.kv("  the right team", "")
    logs.kv("guessing would give", f"{100.0 / max(1, team_size):.1f}%")

    if len(per_half) > 1:
        logs.blank()
        logs.info("  by half:")
        rows = [[f"half {h}", f"{c}/{t}", logs.percent(c, t)]
                for h, (c, t) in sorted(per_half.items(), key=lambda kv: str(kv[0]))]
        logs.table(["half", "correct", "accuracy"], rows, indent="      ")
        values = [c / t for c, t in per_half.values() if t]
        if len(values) > 1 and (max(values) - min(values)) > 0.25:
            logs.warn("one half is far worse than the other",
                      "that is the signature of a wrong --half setting: check "
                      "which half the gallery was tagged in, and whether the "
                      "video really changes ends where you told it")

    if confident_total or unsure_total:
        logs.blank()
        logs.info("  is the confidence score meaningful?")
        logs.table(["confidence", "correct", "accuracy"], [
            ["0.70 and above", f"{confident_right}/{confident_total}",
             logs.percent(confident_right, confident_total)],
            ["below 0.70", f"{unsure_right}/{unsure_total}",
             logs.percent(unsure_right, unsure_total)],
        ], indent="      ")

    if confusions:
        logs.blank()
        logs.info("  most common mix-ups (actual thrower -> predicted):")
        for (actual, guessed), count in sorted(confusions.items(),
                                               key=lambda kv: -kv[1])[:6]:
            logs.bullet(f"{actual} mistaken for {guessed}: {count} time(s)")

    return {
        "total": total, "predicted": with_call,
        "thrower_accuracy": thrower_right / total if total else 0.0,
        "team_accuracy": team_right / with_call if with_call else 0.0,
        "within_team_accuracy": (within_team_right / within_team_total
                                 if within_team_total else 0.0),
    }
