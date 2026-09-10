"""
The predictor: one throw in, one named thrower out.

Two things have to be worked out, and they use different signals:

    WHERE the throw came from   the ball's release point, or failing that the
                                player picked out of the attacking team by
                                proximity, forwardness and throwing pose

    WHOSE home zone that is     matched against the zones recorded when the
                                gallery was tagged, mirrored if this half is
                                not the half they were tagged in

The team is not a third decision - it falls out of the name, because the two
teams stand at opposite ends of the court. Posture is still used to work out
which team is attacking, for the cases that need it and as a cross-check.

Everything is decided several times, on several frames across the start of the
throw, and voted - so one blurred or blocked frame cannot flip the answer.

Every prediction carries a confidence built from three readable parts, so weak
calls can be pulled out and reviewed instead of silently trusted.

WHERE THE TEAM IN THE ANSWER COMES FROM  (--team-source)
    "position" (the default, whenever the gallery has home zones)
        The thrower is named against ALL the players' home zones, and the
        team is simply whichever team the winner plays for. The two teams
        occupy opposite ends of the court, so a release point near one end
        cannot look like a player at the other, and no separate team step is
        needed at all.
        Measured on the reference match: 97.6% correct team, 74.1% correct
        thrower - better than restricting the candidates first (71.8%), and
        better even than restricting them with a PERFECT team step (72.9%),
        because a wrong restriction removes the right answer entirely.

    "posture"
        The older behaviour: decide the attacking team from posture, then name
        the thrower among that team's three players only. Necessary when the
        gallery has no home zones, since appearance alone cannot cope with six
        candidates. It is also the weak link across halves - see
        goalball/teams.py - which is why it is no longer the default.

    Posture is computed either way. It is what picks the thrower out of the
    players on court for the video-only release source, it feeds the
    confidence, and it is reported alongside every prediction so the two
    signals can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import detect, logs, reid, teams
from .video import read_frame


@dataclass
class Settings:
    """Everything tunable about a run. Defaults are the measured-best values."""

    # what to look at
    frames_per_throw: int = 5
    window_fraction: float = 0.6      # only the first 60% of the window: the release
    processing_width: int = 1280

    # picking the thrower out of the attacking team
    weight_ball: float = 0.5          # how close to where the ball came from
    weight_forward: float = 0.3       # throwers step in toward the centre
    weight_pose: float = 0.2          # how much the body looks mid-throw
    use_ball: bool = True

    # naming that player
    team_grouping: str = "relative"   # "relative" (robust) or "absolute"
    team_source: str = "position"     # "position" (robust) or "posture"
    release_source: str = "csv"       # "csv" (ball release point) or "thrower"
    weight_position: float = 1.0      # 1.0 = position only; appearance measured to hurt
    position_spread: float = 0.12     # how far a player may stray from their home zone
    appearance_top_k: int = 3
    flip_augment: bool = True

    # confidence shaping
    confidence_temperature: float = 0.15

    # models
    pose_model: str = "yolov8m-pose.pt"
    ball_model: str = "yolov8m.pt"
    appearance_model: str = "osnet"

    def describe(self) -> None:
        logs.kv("frames per throw", f"{self.frames_per_throw} "
                                    f"(over the first {self.window_fraction:.0%} "
                                    f"of each throw)")
        logs.kv("processing width", f"{self.processing_width} px")
        logs.kv("appearance model", self.appearance_model)
        if self.release_source == "csv":
            logs.kv("thrower position from", "the CSV ball release point "
                                             "(most accurate: ~74% correct)")
        else:
            logs.kv("thrower position from", "the video only, no CSV fields "
                                             "beyond timestamps (~50% correct)")
        logs.kv("identity weighting", f"{self.weight_position:.0%} court position, "
                                      f"{1 - self.weight_position:.0%} appearance")
        logs.kv("ball detection", "on" if self.use_ball else "off (pose only)")
        logs.kv("team grouping", "by colour, ranked within each frame"
                                 if self.team_grouping == "relative"
                                 else "by colour, each crop on its own")
        if self.team_source == "position":
            logs.kv("team in the answer", "from the named player's own team "
                                          "(no separate team step)")
        else:
            logs.kv("team in the answer", "from posture, and only that team's "
                                          "players are considered")


@dataclass
class RunTotals:
    """Counters for the closing summary."""

    predicted: int = 0
    unresolved: int = 0
    ball_seen: int = 0
    by_team: dict = field(default_factory=dict)
    by_half: dict = field(default_factory=dict)
    confidences: list = field(default_factory=list)
    end_agreements: int = 0
    end_checked: int = 0
    reasons: dict = field(default_factory=dict)


class Predictor:
    """Holds the models and reference data; call `predict` once per throw."""

    def __init__(self, cap, video_info, court, half_plan, player_config,
                 gallery_crops, settings: Settings):
        self.cap = cap
        self.video = video_info
        self.court = court
        self.halves = half_plan
        self.player_config = player_config
        self.settings = settings

        self.team_of = {pid: info.get("team")
                        for pid, info in player_config.items()}
        self.players_by_team: dict = {}
        for pid, team in self.team_of.items():
            self.players_by_team.setdefault(team, []).append(pid)
        self.team_sizes = {team: len(players)
                           for team, players in self.players_by_team.items()}

        self.kit_prototypes = teams.build_prototypes(gallery_crops, player_config)
        teams.check_prototypes(gallery_crops, player_config, self.kit_prototypes)

        self.embedder = reid.make(settings.appearance_model)
        self.reference_vectors, self.reference_labels = reid.reference_bank(
            self.embedder, gallery_crops, flip_augment=settings.flip_augment)

        # Home zones, resolved once per half rather than once per throw.
        self.zones_by_half = {
            half: half_plan.home_zones(player_config, court, half)
            for half in half_plan.halves_in_play
        }
        self.ends_by_half = {
            half: half_plan.team_ends(player_config, court, half)
            for half in half_plan.halves_in_play
        }
        self.have_zones = all(bool(z) for z in self.zones_by_half.values())

        self.proc_h = video_info.proc_height(settings.processing_width)
        self.frame_diagonal = float(np.hypot(settings.processing_width, self.proc_h))
        self.court_centre_px = (
            self.court.centre_norm[0] * settings.processing_width,
            self.court.centre_norm[1] * self.proc_h,
        )
        self.totals = RunTotals()

    # ------------------------------------------------------------------
    # One throw
    # ------------------------------------------------------------------

    def predict(self, throw) -> dict:
        s = self.settings
        half = self.halves.half_of_frame(throw.start_frame)
        zones = self.zones_by_half.get(half, {})
        mirrored = self.halves.mirrors_for_half(half)

        base = {
            "throw_id": throw.throw_id,
            "start_time": throw.start_mmss,
            "half": half,
            "home_zones_mirrored": mirrored,
            "csv_team_hint": throw.csv_team_hint,
        }

        # -- where did the ball come from? (from the video, never the CSV) --
        ball_point = None
        if s.use_ball:
            ball_point = detect.ball_release_point(
                self.cap, throw, self.court, s.ball_model,
                s.processing_width, s.window_fraction)
        base["ball_seen"] = ball_point is not None

        # -- look at several frames across the start of the throw ----------
        start, end = throw.window(s.window_fraction)
        observed = []
        for index in np.linspace(start, end, s.frames_per_throw, dtype=int):
            frame = read_frame(self.cap, int(index), s.processing_width)
            if frame is None:
                continue
            found = detect.people(frame, self.court, s.pose_model)
            if s.team_grouping == "relative":
                teams.group_relatively(found, self.kit_prototypes,
                                       self.team_sizes)
            else:
                for person in found:
                    person["team"], person["team_lean"] = teams.team_of_crop(
                        person["crop"], self.kit_prototypes)
            if found:
                observed.append((int(index), found))

        if not observed:
            return self._unresolved(base, "nobody was detected on court in this window")

        everyone = [p for _, found in observed for p in found]
        attacker_team, flatness, separation = teams.attacking_team(everyone)
        base["team_flatness"] = {t: round(v, 3) for t, v in flatness.items()}

        if attacker_team is None:
            return self._unresolved(
                base, "only one team was visible, so the attacking team is unknown")

        base["attacking_team"] = attacker_team

        # -- pick the thrower out of the attacking team, frame by frame ----
        thrower_crops, thrower_positions = [], []
        for _, found in observed:
            attackers = [p for p in found if p.get("team") == attacker_team]
            if not attackers:
                continue
            for person in attackers:
                person["cost"] = self._localisation_cost(person, ball_point)
            best = min(attackers, key=lambda p: p["cost"])
            thrower_crops.append(best["crop"])
            thrower_positions.append(best["feet_norm"])

        if not thrower_crops:
            return self._unresolved(
                base, f"no {attacker_team} player was detected in this window")

        # -- where was the throw released from? ----------------------------
        release, release_from = self._release_position(throw, thrower_positions)
        base["release_position"] = ([round(release[0], 4), round(release[1], 4)]
                                    if release else None)
        base["release_position_from"] = release_from
        if release:
            # Independent cross-check: a throw should come from the attacking
            # team's own end. This never changes the team decision (posture
            # owns that) - it only reports when the two signals disagree.
            nearest = self.halves.team_nearest_to(
                self.player_config, self.court, half, release)
            if nearest:
                base["ball_came_from_team_end"] = nearest
                base["attacking_team_end"] = \
                    self.ends_by_half.get(half, {}).get(attacker_team)
                base["end_agrees_with_half"] = (nearest == attacker_team)

        # -- name that player ----------------------------------------------
        # With home zones, every player is a candidate and the winner's team is
        # the answer: the two teams stand at opposite ends, so the release point
        # settles both questions at once. Restricting the candidates first can
        # only remove the right answer.
        by_position = (s.team_source == "position"
                       and release is not None
                       and all(pid in zones for pid in self.player_config))
        if by_position:
            candidates = sorted(self.player_config)
            base["team_from"] = "the named player's own team"
        else:
            candidates = sorted(self.players_by_team.get(attacker_team, []))
            base["team_from"] = "posture"
        if not candidates:
            return self._unresolved(
                base, f"team '{attacker_team}' is not in the gallery roster")

        appearance = self._appearance_scores(thrower_crops, candidates)
        scores, how = self._identity_scores(candidates, appearance, zones, release)
        chosen = int(max(scores, key=scores.get))

        # -- confidence -----------------------------------------------------
        identity_confidence = self._margin(list(scores.values()))
        coverage = min(len(thrower_crops) / max(1, s.frames_per_throw), 1.0)
        confidence = round(
            identity_confidence * (0.4 + 0.4 * separation + 0.2 * coverage), 3)

        record = dict(base)
        record.update({
            "thrower_id": chosen,
            "thrower_name": self.player_config[chosen]["name"],
            "thrower_team": self.team_of.get(chosen),
            "confidence": confidence,
            "confidence_parts": {
                "identity_margin": round(identity_confidence, 3),
                "team_separation": round(separation, 3),
                "frame_coverage": round(coverage, 3),
            },
            "frames_voted": len(thrower_crops),
            "identity_scores": {str(p): round(v, 4) for p, v in scores.items()},
            "appearance_scores": {str(p): round(v, 4) for p, v in appearance.items()},
            "decided_by": how,
            "resolved": True,
        })
        self._count(record)
        return record

    # ------------------------------------------------------------------
    # Pieces
    # ------------------------------------------------------------------

    def _localisation_cost(self, person, ball_point) -> float:
        """
        Lower is more likely to be the thrower. Three costs, all scaled to the
        frame diagonal so they are comparable and resolution-independent.
        """
        s = self.settings
        x, y = person["feet"]
        if ball_point is None:
            ball_cost = 0.5             # unknown: neither helps nor hurts anyone
        else:
            ball_cost = min(np.hypot(x - ball_point[0], y - ball_point[1])
                            / self.frame_diagonal, 1.0)
        forward_cost = min(np.hypot(x - self.court_centre_px[0],
                                    y - self.court_centre_px[1])
                           / self.frame_diagonal, 1.0)
        pose_cost = 1.0 - person["throw_pose"]
        return (s.weight_ball * ball_cost
                + s.weight_forward * forward_cost
                + s.weight_pose * pose_cost)

    def _release_position(self, throw, thrower_positions) -> tuple:
        """The point used to name the thrower, in normalised frame coordinates."""
        if self.settings.release_source == "csv":
            point = throw.release_point()
            if point:
                return ((point[0] / self.video.width, point[1] / self.video.height),
                        "csv ball release point")
        if not thrower_positions:
            return None, "unavailable"
        array = np.asarray(thrower_positions, dtype=float)
        return ((float(np.median(array[:, 0])), float(np.median(array[:, 1]))),
                "video-localised thrower")

    def _appearance_scores(self, crops, candidates) -> dict:
        vectors = reid.embed(self.embedder, crops,
                             flip_augment=self.settings.flip_augment)
        query = vectors.mean(axis=0)
        query /= (np.linalg.norm(query) + 1e-8)
        return reid.similarity_to_players(
            query, self.reference_vectors, self.reference_labels, candidates,
            top_k=self.settings.appearance_top_k)

    def _identity_scores(self, candidates, appearance, zones, release) -> tuple:
        """Blend the position match with the appearance match."""
        s = self.settings
        usable_zones = release is not None and all(p in zones for p in candidates)
        if not usable_zones:
            return dict(appearance), "appearance only (no home zones available)"

        lo, hi = min(appearance.values()), max(appearance.values())
        spread = (hi - lo) or 1e-9
        scores = {}
        for pid in candidates:
            zx, zy = zones[pid]
            distance_sq = (release[0] - zx) ** 2 + (release[1] - zy) ** 2
            position = float(np.exp(-0.5 * distance_sq / (s.position_spread ** 2)))
            appearance_norm = (appearance[pid] - lo) / spread
            scores[pid] = (s.weight_position * position
                           + (1.0 - s.weight_position) * appearance_norm)
        if s.weight_position >= 1.0:
            how = "court position"
        elif s.weight_position <= 0.0:
            how = "appearance"
        else:
            how = f"court position ({s.weight_position:.0%}) plus appearance"
        return scores, how

    def _margin(self, values) -> float:
        """How clearly the winner won, as a 0..1 softmax share of the top score."""
        ordered = np.sort(np.asarray(values, dtype=float))[::-1]
        if len(ordered) == 1:
            return 1.0
        weights = np.exp((ordered - ordered.max())
                         / max(self.settings.confidence_temperature, 1e-3))
        return float(weights[0] / weights.sum())

    def _unresolved(self, base, reason) -> dict:
        record = dict(base)
        record.update({
            "thrower_id": None, "thrower_name": None, "thrower_team": None,
            "confidence": 0.0, "frames_voted": 0,
            "decided_by": "not decided", "reason": reason, "resolved": False,
        })
        self.totals.unresolved += 1
        self.totals.reasons[reason] = self.totals.reasons.get(reason, 0) + 1
        self.totals.by_half[base["half"]] = self.totals.by_half.get(base["half"], 0) + 1
        return record

    def _count(self, record) -> None:
        t = self.totals
        t.predicted += 1
        t.ball_seen += int(bool(record.get("ball_seen")))
        t.confidences.append(record["confidence"])
        team = record.get("thrower_team") or "?"
        t.by_team[team] = t.by_team.get(team, 0) + 1
        t.by_half[record["half"]] = t.by_half.get(record["half"], 0) + 1
        if "end_agrees_with_half" in record:
            t.end_checked += 1
            t.end_agreements += int(record["end_agrees_with_half"])


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def log_one(record: dict, index: int, total: int) -> None:
    """One tidy line per throw - the thing the user actually watches."""
    tag = f"[{index}/{total}]".ljust(10)
    when = record["start_time"].rjust(5)
    if not record.get("resolved"):
        logs.info(f"{tag} {when}  {logs.yellow('no call')}  "
                  f"{logs.dim(record.get('reason', ''))}")
        return
    who = f"{record['thrower_name']} ({record['thrower_team']})".ljust(16)
    notes = [f"{record['frames_voted']} frames agreed"]
    notes.append("ball seen" if record.get("ball_seen") else "ball not seen")
    if record.get("end_agrees_with_half") is False:
        notes.append(logs.yellow("but the ball came from the other team's end"))
    logs.info(f"{tag} {when}  {who} {logs.bar(record['confidence'])} "
              f"{record['confidence']:.2f}  {logs.dim(' · '.join(notes))}")


def log_summary(totals: RunTotals, output_path: str) -> None:
    logs.section("SUMMARY")
    total = totals.predicted + totals.unresolved
    logs.kv("throws with a thrower", f"{totals.predicted} of {total}")
    if totals.unresolved:
        logs.kv("throws with no call", str(totals.unresolved))
        for reason, count in sorted(totals.reasons.items(), key=lambda kv: -kv[1]):
            logs.bullet(f"{count} x {reason}")

    if totals.confidences:
        confidences = np.asarray(totals.confidences)
        strong = int((confidences >= 0.7).sum())
        weak = int((confidences < 0.4).sum())
        logs.kv("average confidence", f"{confidences.mean():.2f}")
        logs.kv("confident calls", f"{strong} at 0.70 or above "
                                   f"({logs.percent(strong, len(confidences))})")
        logs.kv("worth reviewing", f"{weak} below 0.40 "
                                   f"({logs.percent(weak, len(confidences))})")
    if totals.by_team:
        logs.kv("throws per team", "   ".join(f"{t}: {n}" for t, n in
                                              sorted(totals.by_team.items())))
    if len(totals.by_half) > 1:
        logs.kv("throws per half", "   ".join(f"half {h}: {n}" for h, n in
                                              sorted(totals.by_half.items())))
    logs.kv("ball found in video", f"{totals.ball_seen} of {totals.predicted} throws")
    if totals.end_checked:
        agree = totals.end_agreements
        logs.kv("posture vs the end",
                f"posture and the end the ball came from name the same "
                f"attacking team on {agree}/{totals.end_checked} throws "
                f"({logs.percent(agree, totals.end_checked)})")
        if agree < 0.75 * totals.end_checked:
            logs.warn("those two signals disagree more often than expected",
                      "two things cause this. Either --half does not match this "
                      "footage - check it, and check which half the gallery was "
                      "tagged in. Or the kit-colour grouping is struggling, "
                      "which happens when one team is far from the camera; the "
                      "team in the answer comes from court position and is not "
                      "affected, but a video-only run (--release-source "
                      "thrower) would be")
    logs.kv("written to", output_path)
