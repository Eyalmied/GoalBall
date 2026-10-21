"""
The predictor: one throw in, one named thrower out.

Two things have to be worked out, and they use different signals:

    WHERE the throw came from   the player the goalball detector labels as the
                                thrower; failing that, the player picked out
                                of everyone on court by movement and posture

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

WHEN THE DETECTOR FINDS NOBODY  (--trust-position-always turns this off)
    The crop then comes from the movement-and-posture rule, which is not
    reliably the thrower - so where that player was standing says nothing
    about who threw. Measured on labelled throws it lands on chance: 34.6%
    against 33.3% for guessing. Blending a signal that is known to be noise
    only drags the answer around, so those throws fall back to the attacking
    team and let appearance alone name the player: 50.0% instead of 34.6%.

WHERE THE TEAM IN THE ANSWER COMES FROM  (--team-source)
    "position" (the default, whenever the gallery has home zones)
        The thrower is named against ALL the players' home zones, and the
        team is simply whichever team the winner plays for. The two teams
        occupy opposite ends of the court, so a release point near one end
        cannot look like a player at the other, and no separate team step is
        needed at all.

    "posture"
        Decide the attacking team from posture, then name the thrower among
        that team's three players only. Used automatically when the gallery
        has no home zones, since appearance alone cannot cope with six
        candidates.

    Posture is computed either way. It is what picks the thrower out of the
    players on court when the detector misses, it feeds the confidence, and
    it is reported alongside every prediction so the two signals can be
    compared.
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

    # picking out the thrower. The weights are compared WITHIN each frame:
    # every candidate is ranked against the others, so a camera that is closer
    # or noisier than yesterday's does not change the answer.
    weight_motion: float = 0.4        # the thrower moves; everyone else waits
    weight_upright: float = 0.6       # the thrower is on their feet
    motion_gap: int = 2               # frames apart the motion is measured over

    # naming that player
    # A goalball-trained detector that labels the thrower directly. When one
    # is available it is asked first; the motion rule below catches the throws
    # it misses. 73% correct thrower against 46% for motion alone.
    thrower_model: Optional[str] = None
    thrower_confidence: float = 0.15

    team_source: str = "position"     # "position", or "posture" without zones
    # 0.60 = mostly position, with appearance as a real second opinion. Once
    # the embedding is sharpened (sharpen_appearance below) appearance carries
    # enough signal to overrule a confident but wrong position call; without
    # that sharpening the best value is nearer 0.75.
    weight_position: float = 0.75
    position_spread: float = 0.12     # how far a player may stray from their home zone
    appearance_top_k: int = 3
    flip_augment: bool = True
    # Learn, from the gallery's own labels, which directions of the embedding
    # separate players - and ignore the rest. See goalball/reid.py.
    # Court position only means something if the crop really is the thrower.
    # With no detector hit it does not, so position is dropped for that throw.
    position_needs_detector: bool = True
    sharpen_appearance: bool = True
    appearance_dims: int = 3

    # confidence shaping
    confidence_temperature: float = 0.15

    # models
    pose_model: str = "yolov8m-pose.pt"
    appearance_model: str = "osnet"

    def describe(self) -> None:
        logs.kv("frames per throw", f"{self.frames_per_throw} "
                                    f"(over the first {self.window_fraction:.0%} "
                                    f"of each throw)")
        logs.kv("processing width", f"{self.processing_width} px")
        logs.kv("appearance model", self.appearance_model)
        logs.kv("thrower position from", "where the thrower was standing "
                                         "in the video")
        logs.kv("identity weighting", f"{self.weight_position:.0%} court position, "
                                      f"{1 - self.weight_position:.0%} appearance")
        if self.position_needs_detector:
            logs.kv("if it finds nobody", "appearance alone names the "
                                          "player (position is at chance)")
        weights = [(self.weight_motion, "movement"),
                   (self.weight_upright, "standing up")]
        if self.thrower_model:
            logs.kv("thrower picked by", "the goalball detector's own "
                                         "'throwing_player' class")
            logs.kv("when it finds nobody", "  ".join(f"{name} {w:.0%}"
                                                      for w, name in weights if w))
        else:
            logs.kv("thrower picked by", "  ".join(f"{name} {w:.0%}"
                                                   for w, name in weights if w))
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
    by_team: dict = field(default_factory=dict)
    by_half: dict = field(default_factory=dict)
    confidences: list = field(default_factory=list)
    end_agreements: int = 0
    end_checked: int = 0
    reasons: dict = field(default_factory=dict)


class Predictor:
    """Holds the models and reference data; call `predict` once per throw."""

    def __init__(self, cap, video_info, court, half_plan, player_config,
                 gallery_crops, settings: Settings, crops_by_half=None):
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
        # One bank per half where the gallery has images for it, so a throw is
        # always compared against how the players looked in that half.
        self.banks = (reid.banks_by_half(self.embedder, crops_by_half,
                                         flip_augment=settings.flip_augment)
                      if crops_by_half else {})
        self.reference_vectors, self.reference_labels = reid.reference_bank(
            self.embedder, gallery_crops, flip_augment=settings.flip_augment)
        self.projection = self._learn_projection()

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
        self.totals = RunTotals()

    def _learn_projection(self):
        """
        Work out which directions of the embedding separate these players.

        Fitted across every half at once, because the answer is about who the
        players are, not where they were standing.
        """
        s = self.settings
        if not s.sharpen_appearance or s.weight_position >= 1.0:
            return None
        if self.banks:
            vectors = np.concatenate([v for v, _ in self.banks.values()])
            labels = np.concatenate([l for _, l in self.banks.values()])
        else:
            vectors, labels = self.reference_vectors, self.reference_labels

        before = reid.separation(vectors, labels)
        projection = reid.fit_discriminant(vectors, labels, s.appearance_dims)
        if projection is None:
            logs.warn("the gallery is too thin to learn what separates the "
                      "players, so appearance stays unsharpened",
                      "give every player at least two reference images")
            return None

        after = reid.separation(reid.project(vectors, projection), labels)
        logs.ok(f"learned {projection.shape[1]} direction(s) that tell these "
                f"players apart")
        logs.kv("how distinct they look", f"{before:.2f} before, {after:.2f} "
                                          f"after  (above 1 is usable)")
        if after < 1.0:
            logs.warn("even after sharpening, two photos of one player differ "
                      "more than two players do",
                      "appearance will contribute little - consider "
                      "--position-weight 1.0, or better reference images")
        return projection

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

        # -- look at several frames across the start of the throw ----------
        start, end = throw.window(s.window_fraction)
        observed = []
        for index in np.linspace(start, end, s.frames_per_throw, dtype=int):
            frame = read_frame(self.cap, int(index), s.processing_width)
            if frame is None:
                continue
            # straight after read_frame the capture sits at index + 1, so the
            # motion read costs no extra seek
            motion = (detect.motion_between(self.cap, frame,
                                            s.processing_width, s.motion_gap)
                      if s.weight_motion else None)
            found = detect.people(frame, self.court, s.pose_model)
            for person in found:
                person["motion"] = detect.motion_in_box(motion, person["box"])
            teams.group_relatively(found, self.kit_prototypes,
                                   self.team_sizes)
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
        # Two independent opinions per frame, kept apart on purpose. Mixing
        # them is the trap: if the detector names the thrower in two frames
        # and the motion rule guesses someone else in the other three, the
        # median of all five positions lands between two players and means
        # nothing. So the detector's frames win outright whenever it has any.
        named_crops, named_positions = [], []
        guessed_crops, guessed_positions = [], []
        for index, found in observed:
            if s.thrower_model:
                frame = read_frame(self.cap, index, s.processing_width)
                named = (detect.throwing_players(
                    frame, self.court, s.thrower_model, s.thrower_confidence)
                    if frame is not None else [])
                if named:
                    best = max(named, key=lambda p: p["confidence"])
                    named_crops.append(best["crop"])
                    named_positions.append(best["feet_norm"])
                    continue
            candidates = found
            if not candidates:
                continue
            costs = self._localisation_costs(candidates)
            best = candidates[int(np.argmin(costs))]
            guessed_crops.append(best["crop"])
            guessed_positions.append(best["feet_norm"])

        if named_crops:
            thrower_crops, thrower_positions = named_crops, named_positions
            located_by = "the goalball detector"
        else:
            thrower_crops, thrower_positions = guessed_crops, guessed_positions
            located_by = "movement and posture"
        base["frames_from_detector"] = len(named_crops)
        base["located_by"] = located_by

        if not thrower_crops:
            return self._unresolved(
                base, "no player could be picked out in this window")

        # -- where was the throw released from? ----------------------------
        release, release_from = self._release_position(thrower_positions)
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

        # When the goalball detector found nobody, the crop came from movement
        # and posture, which is not reliably the thrower. Where that player was
        # standing therefore says nothing about who threw - measured on labelled
        # throws it lands on chance (34.6% against 33.3%). Blending a signal we
        # know is noise only drags the answer around, so fall back to the
        # attacking team and let appearance alone name the player.
        weight = s.weight_position
        if s.position_needs_detector and not named_crops:
            team_only = sorted(self.players_by_team.get(attacker_team, []))
            if team_only:
                candidates = team_only
                base["team_from"] = "posture"
                weight = 0.0

        appearance = self._appearance_scores(thrower_crops, candidates, half)
        scores, how = self._identity_scores(candidates, appearance, zones,
                                            release, weight)
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

    @staticmethod
    def _spread(values) -> np.ndarray:
        """
        Scale a list to 0..1 against the others in the same frame.

        Everything is judged relatively, never against a fixed threshold. How
        much a player's pixels change depends on the camera, the compression
        and how near they are; which of the players on court changed MOST does
        not. The same goes for how flat a detection box is. When every
        candidate ties, nobody is favoured.
        """
        array = np.asarray(values, dtype=float)
        low, high = array.min(), array.max()
        if high - low < 1e-9:
            return np.full(len(array), 0.5)
        return (array - low) / (high - low)

    def _localisation_costs(self, candidates: list) -> np.ndarray:
        """
        Lower is more likely to be the thrower, one score per candidate.

        The thrower moves and is on their feet; the defenders are already down
        and the thrower's own team-mates are standing still. Those two signals
        do nearly all the work - see docs/RESULTS.md for what each contributes.
        """
        s = self.settings
        moving = self._spread([c.get("motion", 0.0) for c in candidates])
        # a low box width/height means upright, which is what we want
        upright = 1.0 - self._spread([c["flatness"] for c in candidates])
        return (s.weight_motion * (1.0 - moving)
                + s.weight_upright * (1.0 - upright))

    def _release_position(self, thrower_positions) -> tuple:
        """The point used to name the thrower, in normalised frame coordinates."""
        if not thrower_positions:
            return None, "unavailable"
        array = np.asarray(thrower_positions, dtype=float)
        return ((float(np.median(array[:, 0])), float(np.median(array[:, 1]))),
                "video-localised thrower")

    def _appearance_scores(self, crops, candidates, half) -> dict:
        """
        Compare the thrower's crop against this half's references.

        Falls back to every reference the gallery has if that half was never
        tagged - worse than like-for-like, better than nothing.
        """
        vectors = reid.embed(self.embedder, crops,
                             flip_augment=self.settings.flip_augment)
        query = vectors.mean(axis=0)
        query /= (np.linalg.norm(query) + 1e-8)
        bank = self.banks.get(int(half))
        reference, labels = bank if bank else (self.reference_vectors,
                                               self.reference_labels)
        # The query and the references must go through the same projection.
        reference = reid.project(reference, self.projection)
        query = reid.project(query, self.projection)
        return reid.similarity_to_players(
            query, reference, labels, candidates,
            top_k=self.settings.appearance_top_k)

    def _identity_scores(self, candidates, appearance, zones, release,
                         weight=None) -> tuple:
        """Blend the position match with the appearance match."""
        s = self.settings
        weight = s.weight_position if weight is None else float(weight)
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
            scores[pid] = (weight * position
                           + (1.0 - weight) * appearance_norm)
        if weight >= 1.0:
            how = "court position"
        elif weight <= 0.0:
            how = "appearance alone (the detector did not find the thrower)"
        else:
            how = f"court position ({weight:.0%}) plus appearance"
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
