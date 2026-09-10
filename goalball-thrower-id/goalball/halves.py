"""
Which half of the match are we looking at?

WHY THIS MATTERS
----------------
Identity in this pipeline is decided by WHERE a player stands. Each player has
a canonical "home zone" on court, recorded once when the reference gallery is
tagged, and the thrower is named by matching the release position to the
nearest home zone.

At half-time the teams change ends. Every home zone therefore moves to the
diametrically opposite point of the court: the team that was at the TOP of the
frame is now at the BOTTOM, and left/right swap with it. If the run is told the
wrong half, every position match is compared against the far end of the court
and the naming collapses - it does not degrade gracefully, it inverts.

So the half is an explicit input to a run, set by the user up front:

    --half 1        the whole run is the half the gallery was tagged in
    --half 2        the whole run is the other half - positions are mirrored
    --half auto     the video spans both halves; switch at --half2-start

WHAT "MIRRORED" MEANS
---------------------
A half-turn on the flat court plane, so it stays correct under the camera's
perspective. See goalball/court.py.

WHAT IT TURNS ABOUT
-------------------
Not, by default, the middle of the court polygon - the middle of the RECORDED
HOME ZONES.

Three players a side hold zones that are symmetric about the centre spot, so in
an ideal world the two centres are the same point and the choice does not
matter. In practice they are not: a clicked court polygon usually reaches a
little past where anyone actually stands, and a release point measured by the
upstream detector need not sit in exactly the same coordinate convention as a
box you drew by hand. Turning about the middle of the polygon then throws every
mirrored zone too far down the court.

Turning about the zones' own centre is immune to both, and is identical to the
court centre whenever the zones really are symmetric about it. On the reference
match it agrees with where the teams are actually measured to be in the second
half to within a few pixels, against tens of pixels for the polygon centre.
Pass mirror_about="court" to use the geometric centre instead.

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
ONLY THE POSITIONS CHANGE. The players are the same players.

    changes    each player's home zone: the (x, y) point moves to the opposite
               end of the court
    unchanged  the player ids, the names, and which team each player is on -
               nothing is renumbered, renamed or reassigned
    unchanged  the court fence: it is the same physical court in both halves
    unchanged  the reference images: a player looks the same in both halves

So player 3 is still player 3, still on the same team, still with the same
reference images. The only thing the half decides is WHERE player 3 is expected
to be standing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import logs
from .court import Court

HALF_CHOICES = ("1", "2", "auto")


@dataclass
class HalfPlan:
    """The half setting for one run, resolved against the gallery's half."""

    gallery_half: int              # half the home zones were recorded in
    mode: str                      # "1", "2" or "auto"
    half2_start_frame: Optional[int] = None
    half2_start_text: Optional[str] = None
    mirror_about: str = "zones"    # "zones" (default) or "court"
    _centre: Optional[tuple] = None

    # -- which half is a given moment in? --------------------------------
    def half_of_frame(self, frame_index: int) -> int:
        if self.mode == "auto":
            if self.half2_start_frame is None:
                return 1
            return 2 if frame_index >= self.half2_start_frame else 1
        return int(self.mode)

    def mirrors_for_half(self, half: int) -> bool:
        """True when this half's court is the mirror of the gallery's half."""
        return int(half) != int(self.gallery_half)

    @property
    def halves_in_play(self) -> list:
        return [1, 2] if self.mode == "auto" else [int(self.mode)]

    # -- positions -------------------------------------------------------
    # Everything below moves POSITIONS only. Player ids, names and teams are
    # read straight through from the roster and never altered by the half.
    @staticmethod
    def _recorded(player_config: dict) -> dict:
        """{player_id -> (x, y)} exactly as tagged, before any mirroring."""
        out = {}
        for pid, info in player_config.items():
            xy = info.get("home_zone") or info.get("position")
            if xy:
                out[pid] = (float(xy[0]), float(xy[1]))
        return out

    def mirror_centre(self, player_config: dict, court: Court) -> tuple:
        """The court-plane point the change of ends turns about."""
        if self._centre is not None:
            return self._centre
        centre = (0.5, 0.5)
        if self.mirror_about == "zones":
            recorded = self._recorded(player_config)
            teams = {player_config[pid].get("team") for pid in recorded}
            if len(recorded) >= 2 and len(teams) >= 2:
                plane = np.array([court.to_plane(x, y)
                                  for x, y in recorded.values()])
                centre = (float(plane[:, 0].mean()), float(plane[:, 1].mean()))
        self._centre = centre
        return centre

    def home_zones(self, player_config: dict, court: Court, half: int) -> dict:
        """
        {player_id -> (x, y)} home zones in normalised frame coordinates,
        valid for the requested half. Players with no recorded zone are absent.
        """
        recorded = self._recorded(player_config)
        if not self.mirrors_for_half(half):
            return recorded
        centre = self.mirror_centre(player_config, court)
        return {pid: court.mirror_norm(x, y, centre)
                for pid, (x, y) in recorded.items()}

    def team_bands(self, player_config: dict, court: Court, half: int) -> dict:
        """
        {team -> how far along the court that team's home zones sit}, for the
        requested half. 0 is the end at the top of the frame, 1 the bottom.
        """
        zones = self.home_zones(player_config, court, half)
        along: dict = {}
        for pid, (x, y) in zones.items():
            team = player_config[pid].get("team")
            if team:
                along.setdefault(team, []).append(court.to_plane(x, y)[1])
        return {team: float(np.mean(values)) for team, values in along.items()}

    def team_ends(self, player_config: dict, court: Court, half: int) -> dict:
        """
        {team -> 'top' | 'bottom'} for the requested half, derived from the
        recorded home zones. Nothing is hard-coded: the gallery tells us which
        end each team started at, and the half tells us whether to flip it.

        The two teams are compared against EACH OTHER rather than against the
        middle of the court, so this is right whatever coordinate convention
        the positions happen to use.
        """
        bands = self.team_bands(player_config, court, half)
        if len(bands) < 2:
            return {}
        nearest_top = min(bands, key=bands.get)
        return {team: ("top" if team == nearest_top else "bottom")
                for team in bands}

    def team_nearest_to(self, player_config: dict, court: Court, half: int,
                        point) -> Optional[str]:
        """
        Which team's end a point lies nearer to, in this half.

        Used as an independent cross-check on the posture-based team decision:
        a throw should come from the attacking team's own end. Measured as a
        distance to each team's own band, never against a fixed midpoint,
        because release points do not necessarily span the full court.
        """
        bands = self.team_bands(player_config, court, half)
        if len(bands) < 2 or point is None:
            return None
        _, v = court.to_plane(point[0], point[1])
        return min(bands, key=lambda team: abs(v - bands[team]))

    # -- reporting -------------------------------------------------------
    def describe(self, player_config: dict, court: Court) -> None:
        if self.mode == "auto":
            logs.kv("half", f"both (switching at {self.half2_start_text})")
        else:
            logs.kv("half", f"{self.mode} of the match")
        logs.kv("gallery tagged in", f"half {self.gallery_half}")
        if any(self.mirrors_for_half(h) for h in self.halves_in_play):
            centre = self.mirror_centre(player_config, court)
            source = ("the middle of the tagged home zones"
                      if self.mirror_about == "zones" else
                      "the middle of the court polygon")
            logs.kv("ends swap about", f"{source}  "
                                       f"(court plane {centre[0]:.2f}, {centre[1]:.2f})")
        for half in self.halves_in_play:
            flip = self.mirrors_for_half(half)
            ends = self.team_ends(player_config, court, half)
            where = ("   " + ",  ".join(f"{t} at the {side}"
                                        for t, side in sorted(ends.items()))
                     ) if ends else "   (no home zones recorded yet)"
            state = "home zones MIRRORED (teams changed ends)" if flip \
                else "home zones as tagged"
            logs.kv(f"half {half}", f"{state}{where}")


def build(gallery_half: int, mode: str, half2_start_text: Optional[str],
          fps: float, mirror_about: str = "zones") -> HalfPlan:
    """
    Turn the user's `--half` choice into a plan, complaining clearly about the
    combinations that cannot work.
    """
    mode = str(mode).strip().lower()
    if mode not in HALF_CHOICES:
        logs.fail(f"--half must be one of 1, 2 or auto (got '{mode}')",
                  "use --half 1 or --half 2 for a single half, or --half auto "
                  "with --half2-start MM:SS for a full-match video")

    start_frame = None
    if mode == "auto":
        if not half2_start_text:
            logs.fail("--half auto needs --half2-start MM:SS",
                      "give the video timestamp where the second half begins, "
                      "e.g. --half2-start 24:00; if your video is a single half, "
                      "use --half 1 or --half 2 instead")
        from .events import mmss_to_seconds
        start_frame = int(round(mmss_to_seconds(half2_start_text) * fps))

    if mirror_about not in ("zones", "court"):
        logs.fail(f"--mirror-about must be 'zones' or 'court' (got '{mirror_about}')")

    return HalfPlan(gallery_half=int(gallery_half), mode=mode,
                    half2_start_frame=start_frame,
                    half2_start_text=half2_start_text,
                    mirror_about=mirror_about)


def counts_by_half(plan: HalfPlan, throws) -> dict:
    """{half -> number of throws} for the run summary."""
    out = {}
    for throw in throws:
        half = plan.half_of_frame(throw.start_frame)
        out[half] = out.get(half, 0) + 1
    return out
