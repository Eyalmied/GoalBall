"""
The reference gallery: the one manual step, done once per game.

The user tags all players in a handful of frames. Each tag gives two things:

  * a CROP           - what that player looks like (feeds the appearance model)
  * a HOME ZONE      - where that player stands on court, taken as the
                       bottom-centre of the box (their feet) normalised by the
                       frame size

The home zone is the important half. Goalball players hold their zones, and
with identical uniforms and blackout eyeshades, position is the only signal
that reliably separates two teammates. Recording it at tagging time is what
breaks the chicken-and-egg problem: naming a position slot would otherwise need
appearance, which is exactly the part that does not work.

TAGGING HAPPENS IN COMPLETE ROUNDS
    A frame is saved only when every player has been tagged in it and the user
    confirms. Partial frames are thrown away whole. That way every player ends
    with the same number of crops and the same number of position samples, so
    no one is over- or under-represented.

THE HALF IS RECORDED WITH THE GALLERY
    Home zones are only meaningful together with the half they were tagged in,
    because the teams change ends at half-time. The manifest stores it, and
    every later run compares its own --half against it.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import cv2
import numpy as np

from . import logs

MANIFEST_NAME = "manifest.json"

PLAYER_COLOURS = {
    1: (255, 80, 0), 2: (255, 150, 0), 3: (200, 200, 0),
    4: (0, 255, 0), 5: (0, 200, 100), 6: (0, 150, 255),
}

CELL_W, CELL_H = 110, 220
LABEL_W = 190
SHEET_MAX_H = 1100


def colour_for(player_id: int) -> tuple:
    return PLAYER_COLOURS.get(int(player_id), (220, 220, 220))


# ----------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------

def save(gallery_dir: str, crops: dict, player_config: dict,
         home_zones: Optional[dict], gallery_half: int,
         frame_size: Optional[tuple] = None) -> None:
    """Write the crops and the manifest. Overwrites any previous gallery."""
    out = pathlib.Path(gallery_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "player_config": {str(k): {"name": v.get("name") or f"player {k}",
                                   "team": v.get("team") or "?"}
                          for k, v in player_config.items()},
        "crops": {},
        "gallery_half": int(gallery_half),
        "note": "home_zones are (x, y) as a fraction of the frame, measured at "
                "the player's feet, valid for the half named in gallery_half",
    }
    if frame_size:
        manifest["frame_size"] = [int(frame_size[0]), int(frame_size[1])]

    for pid in sorted(crops):
        files = []
        for i, crop in enumerate(crops[pid]):
            name = f"player_{pid}_{i}.jpg"
            cv2.imwrite(str(out / name), crop)
            files.append(name)
        manifest["crops"][str(pid)] = files

    if home_zones:
        zones = {}
        for pid, samples in home_zones.items():
            if samples:
                arr = np.asarray(samples, dtype=float)
                # median, not mean: one mis-drawn box cannot drag the zone away
                zones[str(pid)] = [float(np.median(arr[:, 0])),
                                   float(np.median(arr[:, 1]))]
        manifest["home_zones"] = zones

    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    total = sum(len(v) for v in crops.values())
    logs.ok(f"gallery saved to {out}/")
    logs.kv("reference images", f"{total} ({total // max(1, len(crops))} per player)")
    logs.kv("home zones", "recorded" if home_zones else logs.yellow("NOT recorded"))
    logs.kv("tagged in", f"half {gallery_half}")


def load(gallery_dir: str) -> tuple:
    """
    Load a gallery.

    Returns (crops, player_config, meta) where
        crops         {player_id -> [image, ...]}
        player_config {player_id -> {"name", "team", "home_zone"?}}
        meta          {"gallery_half": int, "has_home_zones": bool, ...}
    """
    folder = pathlib.Path(gallery_dir)
    manifest_path = folder / MANIFEST_NAME
    if not manifest_path.exists():
        logs.fail(f"no gallery found at {folder}/",
                  "run scripts/build_gallery.py once for this game - it is the "
                  "only manual step and takes a couple of minutes")
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        logs.fail(f"gallery manifest could not be read ({e})",
                  f"delete {manifest_path} and rebuild with scripts/build_gallery.py")

    player_config = {int(k): dict(v)
                     for k, v in manifest.get("player_config", {}).items()}
    if not player_config:
        logs.fail("the gallery manifest has no roster",
                  "rebuild the gallery with scripts/build_gallery.py")

    # "home_zones" is the current name; "positions" is accepted for galleries
    # built by the earlier scripts.
    zones = manifest.get("home_zones") or manifest.get("positions") or {}
    for pid_text, xy in zones.items():
        if int(pid_text) in player_config:
            player_config[int(pid_text)]["home_zone"] = [float(xy[0]), float(xy[1])]

    crops, unreadable = {}, 0
    for pid_text, files in manifest.get("crops", {}).items():
        images = []
        for name in files:
            image = cv2.imread(str(folder / name))
            if image is None:
                unreadable += 1
            else:
                images.append(image)
        crops[int(pid_text)] = images

    total = sum(len(v) for v in crops.values())
    counts = {pid: len(v) for pid, v in sorted(crops.items())}
    gallery_half = int(manifest.get("gallery_half", 1))
    has_zones = bool(zones)

    logs.ok(f"gallery loaded from {folder}/")
    logs.kv("players", ", ".join(f"{pid}:{player_config[pid]['name']}"
                                 f" ({player_config[pid]['team']})"
                                 for pid in sorted(player_config)))
    logs.kv("reference images", f"{total} total  " +
            "  ".join(f"{pid}:{n}" for pid, n in counts.items()))
    if unreadable:
        logs.warn(f"{unreadable} reference image(s) listed in the manifest are missing",
                  "rebuild the gallery, or the affected players will be harder to match")
    if not has_zones:
        logs.warn("this gallery has no home zones, so players cannot be told "
                  "apart by court position",
                  "rebuild it with scripts/build_gallery.py - position is what "
                  "lifts correct-thrower from ~40% to ~70%")
    if "gallery_half" not in manifest:
        logs.warn("this gallery does not say which half it was tagged in - "
                  "assuming half 1",
                  "rebuild it with scripts/build_gallery.py --half <1 or 2>, or "
                  "pass --gallery-half explicitly on later runs")

    meta = {
        "gallery_half": gallery_half,
        "has_home_zones": has_zones,
        "counts": counts,
        "frame_size": manifest.get("frame_size"),
        "half_declared": "gallery_half" in manifest,
    }
    return crops, player_config, meta


# ----------------------------------------------------------------------
# Contact sheet
# ----------------------------------------------------------------------

def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit an image inside width x height without distorting it."""
    ih, iw = image.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / iw, height / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(image, (nw, nh),
                                                interpolation=cv2.INTER_AREA)
    return canvas


_letterbox = letterbox      # kept so older callers keep working


def contact_sheet(crops: dict, player_config: dict) -> Optional[np.ndarray]:
    """One labelled image: a row per player, that player's crops across it."""
    if not crops:
        return None
    columns = max(max((len(v) for v in crops.values()), default=0), 1)
    rows = []
    for pid in sorted(crops):
        label = np.zeros((CELL_H, LABEL_W, 3), dtype=np.uint8)
        colour = colour_for(pid)
        cv2.putText(label, f"{pid}: {player_config.get(pid, {}).get('name', '?')}",
                    (8, CELL_H // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.putText(label, f"({player_config.get(pid, {}).get('team', '?')})",
                    (8, CELL_H // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)
        cells = [letterbox(c, CELL_W, CELL_H) for c in crops[pid]]
        while len(cells) < columns:
            cells.append(np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))
        rows.append(cv2.hconcat([label] + cells))
    sheet = cv2.vconcat(rows)
    if sheet.shape[0] > SHEET_MAX_H:
        scale = SHEET_MAX_H / sheet.shape[0]
        sheet = cv2.resize(sheet, (int(sheet.shape[1] * scale), SHEET_MAX_H),
                           interpolation=cv2.INTER_AREA)
    return sheet


# ----------------------------------------------------------------------
# Interactive tagging
# ----------------------------------------------------------------------

def tag_interactive(cap, video_info, player_config: dict, rounds: int,
                    candidate_frames: Optional[list] = None) -> tuple:
    """
    Collect `rounds` complete rounds of tags.

    Returns (crops, home_zone_samples, frame_size, rounds_saved). The first two
    are keyed by player id, and every player has exactly `rounds_saved`
    entries - the round-based tagging guarantees it.
    """
    players = sorted(player_config)
    target = max(1, rounds)

    if candidate_frames:
        pool = [int(v) for v in candidate_frames]
    else:
        last = max(1, video_info.frame_count - 1)
        # spread wide: different poses, positions and lighting per round
        pool = [int(v) for v in np.linspace(0, last, max(target * 4, 8))]

    crops = {pid: [] for pid in players}
    zones = {pid: [] for pid in players}
    window = "Tag the players"
    state = {"drawing": False, "current": None, "pending": None, "display": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["current"] = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"][2:] = [x, y]
            preview = state["display"].copy()
            cv2.rectangle(preview, (state["current"][0], state["current"][1]),
                          (x, y), (0, 255, 255), 2)
            cv2.imshow(window, preview)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            x1, y1 = state["current"][0], state["current"][1]
            box = [min(x1, x), min(y1, y), max(x1, x), max(y1, y)]
            if (box[2] - box[0]) > 8 and (box[3] - box[1]) > 8:
                state["pending"] = box
                preview = state["display"].copy()
                cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 255), 2)
                cv2.putText(preview, "now press this player's number",
                            (box[0], max(18, box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(window, preview)

    logs.section("HOW TO TAG")
    logs.info(f"You will tag all {len(players)} players in {target} different "
              f"frames ({target} reference images each).")
    logs.info("For each player: drag a box around them, then press their number.")
    logs.info("")
    logs.info("Draw the box around the WHOLE player, feet included - the bottom")
    logs.info("edge of the box is what records where they stand on court.")
    logs.info("")
    logs.table(["key", "what it does"], [
        ["drag mouse", "draw a box around one player"],
        ["1 - %d" % len(players), "tag the box you just drew as that player"],
        ["u", "undo the last tag in this frame"],
        ["c or Enter", "save this frame (only once everyone is tagged)"],
        ["n", "skip this frame and get another one"],
        ["q", "stop tagging (rounds already saved are kept)"],
    ])
    logs.blank()
    logs.info("Roster:  " + "   ".join(
        f"{pid}={player_config[pid]['name']} ({player_config[pid]['team']})"
        for pid in players))
    logs.blank()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    def tag_one_frame(frame, frame_index, saved_rounds):
        tagged: dict = {}
        order: list = []
        state["pending"] = None
        warn_ticks = [0]

        def redraw():
            display = frame.copy()
            height, width = display.shape[:2]
            missing = [str(p) for p in players if p not in tagged]
            cv2.rectangle(display, (0, 0), (width, 92), (0, 0, 0), -1)
            cv2.putText(display, f"Round {saved_rounds + 1} of {target}"
                                 f"     tagged {len(tagged)}/{len(players)}"
                                 f"     frame {frame_index}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if missing:
                cv2.putText(display, "still to tag: " + ", ".join(missing),
                            (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 200, 255), 2)
            else:
                cv2.putText(display, "all tagged - press c to save this round",
                            (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, "drag a box, press 1-%d   |   u undo   |   "
                                 "c save   |   n skip frame   |   q stop"
                        % len(players),
                        (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            for pid, (box, _) in tagged.items():
                colour = colour_for(pid)
                cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), colour, 2)
                cv2.putText(display, f"{pid} {player_config[pid]['name']}",
                            (box[0], max(18, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
            if warn_ticks[0] > 0:
                cv2.putText(display, "tag every player before saving",
                            (12, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                warn_ticks[0] -= 1
            state["display"] = display
            cv2.imshow(window, display)

        redraw()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                if warn_ticks[0] > 0:
                    redraw()
                continue
            if key == ord("n"):
                return "skip", None
            if key == ord("q"):
                return "stop", None
            if key == ord("u") and order:
                tagged.pop(order.pop(), None)
                state["pending"] = None
                redraw()
                continue
            if key in (ord("c"), 13, 10):
                if len(tagged) == len(players):
                    return "save", {pid: tagged[pid] for pid in players}
                warn_ticks[0] = 60
                redraw()
                continue
            if ord("1") <= key <= ord(str(min(9, len(players)))) \
                    and state["pending"] is not None:
                pid = key - ord("0")
                if pid in player_config:
                    x1, y1, x2, y2 = state["pending"]
                    crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
                    if crop.size:
                        if pid not in tagged:
                            order.append(pid)
                        tagged[pid] = (state["pending"], crop)
                        state["pending"] = None
                        redraw()

    saved, pool_index = 0, 0
    skipped = 0
    frame_size = None
    while saved < target and pool_index < len(pool) * 4:
        frame_index = pool[pool_index % len(pool)]
        pool_index += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        result, tagged = tag_one_frame(frame, frame_index, saved)
        if result == "stop":
            logs.info(f"stopped by you - keeping the {saved} complete round(s) so far")
            break
        if result == "skip":
            skipped += 1
            logs.detail(f"frame {frame_index} skipped")
            continue
        height, width = frame.shape[:2]
        frame_size = (width, height)
        for pid in players:
            box, crop = tagged[pid]
            crops[pid].append(crop)
            zones[pid].append((float((box[0] + box[2]) / 2 / width),
                               float(box[3] / height)))
        saved += 1
        logs.ok(f"round {saved} of {target} saved  "
                f"({logs.mmss(video_info.seconds_of(frame_index))} in the video)")

    cv2.destroyWindow(window)
    if skipped:
        logs.detail(f"{skipped} frame(s) were skipped during tagging")
    if saved == 0:
        logs.fail("no complete rounds were tagged, so there is no gallery to save",
                  "re-run and tag every player in at least one frame before "
                  "pressing c")
    return crops, zones, frame_size, saved
