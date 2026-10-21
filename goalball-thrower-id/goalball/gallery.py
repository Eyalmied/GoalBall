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

TAG EACH PLAYER WHEREVER THEY LOOK CLEAREST
    Tagging is counted per player, not per frame. Box whoever is clear in the
    frame in front of you, move to another frame, and carry on. The target is
    the same number of images for each player, so nobody is over- or
    under-represented, but they do not have to come from the same frames.

ONE SET OF IMAGES PER HALF
    A player is tagged separately in each half, and the two sets are kept
    apart. This is not tidiness: the teams change ends, so a player who was
    near the camera in the first half is at the far end in the second. The
    crops differ in size, lighting and which way the player faces, and
    comparing a second-half crop against first-half references compares two
    different-looking things. Each half gets its own references, and a throw
    is matched against the half it happened in.

    One key flips between halves while tagging, as often as you like, for any
    player.

EVERY POSITION SAMPLE REMEMBERS ITS HALF
    This is what makes tagging across halves safe. Teams change ends, so the
    same player stands at opposite ends of the court in the two halves. A
    position is meaningless without knowing which half it came from - average
    the two and you get a zone at the wrong end of the court, with nothing to
    warn you.

    So each sample is stored with its half, and the medians are taken per half.
    A player tagged in both halves ends up with a MEASURED zone for each, which
    is better than the alternative: a zone measured in one half and mirrored
    into the other is only as good as the mirror.
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

def save(gallery_dir: str, crops_by_half: dict, player_config: dict,
         samples: Optional[dict], gallery_half: int,
         frame_size: Optional[tuple] = None) -> None:
    """
    Write the crops and the manifest. Overwrites any previous gallery.

    `crops_by_half` is {half: {player_id: [image, ...]}} - a separate set of
    reference images per half. `samples` is {player_id: [(x, y, half), ...]},
    every position sample with the half it was measured in.
    """
    out = pathlib.Path(gallery_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("player_*.jpg"):
        stale.unlink()

    manifest = {
        "player_config": {str(k): {"name": v.get("name") or f"player {k}",
                                   "team": v.get("team") or "?"}
                          for k, v in player_config.items()},
        "crops_by_half": {}, "crops": {},
        "gallery_half": int(gallery_half),
        "note": "crops_by_half[half][player] are that player's reference "
                "images in that half; home_zones_by_half[half][player] is "
                "(x, y) as a fraction of the frame, at their feet.",
    }
    if frame_size:
        manifest["frame_size"] = [int(frame_size[0]), int(frame_size[1])]

    for half in sorted(crops_by_half):
        for pid in sorted(crops_by_half[half]):
            files = []
            for i, crop in enumerate(crops_by_half[half][pid]):
                name = f"player_{pid}_h{half}_{i}.jpg"
                cv2.imwrite(str(out / name), crop)
                files.append(name)
            if files:
                manifest["crops_by_half"].setdefault(str(half), {})[str(pid)] = files
                manifest["crops"].setdefault(str(pid), []).extend(files)

    by_half: dict = {}
    if samples:
        for pid, points in samples.items():
            grouped: dict = {}
            for point in points:
                x, y = float(point[0]), float(point[1])
                half = int(point[2]) if len(point) > 2 else int(gallery_half)
                grouped.setdefault(half, []).append((x, y))
            for half, values in grouped.items():
                arr = np.asarray(values, dtype=float)
                # median, not mean: one mis-drawn box cannot drag the zone away
                by_half.setdefault(str(half), {})[str(pid)] = [
                    float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))]
    if by_half:
        manifest["home_zones_by_half"] = by_half
        manifest["home_zones"] = dict(by_half.get(str(int(gallery_half)), {}))

    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    logs.ok(f"gallery saved to {out}/")
    for half in sorted(manifest["crops_by_half"], key=int):
        per = manifest["crops_by_half"][half]
        counts = "  ".join(f"{pid}:{len(f)}" for pid, f in sorted(per.items(), key=lambda kv: int(kv[0])))
        logs.kv(f"half {half} images", f"{sum(len(f) for f in per.values())} total   {counts}")
    for half in sorted(by_half, key=int):
        logs.kv(f"half {half} home zones", f"measured for {len(by_half[half])} players")


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

    # Three schemas are understood, newest first:
    #   home_zones_by_half  {half: {player: [x, y]}}   zones measured per half
    #   home_zones          {player: [x, y]}           one half, named by gallery_half
    #   positions           {player: [x, y]}           what the earliest builder wrote
    gallery_half_early = int(manifest.get("gallery_half", 1))
    by_half_raw = manifest.get("home_zones_by_half") or {}
    flat = manifest.get("home_zones") or manifest.get("positions") or {}

    for pid, info in player_config.items():
        measured: dict = {}
        for half_text, players in by_half_raw.items():
            xy = players.get(str(pid))
            if xy:
                measured[int(half_text)] = [float(xy[0]), float(xy[1])]
        if not measured and str(pid) in flat:
            xy = flat[str(pid)]
            measured[gallery_half_early] = [float(xy[0]), float(xy[1])]
        if measured:
            info["home_zone_by_half"] = measured
            info["home_zone"] = measured.get(gallery_half_early) or \
                measured[sorted(measured)[0]]
    zones = flat or by_half_raw

    def read_all(names):
        out, bad = [], 0
        for name in names:
            image = cv2.imread(str(folder / name))
            if image is None:
                bad += 1
            else:
                out.append(image)
        return out, bad

    crops, unreadable = {}, 0
    for pid_text, files in manifest.get("crops", {}).items():
        images, bad = read_all(files)
        unreadable += bad
        crops[int(pid_text)] = images

    # The same images, kept split by half - what the matcher actually uses.
    crops_by_half: dict = {}
    for half_text, per_player in (manifest.get("crops_by_half") or {}).items():
        for pid_text, files in per_player.items():
            images, _ = read_all(files)
            if images:
                crops_by_half.setdefault(int(half_text), {})[int(pid_text)] = images
    if not crops_by_half and crops:
        # an older gallery: all its images belong to the half it was tagged in
        crops_by_half[int(manifest.get("gallery_half", 1))] = crops

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

    for half in sorted(crops_by_half):
        per = crops_by_half[half]
        logs.kv(f"half {half} images",
                "  ".join(f"{pid}:{len(v)}" for pid, v in sorted(per.items())))
    thin = [(h, pid) for h, per in crops_by_half.items()
            for pid, v in per.items() if len(v) < 3]
    if thin:
        logs.warn(f"{len(thin)} player/half combination(s) have fewer than 3 "
                  f"reference images",
                  "re-run scripts/build_gallery.py and tag those players more "
                  "in that half - thin references make appearance matching noisy")

    meta = {
        "gallery_half": gallery_half,
        "has_home_zones": has_zones,
        "crops_by_half": crops_by_half,
        "halves_covered": sorted(crops_by_half),
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

def tag_interactive(cap, video_info, player_config: dict, per_player: int,
                    pools: dict, start_half: int = 1) -> tuple:
    """
    Tag every player `per_player` times IN EACH HALF, in any order.

    `pools` is {half: [frame_index, ...]} - the frames on offer for each half.
    A half with no frames simply cannot be switched to, and is not required.

    Returns (crops_by_half, samples, frame_size, counts):
        crops_by_half  {half: {player_id: [image, ...]}}
        samples        {player_id: [(x, y, half), ...]}
        counts         {(half, player_id): how many images}
    """
    players = sorted(player_config)
    target = max(1, per_player)
    halves_available = [h for h in sorted(pools) if pools[h]]
    if not halves_available:
        logs.fail("no frames to tag",
                  "check --from / --to and the video length")
    half = start_half if start_half in halves_available else halves_available[0]

    crops = {h: {pid: [] for pid in players} for h in halves_available}
    samples = {pid: [] for pid in players}
    history: list = []          # (pid, half) in tag order, for undo
    position = {h: 0 for h in pools}
    frame_size = None
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

    both = len(halves_available) > 1
    logs.section("HOW TO TAG")
    logs.info(f"Give every player {target} reference image(s) IN EACH HALF "
              f"({target * len(halves_available)} each in total). Box whoever "
              f"is clear in the frame in front of you - they do not all have "
              f"to come from the same frame.")
    logs.info("")
    logs.info("Draw the box around the WHOLE player, feet included - the bottom")
    logs.info("edge is what records where they stand on court.")
    logs.info("")
    keys = [["drag mouse", "draw a box around one player"],
            ["1 - %d" % len(players), "file the box you just drew under that player"],
            ["n", "another frame from this half"],
            ["b", "go back a frame"]]
    if both:
        keys.append(["h", "switch half - the other half's view of the same players"])
    keys += [["u", "undo the last tag"],
             ["q", "stop (everything tagged so far is kept)"]]
    logs.table(["key", "what it does"], keys)
    logs.blank()
    if both:
        logs.info("Each half needs its own images: the teams change ends, so "
                  "the same player")
        logs.info("looks different in the two halves. Press h to switch and "
                  "fill in the other side.")
    else:
        logs.warn(f"only half {half} has frames to offer, so every tag will "
                  f"come from it",
                  "pass --half2-start MM:SS so the second half can be offered too")
    logs.blank()
    logs.info("Roster:  " + "   ".join(
        f"{pid}={player_config[pid]['name']} ({player_config[pid]['team']})"
        for pid in players))
    logs.blank()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    def progress_text():
        return "   ".join(f"{pid}:{len(crops[half][pid])}/{target}"
                          for pid in players)

    def outstanding(in_half):
        return [p for p in players if len(crops[in_half][p]) < target]

    def redraw(frame, frame_index):
        display = frame.copy()
        height, width = display.shape[:2]
        cv2.rectangle(display, (0, 0), (width, 96), (0, 0, 0), -1)
        remaining = [str(p) for p in outstanding(half)]
        other = [h for h in halves_available if h != half]
        tail = ("   |   half %d still needs %d" % (
            other[0], len(outstanding(other[0])))) if other and outstanding(other[0]) else ""
        cv2.putText(display, f"HALF {half}     {logs.mmss(video_info.seconds_of(frame_index))}"
                             f"     frame {frame_index}{tail}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        colour = (0, 255, 0) if not remaining else (255, 255, 255)
        cv2.putText(display, progress_text(), (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        if remaining:
            hint = f"half {half} still needs: " + ", ".join(remaining)
        elif any(outstanding(h) for h in halves_available):
            hint = "this half is done - press h for the other one"
        else:
            hint = "both halves complete - press q to finish"
        cv2.putText(display, hint + "    |   "
                    + ("h switch half   " if both else "")
                    + "n next frame   b back   u undo   q done",
                    (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        state["display"] = display
        cv2.imshow(window, display)

    stopped = False
    while not stopped:
        pool = pools[half]
        frame_index = int(pool[position[half] % len(pool)])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            position[half] += 1
            continue
        frame_size = (frame.shape[1], frame.shape[0])
        state["pending"] = None
        redraw(frame, frame_index)

        next_frame = False
        while not next_frame and not stopped:
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key == ord("q"):
                stopped = True
            elif key == ord("n"):
                position[half] += 1
                next_frame = True
            elif key == ord("b"):
                position[half] -= 1
                next_frame = True
            elif key == ord("h") and both:
                half = halves_available[(halves_available.index(half) + 1)
                                        % len(halves_available)]
                logs.detail(f"switched to half {half}")
                next_frame = True
            elif key == ord("u") and history:
                pid, was_half = history.pop()
                if crops[was_half][pid]:
                    crops[was_half][pid].pop()
                if samples[pid]:
                    samples[pid].pop()
                logs.info(f"  undid the last tag for "
                          f"{player_config[pid]['name']} in half {was_half} "
                          f"({len(crops[was_half][pid])}/{target})")
                state["pending"] = None
                redraw(frame, frame_index)
            elif ord("1") <= key <= ord(str(min(9, len(players)))) \
                    and state["pending"] is not None:
                pid = key - ord("0")
                if pid in player_config:
                    x1, y1, x2, y2 = state["pending"]
                    crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
                    if crop.size:
                        height, width = frame.shape[:2]
                        crops[half][pid].append(crop)
                        samples[pid].append((float((x1 + x2) / 2 / width),
                                             float(y2 / height), int(half)))
                        history.append((pid, half))
                        logs.ok(f"{player_config[pid]['name']}: "
                                f"{len(crops[half][pid])}/{target}  "
                                f"(half {half}, "
                                f"{logs.mmss(video_info.seconds_of(frame_index))})")
                        state["pending"] = None
                        redraw(frame, frame_index)

    cv2.destroyWindow(window)

    counts = {(h, pid): len(crops[h][pid])
              for h in halves_available for pid in players}
    never = [pid for pid in players
             if all(counts[(h, pid)] == 0 for h in halves_available)]
    if never:
        logs.fail("these players were never tagged: "
                  + ", ".join(player_config[p]["name"] for p in never),
                  "every player needs at least one reference image - re-run "
                  "and tag them, pressing h to look in the other half if they "
                  "are hard to see in this one")
    short = [(h, pid) for (h, pid), n in counts.items() if n < target]
    if short:
        logs.warn(f"{len(short)} player/half combination(s) have fewer than "
                  f"{target} images: "
                  + ", ".join(f"{player_config[p]['name']} h{h} ({counts[(h,p)]})"
                              for h, p in sorted(short)),
                  "allowed, but those are matched from less evidence - re-run "
                  "to top them up")
    return crops, samples, frame_size, counts
