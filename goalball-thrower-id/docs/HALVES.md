# Halves and changing ends

This is the one thing you must get right, so it gets its own page.

---

## The problem in one picture

The pipeline names a player by **where they stand**. Each player has a home
zone, recorded once when you tag the reference gallery:

```
                 ISR                                   CHN
   ┌───────────────────────────────┐    ┌───────────────────────────────┐
   │  A       B       C            │    │            C       B       A  │
   │  ·       ·       ·            │    │            ·       ·       ·  │
   │ ─────── top of frame ──────── │    │ ─────── top of frame ──────── │
   │                               │    │                               │
   │ ────── bottom of frame ────── │    │ ────── bottom of frame ────── │
   │  ·       ·       ·            │    │            ·       ·       ·  │
   │  A       B       C            │    │            C       B       A  │
   └───────────────────────────────┘    └───────────────────────────────┘
                 CHN                                   ISR
            FIRST HALF                            SECOND HALF
```

At half-time the teams change ends. Every home zone moves to the
**diametrically opposite** point of the court: top becomes bottom, *and* left
becomes right, because the players turn round with the court.

If a run is told the wrong half, it compares every release position against
home zones at the far end. It does not get a bit worse — it systematically
names the wrong team's players, or the wrong end of the same team's line. So
the half is an explicit input, stated up front.

### Only the positions move

The players do not change — just where they stand:

| | |
|---|---|
| **changes** | each player's home zone: the point moves to the opposite end of the court |
| unchanged | the player ids — player 3 is still player 3 |
| unchanged | the names, and which team each player is on |
| unchanged | the reference images — a player looks the same in both halves |
| unchanged | the court itself — same physical court both halves |

Nothing is renumbered, renamed or reassigned. The half only decides *where*
each player is expected to be.

---

## What you type

### When you tag the gallery

```bash
python scripts/build_gallery.py --config config/mygame.json --half 1
```

Tag inside **one** half and say which one. The half is written into
`gallery/manifest.json` as `gallery_half`, so you never state it again.

If your config has `half2_start`, the builder only offers you frames from the
half you named — you cannot accidentally mix the two.

### When you predict

```bash
# the footage is the same half the gallery was tagged in
python scripts/predict_throwers.py --config config/mygame.json --half 1

# the footage is the other half — home zones get mirrored
python scripts/predict_throwers.py --config config/mygame.json --half 2

# one file covering the whole match, ends changing at 22:00
python scripts/predict_throwers.py --config config/mygame.json \
       --half auto --half2-start 22:00
```

`--half auto` decides per throw from its timestamp, so a full-match video needs
one run, not two.

Put `"half2_start": "22:00"` in your config file and you can drop the flag.

---

## What the run tells you

Every run prints the half it resolved, and what that means for the teams:

```
STEP 4/6  Working out which half is being analysed
    half                   both (switching at 22:00)
    gallery tagged in      half 1
    half 1                 home zones as tagged   CHN at the bottom,  ISR at the top
    half 2                 home zones MIRRORED (teams changed ends)   CHN at the top,  ISR at the bottom
    throws in each half    half 1: 91   half 2: 100
```

**Read those two lines.** If the ends look swapped compared with the video, one
of two things is wrong: the gallery was tagged in the other half, or `--half`
is wrong for this footage. Nothing else in the run will be right until they
match.

Nothing is hard-coded here. Which end each team starts at is *derived* from the
home zones you tagged; the half only decides whether to flip them.

---

## How the flip is computed

Not by flipping the image. The camera looks along the court, so the far end
occupies far fewer pixels than the near end — flipping pixel coordinates would
put a player metres from where they actually stand.

Instead, the four court corners you clicked define a perspective mapping (a
homography) onto a flat court whose coordinates run 0..1 along each axis. On
that flat court the change of ends is exactly a half-turn about the centre
spot:

```
image point  ──homography──▶  (u, v)  ──▶  (1−u, 1−v)  ──inverse──▶  image point
```

Which is why `scripts/set_court.py` matters even if you never worried about
crowd filtering: without the four corners the flip falls back to a plain
frame-centre flip, which is only approximate. The run warns you when that
happens.

Applying the mapping twice returns the original point to within 1e-7 of a
frame, so the flip is exact and reversible.

---

## What the half does *not* change

| | affected by the half? |
|---|---|
| Home-zone **positions** | **yes** — mirrored end-for-end |
| Player ids, names, teams | no — read straight from the roster, never altered |
| Reference images | no — a player looks the same in both halves |
| The court fence | no — same physical court both halves |
| Which team is attacking | no — decided by **posture**: defenders lie flat to block, attackers stand. That rule does not care which end anyone is at, which is exactly why it survives the change of ends |
| The ball release point read from the CSV | no — it is measured in the footage, so it is already in the right half's geometry |

---

## Cross-check you get for free

Because the half tells the pipeline which end each team should be attacking
from, every prediction records whether the ball actually came from that end:

```
    posture vs the end     posture and the end the ball came from name the same
                           attacking team on 168/183 throws (91.8%)
```

Two independent signals agreeing is reassurance. If it drops below about 75%
the run says so, and names both causes:

- **`--half` is wrong for this footage** — check it, and check which half the
  gallery was tagged in;
- **or the kit colours are not separating**, which happens when one team is far
  from the camera. The team in the answer comes from court position, so it
  survives that; a strictly video-only run (`--release-source thrower`) would
  not, because it needs the kit groups to localise the thrower.

Individual disagreements are flagged on the throw's own line and stored as
`end_agrees_with_half`, so they are easy to pull out and review. The check
reports; it never overrides a decision.

---

## Symptoms of a wrong half

| What you see | Almost certainly |
|---|---|
| `evaluate.py` shows one half far worse than the other | `--half` right for one half, wrong for the other — check `half2_start` |
| Teams reported the wrong way round throughout | the gallery was tagged in the other half; re-run `build_gallery.py --half <other>` or pass `--gallery-half` |
| `posture vs the end` well below 75% | `--half` does not match this footage, or one team is too far from the camera for the kit colours to separate |
| Teammates consistently swapped (A named C and C named A) | the left/right part of the flip — the half is wrong, not the zones |

`scripts/evaluate.py` prints a per-half breakdown and warns when the two halves
differ by more than 25 points, which is the signature of this mistake.
