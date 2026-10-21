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
| **changes** | each player's home zone: the point moves to the opposite end of the court — *measured directly* if you tagged them in both halves, mirrored if not |
| unchanged | the player ids — player 3 is still player 3 |
| unchanged | the names, and which team each player is on |
| **changes** | which reference images are used — each half has its own set |
| unchanged | the court itself — same physical court both halves |

Nothing is renumbered, renamed or reassigned. The half only decides *where*
each player is expected to be.

---

## Set it once, in the config

**A match has two halves, so that is the default.** Every run covers both and
splits them at `half2_start` from your config — the **single source of truth**
for when the ends change. Every step reads it from there (tagging, the court
check, labelling, prediction), so they cannot disagree about where the match
splits.

```json
{
  "...": "...",
  "half2_start": "22:00"
}
```

With that set, nothing else is needed:

```bash
python scripts/run_pipeline.py --config config/mygame.json
```

The run says what it worked out and where it got it:

```
    half                   auto   (the config gives a half-time, so both halves are in play)
    ends change at         22:00   (from the config file)
```

Because two halves is the default, `half2_start` is **required**. Leave it out
and the run stops rather than guessing:

```
✗ PROBLEM  a match is treated as two halves, but nothing says when the ends change
     what to do: add "half2_start": "22:00" to your config file - the video
     timestamp where the second half begins.
     If this video is a SINGLE half, say so instead: --half 1 (or --half 2).
```

That is deliberate. The old behaviour — quietly treating the whole video as
one half — is exactly how every second-half throw ends up matched against the
wrong end of the court, with nothing on screen to tell you.

**If your video really is a single half**, say so with `--half 1` or
`--half 2`. Then no `half2_start` is needed.

The flags below still exist for one-off experiments. `--half2-start` overrides
the config for that run only, and the run warns you when the two disagree,
because a disagreement means the steps are splitting the match differently.

---

## What you type

### When you tag the gallery

```bash
python scripts/build_gallery.py --config config/mygame.json --half 1
```

Tag from **either** half — press `h` during tagging to switch, as often as you
like, for any player. Each position is stored with the half it came from, so
the two never get averaged together.

Each half gets **its own reference images and its own home zones**, and the
builder asks for five images per player per half. This is not tidiness: a
player near the camera in one half is at the far end in the other, so the
crops differ in size, lighting and which way they face. A throw is matched
against the half it happened in.

A player tagged in both halves ends up with a **measured** zone for each, and
nothing is mirrored for them. A player tagged in only one gets their zone
mirrored across, which works but carries the mirror's error.

`--half` says which half to start in, and which one the gallery reports as its
reference (`gallery_half` in `manifest.json`).

### When you predict

```bash
# normal: the config's half2_start decides everything
python scripts/predict_throwers.py --config config/mygame.json

# this footage is only one half, whatever the config says
python scripts/predict_throwers.py --config config/mygame.json --half 1
python scripts/predict_throwers.py --config config/mygame.json --half 2

# try a different half-time without editing the config (warns that it differs)
python scripts/predict_throwers.py --config config/mygame.json --half2-start 25:00
```

`auto` decides per throw from its timestamp, so a full-match video needs one
run, not two.

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
| Home-zone **positions** | **yes** — measured per half where you tagged, mirrored end-for-end where you did not |
| Player ids, names, teams | no — read straight from the roster, never altered |
| Reference images | **yes** — each half has its own set, and a throw is matched against its own half |
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
