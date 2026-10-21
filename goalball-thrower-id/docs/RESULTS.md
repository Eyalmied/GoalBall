# Results, and what they can realistically reach

The reference match is a Paralympics 2024 semi-final (Israel v China): 191
detected throws, **181 of them hand-labelled** with the true thrower, across
both halves.

Every figure here was measured. Where a number is an upper bound that used the
labels to fit something, it says so.

---

## 1. Where it ended up

| | Correct thrower | Correct team |
|---|---|---|
| **Current pipeline** | **68.5%** | **98.9%** |
| held out (52 throws) | 73.1% | 100% |
| half 1 / half 2 | 73.3% / 64.2% | |
| guessing | 16.7% | 50% |

Correct team is effectively solved. Correct thrower is more than four times
chance, and double what the pipeline managed before the work described in §3.

---

## 2. How the thrower is found — what each option scored

All on the same 181 labelled throws, naming the player by matching a position
to the gallery's home zones.

| where the position comes from | thrower | team |
|---|---|---|
| **`throwing_player` class of the goalball detector** | **68.5%** | **98.9%** |
| movement + uprightness across frames | 46.4% | 94.5% |
| the upstream CSV's `From Coord` | 32.6% | 53.0% |
| a self-detected ball, best of five strategies | 34.9% | 56.0% |
| the goalball detector's own ball class | 31.6% | 60.2% |

### The ball is the wrong thing to look for

Five ways of choosing "the ball at release" were tried: most confident in the
window, most confident in the first 1/2/3 frames, earliest in court, and a
trajectory fitted back to the throw start. The best managed 34.9%.

The row that settles it: picking **perfectly** among every ball the detector
found still gives only **30.7%**. No smarter selection rule helps, because the
detections themselves are not near the thrower. Restricting to the first few
frames — where the release actually is — collapses coverage to 12–30%: at
release the ball is small, motion-blurred and behind the thrower's body. It
only becomes detectable once it is mid-court or at the blocker's end, which is
exactly where it is useless.

### What works instead

The project's own fine-tuned detector (`Model Weights/best.pt`) has a class
`throwing_player`. It answers the question directly rather than inferring it,
and it is consistent: taking its most-confident detection, voting its
per-frame answers, or taking the median of them all give **identical** results.

It finds a thrower in about 83% of throws. The rest fall back to movement +
uprightness.

> **Caveat.** It could not be established whether this match was in that
> detector's training data — the training images are not in the repository. If
> it was, 68.5% is optimistic for unseen footage. The honest test is a
> different match.

---

## 3. Why the number used to be much lower

Three faults, each found by measurement.

### 3a. The CSV release point is in a different coordinate system

`From Coord` does not line up with the video. In half 1 the hand-tagged zones
put ISR at v=0.39 and CHN at v=0.95, while the CSV release points sit at
v=0.05 and v=0.47. A CHN throw therefore landed nearer **ISR's** zone than
CHN's own, and was named as the matching ISR player — the symmetric
ISR↔CHN confusions that dominated the old error list.

Converting between the two systems does not work: an affine fit reaches 51.6%
held-out with a median residual of 0.126 of a frame, about the spacing between
team-mates. Building zones *in the CSV's own units* from labelled throws does
work (~70%), but needs labels. Detecting the thrower in the video sidesteps
the whole problem, because a detection and a tagged zone are both in video
pixels.

### 3b. Ball proximity pointed at the blocker

The old localisation cost was 50% "distance to the detected ball". For the
reason in §2 that systematically selected a defender. Replacing it with
movement + uprightness took the team from 53% to 94.5%.

### 3c. Averaging positions across frames that are about different people

The sampled frames were combined by taking the median position. That is only
valid when every frame picked the same person. When the detector names the
thrower in two frames and the fallback guesses someone else in the other
three, the median lands between two players and means nothing — this cost
**16 points** (59.7% instead of 75.5%) until the two were kept apart. The
detector's frames now win outright whenever it has any.

The same effect appears in the pure-motion path: naming the thrower per frame
and voting, weighted by confidence, beats averaging positions by 6.7 points.

---

## 4. Naming the player: what helps and what does not

### Court position, not appearance

| | within-team |
|---|---|
| naming by court position | 69.3% |
| appearance alone (OSNet) | ~50% |

Identical uniforms and blackout eyeshades cap appearance: OSNet reaches 61% on
clean gallery crops and ~44% on live frames. Jersey numbers would settle it,
but at broadcast distance a back number is ~20 px and often behind the net.

### Appearance as a tie-break does now help

Earlier it did not, and the reason it changed is instructive. When the
localiser was wrong half the time, the crop being compared was often of the
wrong player, so appearance shuffled answers at random — 23 fixed, 20 broken.
With a localiser that actually finds the thrower, the crop is genuinely that
player:

| position weight | thrower | fixed by appearance | broken by it |
|---|---|---|---|
| 1.0 (position only) | 73.0% | – | – |
| **0.75** | ~75% | | |
| 0.7 | 75.5% | 6 | 2 |
| 0.6 | 71.2% | 7 | 10 |
| 0.0 (appearance only) | 52.8% | 17 | 50 |

Three fixes per break at 0.7–0.8, and held-out splits pick that band every
time (77.0% → 79.1%). Below about 0.65 it tips over and does harm. The default
is **0.75**.

### Zone boxes are worse than zone points

Drawing a box per player — the area they throw from — was tried against points.
Both fitted perfectly to the labels as an upper bound:

| | thrower |
|---|---|
| ideal **point** zones | **82.2%** |
| ideal **box** zones | 79.8% |
| the tagged point zones actually in use | 73.0% |

A box treats everything inside as equally good, so where two players' areas
overlap it has nothing left to break the tie; a point keeps the distance.
Boxes were not adopted.

### How much is left in better zones

The gap from 73.0% to the 82.2% ideal is **zone placement** — tagged zones come
from wherever a player happened to stand, not from where they stand when
throwing. Letting the detector's own output refine the zones iteratively,
without labels, gave 73.0% → 76.1% after one round but drifted back to 74.8%:
real but small and not stable.

---

## 5. The ceiling

In **17.2%** of throws the detected position genuinely falls nearer a
team-mate's zone than the thrower's own. Measured a second way, in CSV space:
a player's own throws scatter by 0.05–0.13 while the nearest team-mate sits
0.08–0.31 away, and **22.1%** of throws land nearer a team-mate's usual spot.

That is players crossing and covering for each other, not a measurement
problem, and no zone shape or matching rule fixes it. It caps naming by
position at roughly 78–82% within-team.

```
correct thrower  =  correct team (99%)  x  right player given the team (69%)
```

Getting past ~80% overall needs something position cannot give: readable
numbers or faces, which this resolution does not offer.

---

## 6. Halves

Making the half explicit exposed two failures that a first-half-only
evaluation could not have shown.

**Per-crop kit matching inverts across halves** — 96.7% in the half the gallery
was tagged in, **22.0%** in the other. A player at the far end is a fraction of
the size of one near the camera, their torso crop is mostly not jersey, and the
colour washes out toward whichever kit the gallery recorded at that scale.
Ranking the crops within each frame and splitting them three and three lifts it
to 40%; taking the team from court position instead gives 94.5% / 79.0%.

**The end-swap flip must turn about the zones, not the court polygon.**
Reflecting about the polygon centre put mirrored zones tens of pixels past
where the teams actually are; reflecting about the centre of the recorded zones
lands within a few.

Both are why references and zones are now recorded **per half** rather than
mirrored: a measured zone carries no mirror error at all.

Half 2 still trails half 1 (64.2% vs 73.3%) on a gallery tagged only in half 1.
Tagging both halves is expected to narrow that.

---

## 7. A correction to earlier versions of this document

Earlier revisions claimed **74.1% correct thrower / 97.6% correct team**. That
figure was wrong and has been removed.

It was produced with a gallery whose home zones were derived from the CSV
`From Coord` values of the very throws being evaluated. Both sides therefore
shared one coordinate system *and* the zones were fitted to the test answers.
It validated that position-based naming works as a mechanism; it did not
measure accuracy. Against a real hand-tagged gallery the same configuration
scores 32.6%.

The figures in this document come from a hand-tagged gallery and 181 hand
labels, with held-out splits where a choice was being made.

---

## 8. Reproducing any of this

```bash
python scripts/predict_throwers.py --config config/mygame.json --release-source thrower
python scripts/evaluate.py --config config/mygame.json --holdout 0.3
```

Each comparison above is one flag and a re-run:

```bash
--thrower-model none       # fall back to movement + uprightness
--position-weight 1.0      # court position only, no appearance
--ball-weight 0.5          # bring ball proximity back
--release-source csv       # use the upstream CSV release point
--localise-among attackers # restrict to the posture-derived attacking team
--team-grouping absolute   # per-crop kit matching (the one that inverts)
--mirror-about court       # flip zones about the polygon centre
```

Read the held-out block. Quoting a number you tuned against measures nothing.
