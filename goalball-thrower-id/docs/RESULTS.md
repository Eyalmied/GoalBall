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
| **Current pipeline** | **72.4%** | **99.4%** |
| half 1 / half 2 | 74.4% / 70.5% | |
| guessing | 16.7% | 50% |

A single `--holdout 0.3` split of this run gives 82.7% held out against 68.2%
on the tune block. Both come from the same 181 throws and the same model, so
the 14-point gap is the split, not the method — which is the reason this
document quotes the all-throws figure and treats differences of a few throws
as unresolved.

Correct team is effectively solved. Correct thrower is more than four times
chance, and double what the pipeline managed before the work described in §3.

---

## 2. How the thrower is found — what each option scored

All on the same 181 labelled throws, naming the player by matching a position
to the gallery's home zones.

Each row is the localiser swapped, with everything else held at the settings
of the time; the current pipeline adds the fallback gate below and reaches
**72.4%**.

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
> it was, 72.4% is optimistic for unseen footage. The honest test is a
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

### The raw embedding is worse than useless, and why

On this gallery, measured directly:

| | two photos of the **same** player | two different **team-mates** | ratio |
|---|---|---|---|
| raw OSNet (512 numbers) | 0.864 apart | 0.323 apart | **0.37** |
| after the learned projection (3 numbers) | 0.426 | 1.139 | **2.67** |

A player resembles a stranger more than they resemble themselves. The reason
is visible in the embedding's strongest direction, which accounts for 27% of
all its variance:

| that direction encodes… | separation |
|---|---|
| **how far the player is from the camera** | **5.08** |
| which team | 0.15 |
| which half | 0.07 |

OSNet is trained for street re-identification, where clothing is the main
discriminator. Both teams wear the same kit, so its strongest feature carries
nothing here, and what is left is dominated by crop scale.

`reid.fit_discriminant` learns, from the gallery's own labels, the few
directions along which different players are far apart while one player's own
photos stay close — Fisher's linear discriminant — and discards the other 509.
The kit direction scores nothing between team-mates; the camera-distance
direction scores badly because a player's own photos spread along it.

| appearance method | right team-mate (chance 33%) |
|---|---|
| OSNet, whole crop | 52.1% |
| OSNet, 3 horizontal stripes | 54.0% |
| **OSNet + learned projection** | **60.7%** |

The best-to-runner-up gap grows from 0.020 to 0.270. It is fitted per match
from that match's own gallery, so nothing transfers between games and nothing
is hand-tuned.

**But a sharper appearance signal does not make the pipeline more accurate.**
Measured end to end on all 181 throws, every appearance setting lands in the
same three-throw band, and switching appearance off entirely lands in it too:

| configuration | correct thrower |
|---|---|
| position only (appearance off) | 126/181 = 69.6% |
| raw appearance, weight 0.85 | 127/181 = 70.2% |
| raw appearance, weight 0.60 | 126/181 = 69.6% |
| projection, weight 0.85 | 126/181 = 69.6% |
| projection, weight 0.75 | 125/181 = 69.1% |

Paired McNemar over every pair: no comparison is distinguishable from chance
(all p ≥ 0.38; the largest disagreement anywhere is 4 throws against 3, and
position-only against the best appearance setting differs on **one** throw).
The discordance is symmetric — 3/3, 2/2, 4/3 — which is what noise looks like;
a weak but real signal flips more right-than-wrong.

The reason is that these are the throws the detector located, where court
position is already right 75.5% of the time, and appearance alone reaches only
59.4%. The weaker signal has nothing to add. Where appearance *is* the
deciding signal — the throws the detector missed — the projection does help,
and that is "When the detector finds nobody" below.

### Removing the background makes it worse — measured

Masking the crop to the player with `yolov8m-seg.pt`, which should in theory
remove court texture that varies with camera distance:

| crop prepared as | separation | right team-mate | end to end |
|---|---|---|---|
| **raw** | **2.67** | **60.7%** | **78.5%** |
| blacked out | 1.46 | 42.9% | 74.8% |
| filled with grey | 1.59 | 43.6% | 74.8% |
| masked then cropped tight | 1.85 | 46.0% | 74.2% |

Every variant is worse. The likely cause is that segmentation fails on 29% of
crops (13 of 60 gallery images, 47 of 163 thrower crops), so the gallery ends
up holding two incompatible kinds of image — and the projection spends its
dimensions on that artefact instead of on the players. A masking step that
never falls back might do better; this one is not used.

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

Below about 0.65 it tips over and does harm. The default is **0.75**.

Read this table together with the one in "The raw embedding is worse than
useless": on the full labelled set the
differences across the 0.6–1.0 band are not statistically separable. What the
table does establish is the *direction* — leaning hard on appearance is clearly
worse — not that 0.75 beats 0.85 by a measurable margin.

### When the detector finds nobody, position is worthless

The goalball detector names the thrower on 155 of the 181 labelled throws. On
the other 26 the crop comes from the movement-and-posture rule, which is not
reliably the thrower — so where that player was standing says nothing about
who threw:

| crop located by | position alone | appearance alone (projected) |
|---|---|---|
| the goalball detector (155) | **75.5%** | 59.4% |
| movement and posture (26) | **34.6%** | **50.0%** |

Chance within a three-player team is 33.3%. Position on the fallback throws is
*at* chance. Blending a signal that carries no information, at 75% weight,
against one that does is why that bucket used to score 26.9% — worse than
either of its two parts alone.

So the position term is dropped for those throws, and appearance names the
player within the posture-derived attacking team. Everything else is untouched:

| | correct thrower |
|---|---|
| position used everywhere (`--trust-position-always`) | 125/181 = 69.1% |
| **position only where the detector found the thrower** | **131/181 = 72.4%** |

Nine throws fixed, three broken, all twelve inside that 26-throw bucket; the
155 detector-located throws are byte-identical. McNemar gives p = 0.146, so
the *size* of the gain is not established on 26 throws. What justifies the
change is not the gain but the measurement above it: position there is at
chance, and blending noise can only move answers around.

This is also the only place the learned projection earns its keep — raw
appearance reaches 42.3% on this bucket against 50.0% projected.

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

Half 2 trailed half 1 badly (64.2% vs 73.3%) on a gallery tagged only in
half 1. With both halves tagged the gap narrows to 70.5% against 74.4%.

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

### The learned projection was reported as a gain it did not deliver

A later revision claimed the projection lifted the held-out number from
**79.2% to 82.0%**, "and all six splits agreed". That claim has been removed.

The six splits were random resamples of the *same* 163 throws, so they overlap
heavily — one weak piece of evidence counted six times, not six independent
ones. A 2.8-point gap on 52-throw test sets is about 1.5 throws. Re-run end to
end on all 181 labelled throws, the projection produced no measurable change
at all (see the table in "The raw embedding is worse than useless").

What survives is narrower and still true: the projection does sharpen the
embedding (same-player against team-mate distance ratio 0.37 → 2.67, and
within-team appearance accuracy 52.1% → 60.7%), and that sharpening matters on
the throws where appearance is the deciding signal. It does not improve the
throws where court position already decides.

The figures in this document come from a hand-tagged gallery and 181 hand
labels. Where a comparison is within a few throws it is reported as
unresolved rather than as a result.

---

## 8. Reproducing any of this

```bash
python scripts/predict_throwers.py --config config/mygame.json
python scripts/evaluate.py --config config/mygame.json --holdout 0.3
```

Each comparison still reachable is one flag and a re-run:

```bash
--thrower-model none        # fall back to movement + uprightness
--position-weight 1.0       # court position only, no appearance
--no-sharpen-appearance     # raw embedding, no LDA projection
--trust-position-always     # use position even when the detector found nobody
--team-source posture       # decide the team first, then name within it
--mirror-about court        # flip zones about the polygon centre
```

The losing arms of the other comparisons — ball proximity, throwing-pose
scoring, the CSV release point, restricting candidates to the posture-derived
attacking team, and per-crop kit matching — were removed from the code once
measured, so they are no longer reachable by a flag. See §9.

Read the held-out block. Quoting a number you tuned against measures nothing.

---

## 9. What was removed from the code, and why

Each of these was measured, lost, and then deleted rather than left behind a
default-off flag. The measurements stay here; the code does not.

| removed | what it did | why it went |
|---|---|---|
| ball detection (`--ball-weight`, `--no-ball`, `ball_model`) | generic COCO "sports ball" class, used to place the thrower | the ball is seen most confidently at the **blocker's** end, so it pointed at the wrong team. Found in 0 of 191 throws on the reference match at the default confidence |
| throwing-pose scoring (`--pose-weight`, `goalball/pose.py`) | arm extension and stance width from the skeleton | measured to make things worse; the weight had been 0 since. The pose *model* stays — it is what finds the people |
| CSV release point (`--release-source csv`) | the upstream detector's `From Coord` as the thrower's position | a different coordinate space from the video. 32.6% correct thrower against 72.4% for the video-localised position. Still read, but only to seed the labelling tool |
| candidate restriction (`--localise-among attackers`) | only the posture-derived attacking team could be the thrower | a wrong team removes the right answer entirely; naming against everyone scored better, and better even than restricting with a *perfect* team step |
| per-crop kit matching (`--team-grouping absolute`) | each crop matched to a kit prototype on its own | inverts in the half the gallery was not tagged in. Ranking within each frame survives the change of ends |

What stayed, and why:

- **movement + uprightness** — still the fallback that produces a crop on the
  26 throws the goalball detector misses. Without it those throws have no
  candidate at all.
- **posture / kit grouping** — decides which team is attacking, and is the
  fallback candidate pool when the detector misses.
- **`--team-source posture`** — not a superseded experiment: it is required,
  and selected automatically, when a gallery has no home zones.
- **LDA (`--no-sharpen-appearance` to disable)** — contributes nothing on
  detector-located throws, but lifts the no-detector bucket from 42.3% to
  50.0%, which is the only place appearance is load-bearing.
