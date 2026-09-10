# Design — how it works, and why it is built this way

For instructions, see [USAGE.md](USAGE.md). For the numbers and their limits,
[RESULTS.md](RESULTS.md). This page is the reasoning.

---

## 1. The problem

**In:** a match video and a list of throw timestamps.
**Out:** for every throw, the player who threw it, their team, and a
confidence.

Goalball is hostile to every standard way of identifying a person on video:

- both teams wear identical uniforms, so kit tells you the team and nothing more;
- every player wears blackout eyeshades, so there are no faces;
- at broadcast distance a back number is about 20 pixels tall, and for the
  defenders it is behind the net — OCR is not on the table;
- players dive, cross and lie on top of one another constantly.

Face recognition, number recognition and colour matching are all out. That
constraint shaped every decision below.

---

## 2. The one idea: split the question

"Who threw?" is really several questions, and separating them is what made the
problem tractable — each has a different reliable signal, and each can be
measured on its own.

| | Question | What answers it | Measured |
|---|---|---|---|
| **Localise** | Which person on court is throwing? | nearest where the ball came from, most stepped-in, most throw-like pose | — |
| **Identify** | Who is that player? | their **court position** matched to home zones recorded at tagging time | **~76–81%** |
| **Team** | Which team is that? | it falls out of the name — the two teams stand at opposite ends | **~98%** |
| *(also)* | Which team is attacking? | **posture** — defenders lie flat to block, attackers stand | **89%** |

Correct thrower = right team **and** right player, so the end-to-end figure is
the product. Measuring the pieces separately is what let each bottleneck be
found and fixed instead of guessed at.

The team started life as a separate decision made from posture, and it still is
computed that way — it localises the thrower for a video-only run, it feeds the
confidence, and it is a useful second opinion. But it is no longer what the
answer's team comes from: naming the thrower against **every** player's home
zone gives both at once, and measured better (§4.8). The rows are in the order
the pipeline actually runs them.

---

## 3. The flow

```
ONCE PER MATCH  (interactive, ~6 minutes)
  ┌──────────────────────────────────────────────────────────────┐
  │  Tag all 6 players in K complete rounds, inside ONE half     │
  │    -> reference images per player                            │
  │    -> HOME ZONE per player (median foot point)               │
  │    -> the half it was tagged in, stored with the gallery      │
  │  Click the 4 court corners                                    │
  │    -> the fence, and the flat court used to flip ends         │
  └──────────────────────────────────────────────────────────────┘
                                │
PER THROW  (driven by the timestamp alone)
  ┌──────────────────────────────────────────────────────────────┐
  │ 1.  which half is this throw in?  -> home zones for it        │
  │ 2.  sample N frames across the start of the throw window      │
  │ 3.  detect people + skeletons (one YOLO-pose pass)            │
  │ 4.  drop anyone whose FEET are outside the court              │
  │ 5.  group into two teams by kit colour (ranked in-frame)      │
  │ 6.  POSTURE: the more upright team is attacking                │
  │ 7.  find where the ball was released                          │
  │ 8.  LOCALISE the thrower among the attackers                   │
  │ 9.  IDENTIFY: release position -> nearest home zone,          │
  │              over every player; the winner's team is the team  │
  │ 10. vote across the frames                                     │
  │ 11. emit player + team + confidence + why                      │
  └──────────────────────────────────────────────────────────────┘
```

---

## 4. The parts

### 4.1 Reference gallery and home zones — `goalball/gallery.py`

Tagging happens in **complete rounds**: a frame is saved only when every player
has been boxed in it and confirmed. Partial frames are discarded whole, so
every player ends with the same number of references and the same number of
position samples — nobody is over- or under-represented.

Each box gives two things. The pixels feed the appearance model. The
bottom-centre of the box — the player's feet, as a fraction of the frame — is
their **home zone**, taken as the median across rounds so one mis-drawn box
cannot drag it away.

```json
{
  "player_config": {"1": {"name": "...", "team": "ISR"}, "...": {}},
  "crops":       {"1": ["player_1_0.jpg", "..."]},
  "home_zones":  {"1": [0.34, 0.39]},
  "gallery_half": 1
}
```

Recording the home zone *at tagging time* is the move that makes
position-based identity possible at all. Naming a position slot otherwise
requires appearance to link the slot to a name — and appearance is the broken
part. Tagging breaks that circle for the cost of one extra number per box.

`gallery_half` is stored because a home zone means nothing without the half it
was measured in (§4.3).

### 4.2 The court — `goalball/court.py`

Four clicked corners, doing two jobs.

**The fence.** Anyone whose **feet** fall outside the polygon is dropped.
Testing the feet rather than the box centre matters: a spectator in the front
row easily overlaps the court in the image, but their feet never do. This
removes the single largest source of wrong candidates before any reasoning
starts.

**The flat court.** The same corners define a homography onto a rectangle whose
coordinates run 0..1 along each axis. Positions can then be reasoned about in
real court terms rather than in pixels, which is what makes the half-time flip
correct rather than approximate (§4.3).

### 4.3 Halves and changing ends — `goalball/halves.py`

Teams change ends at half-time, which moves every home zone to the opposite
point of the court. The half is therefore an explicit input to a run, and the
flip is applied to the home zones when the run's half differs from the
gallery's.

Two details make it right rather than roughly right:

**It happens on the flat court, not in the image.** The camera looks along the
court, so the far end occupies far fewer pixels than the near end. Flipping
pixel coordinates would put a player metres from where they stand. Mapping to
the flat court, turning, and mapping back is exact — applying it twice returns
the original point to within 1e-7 of a frame.

**It turns about the middle of the recorded zones, not the middle of the
polygon.** Three players a side hold zones symmetric about the centre spot, so
in an ideal world the two centres coincide. In practice a clicked polygon
usually reaches past where anyone stands, and a release point measured by the
upstream detector need not use exactly the same coordinate convention as a box
drawn by hand; turning about the polygon centre then pushes every mirrored zone
too far down the court. On the reference match, turning about the zones' own
centre lands within a few pixels of where each team is actually measured to be
in the second half, against tens of pixels for the polygon centre.
`--mirror-about court` selects the geometric centre if you want it.

What the half deliberately does **not** touch is the team decision. That is
posture-based (§4.5), which does not care which end anyone is at — and that is
precisely why the team call survives the change of ends. Full detail in
[HALVES.md](HALVES.md).

### 4.4 Detection — `goalball/detect.py`

One pass of `yolov8m-pose` per frame returns a box and a 17-point COCO skeleton
per person. Posture and throwing shape both come out of that same pass, so
there is no second model to run or keep in step.

### 4.5 Team by posture — the decisive fix — `goalball/teams.py`

The obvious rule — "whoever is nearest the ball threw it" — scored **32%**,
worse than a coin toss. The reason is visible as soon as you look at the
frames: the ball is easiest to detect where it is largest and slowest, and that
is at the **blocker's** end. Proximity to the ball picks the defending team
almost every time.

Posture works instead. Defenders throw themselves flat across the goal line;
attackers stay on their feet to throw. A detection box therefore reports the
role: lying players are wide and short, standing players tall and narrow.
Average that flatness per team and the more upright team is attacking. **32% →
89%.**

It also generalises, which the ball rule never could: it depends on how goalball
is played, not on which end a team defends, so it survives the change of ends
and transfers to other matches untouched.

Players are grouped into the two teams by kit colour — a hue/saturation
histogram of the torso against prototypes built from the gallery. The two kits
are fully separable this way (100% on the reference gallery), so colour is
reliable for "same team or not" even though it is useless for telling teammates
apart. The run checks that separability at start-up and says so, because if the
kits are not separable in your footage you should know before waiting for a
full run.

**But not crop by crop.** Asking each crop on its own which kit it is nearer to
is right 96.7% of the time in the half the gallery was tagged in and **22%** in
the other half — inverted, not noisy. A player at the far end of the court is a
fraction of the size of one near the camera, their torso crop is mostly not
jersey, and the colour washes out toward whichever kit the gallery recorded at
that scale; when the ends swap, so does which team is far away. So the crops in
a frame are *ranked* by how much more they look like one kit than the other and
split three and three, which cancels a bias that moves them all the same way
(22% → 40%). It is still the weakest signal in the pipeline, which is why the
answer's team no longer depends on it (§4.8).

### 4.6 Where the ball came from — `goalball/detect.py`

The generic COCO "sports ball" class over a few early frames, court-gated, most
confident detection wins. It is good enough to say roughly where the release
was and no better. When the ball is never confidently seen the throw falls back
to posture and pose rather than trusting a bad point — a deliberate choice to
degrade rather than guess.

This is the weakest link in the whole system and the honest reason for the
video-only accuracy figure; see [RESULTS.md](RESULTS.md) §3.

### 4.7 Localising the thrower — `goalball/predict.py`

Among the attacking team's detections, each candidate gets a cost:

```
cost =  0.5 x  distance to where the ball came from
     +  0.3 x  distance to the centre of the court   (throwers step in)
     +  0.2 x  (1 - how much the body looks mid-throw)
```

All three are scaled by the frame diagonal, so they are comparable and
resolution-independent. Lowest cost wins that frame.

The pose term is torso-normalised: arm extension and stance width are divided
by the player's own torso length, so the same number means the same posture for
a player at the far end of the court as for one near the camera. Raw pixel
distances would mean nothing across those two cases.

### 4.8 Identity — position first, and the team with it — `goalball/predict.py`

The release position is matched against the home zones for this half — of
**every** player, not just the attacking team's three:

```
position_score(player)   = exp( -0.5 * distance(release, home_zone)^2 / spread^2 )
appearance_score(player) = mean of the top-k cosine matches against that
                           player's reference images
final = w * position_score + (1 - w) * appearance_score
```

`w` defaults to **1.0** — pure position. Appearance is available as a tie-break,
but weighting it higher was **measured to make things worse**: identical
uniforms mean the embedding captures the kit perfectly and the person poorly.
This is not an oversight to be tuned away.

**Why every player and not just the attacking three.** The two teams stand at
opposite ends of the court, so a release point near one end cannot look like a
player at the other — the position match settles the team as a side effect. And
narrowing the candidates first turns out to be worse *even when the narrowing
is perfect*, because a wrong restriction removes the right answer entirely
while an unrestricted match simply never puts an opponent at the wrong end:

| | Correct thrower | Correct team |
|---|---|---|
| restrict to the posture team, then name | 71.8% | 97.6% |
| **name against everyone** (default) | **74.1%** | **97.6%** |
| restrict to a *perfect* team, then name | 72.9% | 100% |

Across halves the difference is much larger — 40% against 79% for the team in
the second half — because narrowing depends on the kit grouping, and that is
what breaks (§4.5). `--team-source posture` restores the older behaviour, and
is used automatically when the gallery has no home zones, since appearance
alone cannot cope with six candidates.

The release position comes from one of two places, chosen with
`--release-source`:

- **`csv`** (default) — the upstream detector's ball release point, used for
  the thrower's *position only*, never for the team. ~71% correct thrower.
- **`thrower`** — the video-localised thrower's median foot position. Strictly
  video-only. ~50%.

### 4.9 Appearance — `goalball/reid.py`

OSNet (purpose-built for person re-identification) or DINOv2 (general, easier
to install). Two standard practices: horizontal-flip augmentation, since
players face every direction across a throw; and multi-shot matching, keeping
every reference image as its own labelled point rather than averaging a player
into one prototype, so front, back and prone appearances all survive.

### 4.10 Voting

Steps 3–9 run on several frames across the start of the throw and the results
are aggregated, so one blurred, blocked or mid-dive frame cannot flip the
answer. This is the aggregation principle behind tracking-by-detection, applied
to a short window.

### 4.11 Confidence

Three readable parts, multiplied:

| part | meaning |
|---|---|
| `identity_margin` | how clearly the winner beat the runner-up |
| `team_separation` | how differently the two teams are standing — near zero means nobody is down yet and the team call is a guess |
| `frame_coverage` | how many of the sampled frames actually contributed |

Exposed per throw as `confidence` and `confidence_parts`, so weak calls can be
pulled out and reviewed rather than silently trusted.
`scripts/evaluate.py` reports accuracy split by confidence, which is the only
way to know the score means anything on your footage.

---

## 5. Every rule came from a measured failure

| Symptom, measured | Cause | Fix | Result |
|---|---|---|---|
| Team wrong on 68% of throws | the most confidently detected ball sits at the **blocker's** end | team by **posture** | 32% → **89%** |
| Teammates confused | identical uniforms cap appearance re-ID at ~44% in-game | identify by **court position** | 44% → ~80% |
| Position slots cannot be named without labels | naming needs appearance, which is the broken part | record **home zones at tagging time** | breaks the circle |
| Single-frame answers unstable | blur, occlusion, players crossing | vote across frames | robust |
| Second-half calls inverted | teams change ends; home zones do not follow | **explicit half** + flip on the flat court | correct in both halves |
| Kit matching inverts in the other half | far-end crops wash out toward the gallery's kit | rank the crops **within the frame**, and take the team from **position** instead | 22% → 40% → **79%** in half 2 |
| Narrowing to one team loses answers | a wrong restriction removes the right player | name against **every** home zone | 71.8% → **74.1%** |
| Mirrored zones landing too far down court | the polygon centre is not the centre of play | turn about the **zones' own** centre | matches the measured second half |
| Not all calls equally reliable | — | confidence, in three parts | reviewable output |

The thread running through all of it: every rule is grounded in **how goalball
is played** — defenders lie down, players hold zones, teams change ends — not
tuned to one match. That is what makes it transfer.

---

## 6. Deliberate anti-overfitting

- **Nothing about a match is in the source.** Roster, paths and the half-time
  timestamp live in a config file; home zones and the tagged half live in the
  gallery; the court lives in its own file; resolution and frame rate come from
  the video; timestamps come from the CSV.
- **Only timestamps are taken from the CSV.** The throwing team is never used
  to decide anything — it is kept per record as `csv_team_hint` for comparison.
  The ball release point is opt-out and affects position only.
- **Every weight is a documented CLI flag** with a measured default, not a
  fitted constant.
- **Features are ratio-normalised** (torso length, frame diagonal, fractions of
  the frame), so no threshold depends on camera distance, zoom or player size.
- **`evaluate.py --holdout`** splits deterministically, so tuning happens on
  one part and the reported number comes from another.

---

## 7. The code

| File | Job |
|---|---|
| `goalball/logs.py` | every user-facing line, in one place |
| `goalball/config.py` | run settings: config file plus CLI overrides |
| `goalball/video.py` | video handle, frame sampling, downscaling |
| `goalball/events.py` | the throws CSV to Throw objects |
| `goalball/court.py` | the fence, the flat court, the end-for-end flip |
| `goalball/halves.py` | which half, and what that does to home zones |
| `goalball/gallery.py` | tag, save and load the reference gallery |
| `goalball/detect.py` | people, poses and the ball |
| `goalball/pose.py` | "does this body look like it is throwing" |
| `goalball/teams.py` | kit-colour grouping, posture-based attacking team |
| `goalball/reid.py` | appearance embeddings, reference bank |
| `goalball/predict.py` | the per-throw predictor and the run summary |
| `goalball/labels.py` | the ground-truth file |
| `goalball/evaluate.py` | accuracy, per half, per confidence, confusions |
| `scripts/run_pipeline.py` | **the whole flow in one command** |
| `scripts/build_gallery.py` | step 1, tag the players |
| `scripts/set_court.py` | step 2, mark the court |
| `scripts/predict_throwers.py` | step 3, the prediction |
| `scripts/label_throwers.py` | optional, record ground truth |
| `scripts/evaluate.py` | step 4, measure it |
| `scripts/review_predictions.py` | look at the calls next to the truth |
| `scripts/selftest.py` | check the install and the geometry |

---

## 8. References

- Gudauskas & Matusevičius, *Multiple object tracking for video-based sports analysis*, CEUR-WS 2915, 2021 — tracking by detection, trajectory back-projection.
- Zhou et al., *Omni-Scale Feature Learning for Person Re-Identification (OSNet)*, ICCV 2019.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- Luo et al., *Bag of Tricks and a Strong Baseline for Deep Person Re-ID*, CVPR-W 2019 — flip augmentation, multi-shot matching.
- Bewley et al., *SORT*, ICIP 2016; Zhang et al., *ByteTrack*, ECCV 2022 — aggregating per-frame detections.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, ch. 7 — held-out assessment.
- Project thesis: *Strategic Spatio-Temporal Event Attribution and Multi-Player Tracking in Paralympic Goalball Analytics* — posture and activity states, pose biometrics, ball-player matching, tactical position priors.
