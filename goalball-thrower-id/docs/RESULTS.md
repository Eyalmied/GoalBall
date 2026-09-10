# Results, and what they can realistically reach

The reference match is a Paralympics 2024 semi-final (Israel v China): 191
detected throws, 85 of them hand-labelled with the true thrower.

Read this before trying to push the accuracy higher. The main limit is
arithmetic, not effort.

**How to read the tables.** Two kinds of number appear, and they are marked:

- **Absolute accuracy** — against the 85 hand labels. Those labels are all in
  the **first half**, which is why the second half is measured differently.
- **Comparisons** — one option against another, holding everything else fixed.
  Home zones for these were derived from first-half release points rather than
  hand-tagged, so the absolute level runs a little high; the *differences*
  between options are the point and are unaffected.

---

## 1. Where it ended up

Against the 85 hand labels:

| | Correct thrower | Correct team | Right player, given the right team |
|---|---|---|---|
| Starting point | 12.9% | 31.8% | — |
| Posture team + appearance identity | 39.5% | 88.9% | 44.4% |
| Posture team + position identity | ~71% | ~89% | ~79% |
| **Position team + position identity** (current default) | **74.1%** | **97.6%** | **75.9%** |
| Strictly video-only (`--release-source thrower`) | ~50% | ~89% | ~56% |

Guessing at random would give 1 in 6 overall, and 1 in 3 within a team.

The confidence score carries real information: calls at 0.70 and above are
right **77.6%** of the time against **61.1%** below it. So sorting by confidence
and reviewing the bottom is worth doing. `scripts/evaluate.py` prints that
split for your own footage, which is the only way to know it holds there.

---

## 2. Why roughly 72–75% is the ceiling

```
correct thrower  =  correct team  x  right player given the right team
```

Both factors are capped, and multiplying two capped numbers caps the product:

- **Team is close to solved** — 89% from posture, and ~98% once the team simply
  falls out of the position match (§4). Getting the product to 80% would need
  the second factor at about 82%.
- **The second factor cannot get much past 81%.** Measured against a perfect
  ball release point, position identity tops out there, because two teammates
  genuinely throw from overlapping parts of the court. On this match those two
  are ISR-A and ISR-C, and they account for most of the remaining errors:

  ```
  ISR-A mistaken for ISR-C: 7 times
  ISR-C mistaken for ISR-A: 4 times
  ```

`0.98 x 0.79 ≈ 0.77`. The measured 74.1% sits just under that, and the gap to
80% is those two players.

### Why appearance cannot rescue it

| | |
|---|---|
| OSNet on clean gallery crops, telling teammates apart | 61% |
| OSNet on live match frames | 44% |
| OSNet telling the two **teams** apart | 100% |

The embedding captures the kit perfectly and the person poorly — exactly what
identical uniforms and blackout eyeshades produce. Jersey numbers would settle
it, but at broadcast distance a back number is about 20 pixels tall and, for
the defenders, behind the net. OCR is not viable.

This is why `--position-weight` defaults to 1.0. Mixing appearance in was
measured to make things **worse**, not better. It is kept as a tie-break only.

---

## 3. The one thing that would move the number

The release point. Position identity is only as good as the position it is
given:

| Where the release position comes from | Right player, given the right team |
|---|---|
| Upstream detector's ball release point (`--release-source csv`) | **81%** |
| Generic COCO "sports ball" detector on the video | ~55% |
| Motion-blob tracking with trajectory fitting | ~48% |
| The video-localised thrower's own feet (`--release-source thrower`) | ~56% |

A model-free motion tracker was tried and measured: it finds *a* fast-moving
blob on essentially every throw, but not reliably *the ball at release*, so it
scores no better than the generic detector.

**The open path** is a small-object tracker trained on goalball balls
(TrackNet-style, built for tiny, fast, motion-blurred balls). If it approached
the upstream detector's quality it would give the `csv` numbers while staying
strictly video-only. Higher resolution or closer footage — enough to read a
number or see a face — would break the ceiling itself rather than just reach
it.

---

## 4. What the half work found

Making the half an explicit input exposed three things that a first-half-only
evaluation could not have shown. All three are second-half failures, and the
hand labels are all first-half, so none had ever been measured.

Second-half figures below are against the **upstream detector's own team call**
as a stand-in for truth. It is not perfect, but it is independent of everything
this pipeline does, and it agrees with the pipeline 96.7% of the time in the
first half — where hand labels confirm the pipeline is right — so it is a fair
yardstick for the second.

### 4a. Deciding each crop's team on its own inverts across halves

Matching a player's torso colour to the nearer kit prototype, crop by crop:

| | Half 1 | Half 2 |
|---|---|---|
| each crop matched on its own (`--team-grouping absolute`) | 96.7% | **22.0%** |
| ranked within the frame (`--team-grouping relative`, default) | 96.7% | 40.0% |

Not noisy — **inverted**. A player at the far end of the court is a fraction of
the size of one near the camera, their torso crop is mostly not jersey, and the
colour washes out toward whichever kit the gallery happened to record at that
scale. When the ends swap, so does which team is far away.

Ranking the crops within each frame and splitting them three and three helps
(22% → 40%) because a bias that moves every crop the same way cancels. It is
not enough on its own: here the bias depends on distance, and distance is
exactly what differs between the two ends.

### 4b. So the team should come from the position, not from the kit

The two teams stand at opposite ends of the court, so a release point near one
end cannot look like a player at the other. Naming the thrower against **all**
the players' home zones therefore answers both questions at once, and the team
falls out of the name:

| Where the team in the answer comes from | Half 1 | Half 2 |
|---|---|---|
| posture, on kit groups (`--team-source posture`) | 96.7% | 40.0% |
| **court position** (`--team-source position`, default) | 94.5% | **79.0%** |

It is also better overall, against the hand labels:

| | Correct thrower | Correct team |
|---|---|---|
| restrict to the posture team, then name | 71.8% | 97.6% |
| **name against everyone** (default) | **74.1%** | **97.6%** |
| restrict to a *perfect* team, then name | 72.9% | 100% |

The last row is the interesting one. Restricting the candidates first is worse
*even when the restriction is perfect*, because a wrong restriction removes the
right answer altogether, while an unrestricted position match simply never puts
an opponent at the wrong end of the court.

Posture is still computed. It picks the thrower out of the players on court
when the release position has to come from the video, it feeds the confidence,
and it is reported next to every prediction as a second opinion.

### 4c. The flip must turn about the zones, not the polygon

Reflecting home zones about the centre of the clicked court polygon put them
tens of pixels beyond where the teams are actually measured to be in the second
half. Reflecting about the centre of the *recorded zones* lands within a few
pixels of the measured positions.

Both are the same point when the zones are symmetric about the court centre.
They are not, in practice: a clicked polygon reaches past where anyone stands,
and a release point measured upstream need not use exactly the same coordinate
convention as a box drawn by hand. `--mirror-about court` selects the geometric
centre if you want it.

### A free cross-check

Because the half says which end each team should be attacking from, every
throw can be checked against where the ball actually came from — two
independent signals, reported per throw as `end_agrees_with_half` and summed up
at the end of a run. Agreement above ~90% means the half is set right and the
scene is being read cleanly. Below 75% the run says so: either the half is
wrong, or the kit grouping is struggling. On this match it flags the second
half correctly, which is how 4a was found.

---

## 5. Where each decision came from

| Measured symptom | Cause | Fix | Effect |
|---|---|---|---|
| Team wrong on 68% of throws | the ball is most confidently detected at the **blocker's** end, so proximity picks the defenders | team by posture | 32% → **89%** |
| Teammates confused | identical uniforms cap appearance at ~44% in-game | identity by court position | 44% → ~76-79% |
| Position slots cannot be named | naming needs appearance, the broken part | record home zones when tagging | breaks the circle |
| Single frames unstable | blur, occlusion, players crossing | vote across frames | robust |
| Second half systematically wrong | teams change ends; home zones did not follow | explicit `--half`, flip on the flat court | correct in both halves |
| Kit matching inverts in the other half | far-end crops wash out toward the gallery's kit | team from position, not kit | 40% → **79%** in half 2 |
| Mirrored zones landing too far down court | the polygon centre is not the centre of play | turn about the zones' own centre | matches the measured second half |

---

## 6. Honest status

| | |
|---|---|
| Team | **solved** — ~89% from posture, ~98% from position, by rules that follow from how goalball is played |
| Correct thrower, with the upstream release point | **74.1%**, close to the measured ceiling |
| Correct thrower, strictly video-only | **~50%**, limited by ball detection, not by the method |
| Reaching 80% | needs a better release point *and* something that separates two teammates whose zones overlap. Neither exists at this resolution |
| Reliability | confidence is informative (77.6% vs 61.1%); sort by it and review the bottom |
| Second half | correct once the half is declared. Before that work it was silently inverted |

---

## 7. Reproducing any of this

```bash
python scripts/predict_throwers.py --config config/mygame.json --half auto \
       --half2-start 22:00
python scripts/evaluate.py --config config/mygame.json --holdout 0.3
```

Every comparison above is one flag and a re-run:

```bash
--team-source posture          # decide the team first, then name within it
--team-grouping absolute       # per-crop kit matching (the one that inverts)
--release-source thrower       # video-only, no CSV fields but the timestamps
--appearance dinov2            # the other appearance model
--position-weight 0.5          # mix appearance into the identity decision
--mirror-about court           # flip about the polygon centre
--no-ball                      # localise by posture and pose alone
```

Use `--holdout` and read the held-out block. Quoting a number you tuned against
measures nothing.
