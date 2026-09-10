# The output file

`predictions.json` is a list, one record per throw, rewritten after every throw
so it is always valid even if you stop the run.

```json
{
  "throw_id": "throw_12",
  "start_time": "00:34",
  "half": 1,
  "home_zones_mirrored": false,

  "thrower_id": 2,
  "thrower_name": "Maya",
  "thrower_team": "ISR",
  "team_from": "the named player's own team",
  "attacking_team": "ISR",

  "confidence": 0.85,
  "confidence_parts": {
    "identity_margin": 0.94,
    "team_separation": 0.81,
    "frame_coverage": 1.0
  },

  "decided_by": "court position",
  "frames_voted": 5,
  "identity_scores":   {"1": 0.21, "2": 0.88, "3": 0.34},
  "appearance_scores": {"1": 0.52, "2": 0.61, "3": 0.55},

  "ball_seen": true,
  "release_position": [0.34, 0.39],
  "release_position_from": "csv ball release point",
  "ball_came_from_team_end": "ISR",
  "attacking_team_end": "top",
  "end_agrees_with_half": true,

  "team_flatness": {"ISR": 0.71, "CHI": 1.04},
  "csv_team_hint": "ISR",
  "resolved": true
}
```

## What each field means

### Which throw

| field | meaning |
|---|---|
| `throw_id` | the id from the throws CSV, e.g. `throw_12` |
| `start_time` | when it happens in the video, `mm:ss` |
| `half` | which half this throw was treated as |
| `home_zones_mirrored` | whether the home zones were flipped end-for-end for this throw, i.e. whether this half differs from the one the gallery was tagged in |

### The answer

| field | meaning |
|---|---|
| `thrower_id` / `thrower_name` | the player, from the gallery roster. `null` when no call could be made |
| `thrower_team` | that player's team — **this is the team in the answer** |
| `team_from` | how the team was arrived at: the named player's own team, or posture |
| `attacking_team` | which team posture says was attacking. Reported even when it did not decide the answer, so the two can be compared |

### How sure

| field | meaning |
|---|---|
| `confidence` | 0..1, the product of the three parts below |
| `identity_margin` | how clearly the winner beat the runner-up |
| `team_separation` | how differently the two teams are standing. Near zero means nobody is down yet and the scene is hard to read |
| `frame_coverage` | how many of the sampled frames actually contributed |

Confidence carries real information: on the reference match, calls at 0.70 and
above are right far more often than the ones below. `scripts/evaluate.py`
prints that split for your own footage. Review from the bottom:

```bash
python scripts/review_predictions.py --config ... --only-uncertain 0.4
```

### Why

| field | meaning |
|---|---|
| `decided_by` | "court position", or "court position plus appearance", or "appearance only (no home zones available)" |
| `frames_voted` | how many frames produced a thrower to vote on |
| `identity_scores` | the final score per candidate player — shows how close the runner-up was |
| `appearance_scores` | the raw appearance similarity per candidate, kept even when it did not decide anything |

### The ball, and the end it came from

| field | meaning |
|---|---|
| `ball_seen` | whether the ball was confidently found in the video for this throw |
| `release_position` | the point used to name the thrower, as a fraction of the frame |
| `release_position_from` | `csv ball release point` or `video-localised thrower` |
| `ball_came_from_team_end` | **which team's** end that point is nearer to, in this half |
| `attacking_team_end` | which **side of the frame** — top or bottom — posture's attacking team is at in this half |
| `end_agrees_with_half` | whether `ball_came_from_team_end` and `attacking_team` name the same team. Two independent signals, so it is a free cross-check on the `--half` setting — see [HALVES.md](HALVES.md) |

### Diagnostics

| field | meaning |
|---|---|
| `team_flatness` | mean box width/height per team. Higher means more players lying down. This is the raw material of the posture decision |
| `csv_team_hint` | the upstream detector's own team call. **Never used by anything** — carried so you can compare |
| `resolved` | false when no call was made. `reason` then says why in plain words |

## When a throw gets no call

```json
{
  "throw_id": "throw_57",
  "start_time": "12:03",
  "half": 1,
  "thrower_id": null,
  "confidence": 0.0,
  "decided_by": "not decided",
  "reason": "nobody was detected on court in this window",
  "resolved": false
}
```

The pipeline says nothing rather than guessing. The reasons you may see:

| reason | what happened |
|---|---|
| `nobody was detected on court in this window` | players fully occluded, or the court fence is too tight |
| `only one team was visible, so the attacking team is unknown` | one side was not detected at all |
| `no <TEAM> player was detected in this window` | the attacking team was grouped but never localised |
| `team '<TEAM>' is not in the gallery roster` | the roster and the gallery disagree — rebuild the gallery |

The run summary counts them and groups them by reason.

## Reading it yourself

```python
import json

records = json.load(open("predictions.json"))

# every call the pipeline is unsure about, worst first
for r in sorted(records, key=lambda r: r["confidence"]):
    if r["confidence"] < 0.4:
        print(r["throw_id"], r["start_time"], r["thrower_name"], r["confidence"])

# throws per player
from collections import Counter
print(Counter(r["thrower_name"] for r in records if r["resolved"]))

# where the two team signals disagree
odd = [r for r in records if r.get("end_agrees_with_half") is False]
print(len(odd), "throws where the ball came from the other team's end")
```
