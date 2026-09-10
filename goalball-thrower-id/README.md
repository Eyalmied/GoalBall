# Goalball thrower identification

Given a match video and the throw timestamps from an upstream throw detector,
work out **which player threw each ball**, on which team, and how sure we are.

```
  [12/191]   00:34  Maya (ISR)       ████████░░ 0.85  5 frames agreed · ball seen
```

Goalball defeats every normal way of identifying a person on video: both teams
wear identical uniforms, every player wears blackout eyeshades, and at
broadcast distance a jersey number is about twenty pixels tall and usually
behind the net. No faces, no numbers, and colour only ever tells you the team.

So this does something else. It works out **which player** from where the throw
came from, matched against home zones recorded once by hand — and the team
comes free, because the two teams stand at opposite ends of the court. Posture
tells it which team is attacking (defenders throw themselves flat to block,
attackers stay on their feet), which is what picks the thrower out of the
players on court. Nothing depends on seeing a face or reading a number.

| | Measured on 85 hand-labelled throws |
|---|---|
| Correct team | **97.6%** |
| Correct thrower | **74.1%** |
| Strictly video-only | ~50% |
| Guessing | 17% |

About 77% is the arithmetic ceiling, and
[docs/RESULTS.md](docs/RESULTS.md) shows why: two teammates genuinely throw
from overlapping parts of the court, and nothing at this resolution separates
them.

---

## Try it

```bash
python -m pip install -r requirements.txt
python -m pip install torch torchreid tensorboard     # the appearance model
python scripts/selftest.py                            # checks the install and the maths

cp config/example.json config/mygame.json             # edit paths + roster
python scripts/run_pipeline.py --config config/mygame.json --half 1
```

`run_pipeline.py` is the one command to remember. It works out what is already
done, does what is missing, and stops with a plain explanation if it needs you:

1. **tag the players** once — opens a window, about five minutes
2. **mark the court** once — click four corners
3. **predict** every throw — about a second each
4. **measure it**, if you have labelled some throws

Step by step, with every key press and every flag:
**[docs/USAGE.md](docs/USAGE.md)**.

---

## Say which half

```bash
python scripts/run_pipeline.py --config config/mygame.json --half 1
python scripts/run_pipeline.py --config config/mygame.json --half 2
python scripts/run_pipeline.py --config config/mygame.json --half auto --half2-start 22:00
```

Teams change ends at half-time, which moves every player's home zone to the
opposite end of the court. Since identity *is* position here, telling it the
wrong half does not make the answer slightly worse — it inverts it.

Tag the gallery inside one half and name it (`build_gallery.py --half 1`); it
is stored with the gallery, and every later run mirrors the zones as needed.
`--half auto` handles a video that covers the whole match in one pass.

Every run prints what it worked out, and it is worth reading those two lines:

```
    half                   both (switching at 22:00)
    gallery tagged in      half 1
    half 1                 home zones as tagged   CHN at the bottom,  ISR at the top
    half 2                 home zones MIRRORED (teams changed ends)   CHN at the top,  ISR at the bottom
```

Full detail, including how the flip is computed and how to spot a wrong
setting: **[docs/HALVES.md](docs/HALVES.md)**.

---

## What you get

`predictions.json`, one record per throw:

```json
{
  "throw_id": "throw_12",
  "start_time": "00:34",
  "half": 1,
  "home_zones_mirrored": false,
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
  "ball_seen": true,
  "release_position": [0.34, 0.39],
  "ball_came_from_team_end": "ISR",
  "end_agrees_with_half": true,
  "csv_team_hint": "ISR",
  "resolved": true
}
```

Confidence carries real information — the calls at 0.70 and above are right far
more often than the ones below — so sort by it and review the bottom:

```bash
python scripts/review_predictions.py --config config/mygame.json --only-uncertain 0.4
```

---

## The scripts

| | |
|---|---|
| `scripts/run_pipeline.py` | **the whole flow, one command** |
| `scripts/build_gallery.py` | step 1 — tag the players (the only manual step) |
| `scripts/set_court.py` | step 2 — click the four court corners |
| `scripts/predict_throwers.py` | step 3 — name the thrower for every throw |
| `scripts/evaluate.py` | step 4 — accuracy, per half, per confidence, confusions |
| `scripts/label_throwers.py` | optional — record who really threw, to measure against |
| `scripts/review_predictions.py` | look at the calls next to the truth |
| `scripts/selftest.py` | check the install and the geometry, no video needed |

Every script takes `--help`, `--verbose`, and the same `--config`.

---

## The documentation

| | |
|---|---|
| **[docs/USAGE.md](docs/USAGE.md)** | run it, step by step, including what each window wants |
| **[docs/HALVES.md](docs/HALVES.md)** | halves and changing ends — the one thing to get right |
| **[docs/DESIGN.md](docs/DESIGN.md)** | how it works and why it is built this way |
| **[docs/RESULTS.md](docs/RESULTS.md)** | what was measured, and what it can realistically reach |
| **[docs/OUTPUT.md](docs/OUTPUT.md)** | every field in the output file |
| **[config/README.md](config/README.md)** | every config key |

---

## What it needs

- Python 3.9 or newer, and a display for the two setup steps
- The match video, and the throw detector's CSV — **only the timestamps** are
  used to decide a thrower
- One appearance model: `torchreid` (better) or `transformers` (easier)
- No GPU required; roughly a second per throw on a laptop CPU

---

## What is deliberately not here

- **No match-specific values in the source.** Roster, paths and the half-time
  timestamp live in a config file; home zones and the tagged half live in the
  gallery; the court lives in its own file; resolution and frame rate come from
  the video.
- **Nothing from the CSV but timestamps** to decide the thrower. The upstream
  detector's own team call is kept per record as `csv_team_hint`, for
  comparison, and never used. Its ball release point is opt-out and only ever
  sets the thrower's *position*, never the team.
- **No fitted constants.** Every weight is a documented flag with a measured
  default.
