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

So this does something else. A goalball-trained detector points at the player
who is throwing; we take **where they are standing** and match it against home
zones you recorded once by hand. The team comes free, because the two teams
stand at opposite ends of the court. Nothing depends on seeing a face or
reading a number.

When the detector finds nobody — about one throw in six — it falls back to the
two cues that give the thrower away anyway: **the thrower is the one moving**,
and **the thrower is on their feet** while the defenders are already flat.

| | Measured on 181 hand-labelled throws, both halves |
|---|---|
| Correct team | **99.4%** |
| Correct thrower | **72.4%** |
| half 1 / half 2 | 74.4% / 70.5% |
| Guessing | 17% / 50% |

Around 80% is the arithmetic ceiling, and
[docs/RESULTS.md](docs/RESULTS.md) shows why: in 17% of throws a player
genuinely throws from a team-mate's spot, and nothing at this resolution
separates them.

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

1. **tag the players** once — opens a window, about eight minutes (`h` switches half)
2. **mark the court** once — click four corners
3. **predict** every throw — about a second each
4. **measure it**, if you have labelled some throws

Every command in order, with what to check after each:
**[docs/RUNBOOK.md](docs/RUNBOOK.md)**.

---

## Say which half

```json
{ "half2_start": "22:00" }
```

```bash
python scripts/run_pipeline.py --config config/mygame.json
```

A match is assumed to have **two halves**, so every run covers both and splits
them at `half2_start` — the single source of truth for when the ends change.
Every step reads it from there, so none of them can disagree.

It is required for that reason: leave it out and the run stops rather than
quietly treating the whole video as one half, which is what matches
second-half throws against the wrong end of the court. If your video really is
a single half, say so with `--half 1` or `--half 2`.

Teams change ends at half-time, which moves every player's home zone to the
opposite end of the court. Since identity *is* position here, telling it the
wrong half does not make the answer slightly worse — it inverts it.

When tagging, `h` switches between the halves, so you can take each player's
reference images from wherever they are clearest. Every position is stored with
the half it came from: tag a player in both and they get a *measured* zone for
each, tag them in one and the other is mirrored. A video covering the whole
match is then handled in one pass.

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
  "decided_by": "court position (75%) plus appearance",
  "located_by": "the goalball detector",
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
| **[docs/RUNBOOK.md](docs/RUNBOOK.md)** | **every command in order, with what to check after each** |
| **[docs/USAGE.md](docs/USAGE.md)** | the same ground at more length, with the reasoning |
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
- One appearance model: `torchreid` (better) or `transformers` (easier). The
  gallery's own labels are used to learn which part of the embedding actually
  tells these players apart — without that step the raw embedding puts two
  photos of one player further apart than two different players
- Optionally the project's `Model Weights/best.pt`, whose `throwing_player`
  class picks the thrower directly. Without it the pipeline falls back to
  movement and posture, at a cost of about 22 points
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
