# Runbook

Every command, in order, with what to check before moving on. If you just want
to run the thing, this is the page. For *why* any of it works, see
[DESIGN.md](DESIGN.md); for the numbers, [RESULTS.md](RESULTS.md).

Each step says whether it needs you at the keyboard. Two of them open a window;
the rest you can leave running.

---

## Before you start

| you need | notes |
|---|---|
| the match video | the same resolution the throw detector ran on |
| the throws CSV | `..._Throws_data_predicted.csv`; only the timestamps are used |
| a display | for steps 2 and 3 only |
| ~15 minutes of your time | the machine does the rest |

```bash
cd goalball-thrower-id
python -m pip install -r requirements.txt
python -m pip install torch torchreid tensorboard      # the appearance model
```

---

## Step 1 — describe the match *(once, 2 minutes, no window)*

```bash
cp config/example.json config/mygame.json
```

Edit it: the video and CSV paths, the six players, and `half2_start` — the
video timestamp where the second half begins. Find it by scrubbing to the
restart after the interval; anywhere inside the break is fine.

```json
{
  "video":  "../mygame.mp4",
  "events": "../mygame_Throws_data_predicted.csv",
  "gallery_dir": "gallery_mygame",
  "court_file":  "court_mygame.json",
  "labels":      "labels_mygame.csv",
  "predictions": "predictions_mygame.json",
  "thrower_model": "../Model Weights/best.pt",
  "half2_start": "22:00",
  "roster": {
    "1": {"name": "Ada",  "team": "ISR"},
    "...": {}
  }
}
```

`half2_start` is **required** — a match is assumed to have two halves. If your
video really is a single half, leave it out and pass `--half 1` everywhere.

**Check:**

```bash
python scripts/selftest.py --config config/mygame.json
```

Everything should say `yes` except the gallery, court and labels, which say
`not yet`. If the thrower model says `NO`, see *Troubleshooting*.

---

## Step 2 — mark the court *(once per camera angle, 1 minute, opens a window)*

```bash
python scripts/set_court.py --config config/mygame.json --at 07:30
```

Click the four corners of the playing surface, then `c`.

| key | does |
|---|---|
| click | place a corner (any order) |
| `u` | undo the last corner |
| `c` / Enter | save |
| `q` | quit without saving |

Click the **wooden area between the goal lines**, not the side aprons, and
err tight rather than loose — a generous polygon lets referees and
substitutes count as players.

**Check:** the run prints which end each team is on. It should match the video.

---

## Step 3 — tag the players *(once per match, 8 minutes, opens a window)*

```bash
python scripts/build_gallery.py --config config/mygame.json --rounds 5
```

Five reference images per player **in each half** — 60 tags in total. Drag a
box around a player, press their number; that files one image for them.

| key | does |
|---|---|
| drag mouse | box one player |
| `1`–`6` | file that box under that player |
| `n` / `b` | next / previous frame in this half |
| **`h`** | **switch half** |
| `u` | undo the last tag |
| `q` | stop (everything tagged is kept) |

Progress is tracked per player per half along the top: `1:4/5 2:5/5 …`, plus
a reminder of what the other half still needs.

Three things that matter:

- **Include the feet.** The bottom edge of your box is the position recorded.
  A box cut off at the knees puts that player metres up the court.
- **Both halves need their own images.** The teams change ends, so a player
  near the camera in one half is at the far end in the other — different size,
  lighting and facing. Press `h` and fill in the other side.
- **Skip bad frames** with `n`. Five good crops beat five rushed ones.

**Check:** it prints each team's end per half and how many zones were measured
where. If the ends look swapped against the video, stop and re-read
[HALVES.md](HALVES.md) before going on.

---

## Step 4 — predict *(3 minutes, no window)*

```bash
python scripts/predict_throwers.py --config config/mygame.json
```

One line per throw:

```
  [12/191]   00:34  Maya (ISR)   ████████░░ 0.85  5 frames agreed · ball not seen
```

Safe to interrupt; add `--resume` to carry on.

**Check the header** before letting it run:

```
learned 3 direction(s) that tell these players apart
how distinct they look 0.41 before, 3.23 after  (above 1 is usable)
thrower picked by      the goalball detector's own 'throwing_player' class
when it finds nobody   movement 40%  standing up 60%
half                   auto - both halves, the default
ends change at         22:00   (from the config file)
half 1                 home zones: 6 measured in this half
half 2                 home zones: 6 measured in this half
```

If `half 2` says *mirrored from the other*, you only tagged one half — go back
to step 3 and press `h`.

> The thrower's position is the one found in the video. The upstream
> detector's CSV release point is in a different coordinate system and scored
> roughly half as well, so it is no longer used for prediction at all — only
> to seed the labelling tool. See [RESULTS.md](RESULTS.md) §3a.

---

## Step 5 — label some throws *(optional, opens a window)*

Only needed to **measure** accuracy. Predictions work without it.

```bash
python scripts/label_throwers.py --config config/mygame.json
```

| key | does |
|---|---|
| `1`–`6` | this player threw it, then move on |
| `0` / `u` | not sure — left out of the figures |
| `SPACE` | play / pause |
| `a` / `d` | step one frame |
| `b` / `n` | previous throw / skip |
| `q` | stop |

Saves after every label; `--skip-labelled` jumps to the ones you have not done.
About 30 labelled throws makes the figures meaningful; spread them across both
halves and across players.

---

## Step 6 — measure *(seconds, no window)*

```bash
python scripts/evaluate.py --config config/mygame.json --holdout 0.3
```

Omit `--holdout` for "how did it do on everything". Use it when you have been
changing settings — the held-out block is the honest number.

**What good looks like**, on the reference match:

| | |
|---|---|
| correct team | ~99% |
| correct thrower | ~69% |
| a big gap between halves | something is wrong with the half setup |

---

## Step 7 — look at the mistakes *(opens a window)*

```bash
python scripts/review_predictions.py --config config/mygame.json --only-wrong
python scripts/review_predictions.py --config config/mygame.json --only-uncertain 0.4
python scripts/review_predictions.py --config config/mygame.json --export review/
```

`SPACE` play/pause · `a`/`d` step a frame · `n`/`b` next/previous · `q` quit.
Green border = right, red = wrong, with predicted and true player side by side.

`--only-uncertain` is the one to use when you have no labels: it shows the
calls the pipeline itself doubts, which is where the errors are.

---

## All of it in one command

```bash
python scripts/run_pipeline.py --config config/mygame.json
```

Runs steps 2, 3, 4 and 6, skipping whatever is already done. Add
`--redo gallery,court` to force a setup step, `--resume` to continue an
interrupted prediction, `--dry-run` to see the plan without doing anything.

---

## Running it again

| you changed | re-run |
|---|---|
| nothing, just want the results again | step 6 and 7 |
| the court corners | steps 2, 4 |
| the roster, or want better references | steps 3, 4 |
| a tuning flag | step 4 (delete the predictions file first, or use `--redo predict`) |
| the video or the CSV | everything |

The gallery and the court survive everything except a change to the roster or
the camera angle. Nothing is overwritten without telling you.

---

## Troubleshooting

| what you see | what to do |
|---|---|
| `no gallery found at ...` | run step 3 |
| `no court file at ...` (a warning) | run step 2; without it crowd and referees count as players |
| `no goalball detector at ...` | the thrower falls back to movement + posture, 46% against 69%. Point `--thrower-model` at the project's `Model Weights/best.pt` |
| `a match is treated as two halves, but nothing says when the ends change` | add `half2_start` to the config, or pass `--half 1` if the video is a single half |
| `this gallery has no home zones` | it was built by an older version — re-run step 3 |
| `how distinct they look` ends below 1.0 | the reference images are too varied to tell the players apart. Re-tag with clearer crops, or run with `--position-weight 1.0` |
| `half 2 … mirrored from the other` | you tagged only one half; press `h` during step 3 |
| teams reported the wrong way round | you tagged in the other half — re-run step 3 with `--half 2` |
| one half much worse than the other | `half2_start` is wrong, or that half has no references |
| lots of wrong **teams** | the court polygon is too loose; re-run step 2 tighter |
| the wrong **team-mate**, often | the known hard case — 17% of throws genuinely come from a team-mate's spot. See [RESULTS.md](RESULTS.md) §5 |
| some throws get **no call** | nobody was detected in that window. They are reported, not guessed; the summary counts them |

Add `--verbose` to any script to see the internals, including the framework
output that is normally hidden.
