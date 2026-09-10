# Usage — start to finish

Everything you need to go from a match video to a scored list of throwers. No
knowledge of the internals is assumed; follow the steps in order.

**What you end up with:** a JSON file saying, for every throw, *which player
threw it, on which team, and how sure the pipeline is* — plus a viewer to check
the calls against the video.

---

## 0. Before you start

### 0.1 What you need

| | |
|---|---|
| The match video | any format OpenCV can read |
| The throws CSV | from the upstream throw detector, `..._Throws_data_predicted.csv`. Only the timestamps are used to decide a thrower |
| A display | two setup steps open a window, so do them on a desktop, not over SSH |
| Roughly 15 minutes | ~5 to tag the players, ~1 to mark the court, the rest is the machine's |

### 0.2 Install

```bash
python -m pip install -r requirements.txt
```

That covers the detector and the pipeline. Then pick **one** appearance model:

```bash
# recommended
python -m pip install torch torchreid tensorboard

# easier to install, slightly weaker — then pass --appearance dinov2
python -m pip install torch transformers pillow
```

Check it worked:

```bash
python scripts/selftest.py
```

### 0.3 Describe your match once

Copy the example config and edit it:

```bash
cp config/example.json config/mygame.json
```

```json
{
  "name":   "My match",
  "video":  "../mygame.mp4",
  "events": "../mygame_Throws_data_predicted.csv",
  "gallery_dir": "gallery_mygame",
  "court_file":  "court_mygame.json",
  "labels":      "labels_mygame.csv",
  "predictions": "predictions_mygame.json",
  "half2_start": "22:00",
  "roster": {
    "1": {"name": "Ada",  "team": "ISR"},
    "2": {"name": "Beth", "team": "ISR"},
    "3": {"name": "Cara", "team": "ISR"},
    "4": {"name": "Dana", "team": "CHN"},
    "5": {"name": "Eve",  "team": "CHN"},
    "6": {"name": "Fay",  "team": "CHN"}
  }
}
```

The team labels are yours to choose — they are what the predictions report.
Paths are relative to this repository folder. Full key list:
[config/README.md](../config/README.md).

`half2_start` is the video timestamp where the second half begins. Find it by
scrubbing to the restart after the interval; a few seconds either way is fine,
as long as it falls inside the break.

---

## The short version

```bash
python scripts/run_pipeline.py --config config/mygame.json \
       --half auto --half2-start 22:00
```

That runs every step below, skipping any that is already done. The rest of this
page explains each one, because you will want to know what the windows are
asking of you.

---

## Step 1 — Tag the players *(once per match, opens a window)*

```bash
python scripts/build_gallery.py --config config/mygame.json --half 1
```

You will tag all six players in five different frames. Each tag records what
that player looks like **and where they stand on court** — their home zone.
The home zone is the part that matters: identical uniforms and blackout
eyeshades defeat every appearance model, so position is what separates two
teammates.

**Per player: drag a box around them, then press their number.** When all six
are boxed, press `c`. Repeat for five rounds.

| key | does |
|---|---|
| drag mouse | draw a box around one player |
| `1`–`6` | tag the box you just drew as that player |
| `u` | undo the last tag in this frame |
| `c` / Enter | save this round — only works once everyone is tagged |
| `n` | skip this frame, get another one |
| `q` | stop (rounds already saved are kept) |

Three things worth getting right:

- **Include the feet.** The bottom edge of your box is the position that gets
  recorded. A box cut off at the knees puts the player metres up the court.
- **Skip messy frames.** If players overlap, press `n`. Only complete rounds
  are saved, so everyone ends up with the same number of good references.
- **`--half` matters.** Tag inside one half and say which one. With
  `half2_start` in your config the builder only offers you frames from that
  half, so you cannot mix them by accident. See [HALVES.md](HALVES.md).

When it finishes it reads the gallery back and prints which end each team is
on. **Check that against the video.** If it is swapped, you tagged the other
half — re-run with the other `--half`.

---

## Step 2 — Mark the court *(once per camera angle, opens a window)*

```bash
python scripts/set_court.py --config config/mygame.json --at 07:30
```

Click the four corners of the playing surface, then `c`.

This fences out crowd, referees, coaches and substitutes — anyone whose feet
land outside the shape is not a player. The same four corners also define the
flat court used to flip home zones at half-time, so this step is what makes
`--half` exact rather than approximate.

Click the **wooden playing area between the goal lines**, not the aprons at the
sides. Click a little tight rather than a little loose: a generous polygon lets
referees back in and stretches the half-time flip.

| key | does |
|---|---|
| click | place a corner (any order) |
| `u` | undo the last corner |
| `c` / Enter | save |
| `s` | skip the fence entirely (not recommended) |
| `q` | quit without saving |

---

## Step 3 — Predict *(the actual work)*

```bash
python scripts/predict_throwers.py --config config/mygame.json \
       --half auto --half2-start 22:00
```

One line per throw:

```
  [12/191]   00:34  Maya (ISR)       ████████░░ 0.85  5 frames agreed · ball seen
```

and a JSON record per throw with the player, the team, the confidence and its
three parts, the half, the release position and the reasoning behind the call.
Every field is listed in [OUTPUT.md](OUTPUT.md).

Worth knowing:

- **The first run pauses** while the models load. After that it is roughly a
  second per throw on a laptop CPU.
- **You can stop it.** The output file is rewritten after every throw; add
  `--resume` to carry on where you left off.
- **`--limit 5`** does a quick five-throw check before committing to the lot.

### The two flags that change the answer

```bash
# default: use the upstream detector's ball release point as the thrower's
# POSITION only. ~74% correct thrower.
--release-source csv

# strictly video-only: nothing from the CSV but the timestamps. ~50%.
--release-source thrower
```

The difference is only where the thrower's position comes from; everything else
is identical. Use `csv` unless you specifically need a video-only result — see
[RESULTS.md](RESULTS.md) for why the gap is so large.

```bash
--appearance osnet      # default, stronger
--appearance dinov2     # if torchreid would not install
```

One more worth knowing about, though the default is right:

```bash
--team-source position   # default: name the thrower against EVERY player's
                         # home zone, and report the winner's own team
--team-source posture    # older behaviour: decide the team from posture first,
                         # then only consider its three players
```

`position` measured better in both halves, and by a wide margin in the half the
gallery was not tagged in. It switches to `posture` on its own if the gallery
has no home zones.

Everything else has a measured-best default. `python scripts/predict_throwers.py
--help` lists the rest with what each one does.

---

## Step 4 — Measure it *(optional)*

To get a number you can trust, label some throws first:

```bash
python scripts/label_throwers.py --config config/mygame.json --half auto
```

Each throw loops with the ball's release point marked and the gallery beside
it. Press the thrower's number, or `0`/`u` if you are not sure. Saved after
every label; stop and resume whenever. About 30 labelled throws is the point
where the figures stop swinging around.

| key | does |
|---|---|
| `SPACE` | play / pause |
| `a` `d` | step one frame back / forward |
| `1`–`6` | this player threw it, then move on |
| `0` `u` | not sure — left out of the accuracy figures |
| `b` `n` | previous throw / next throw without labelling |
| `q` | stop |

Then:

```bash
python scripts/evaluate.py --config config/mygame.json --holdout 0.3
```

```
HELD OUT - this is the number that counts
    labelled throws        20
    of those, predicted    20 (100.0%)

    CORRECT THROWER        17/20 = 85.0%
    correct team           20/20 = 100.0%
    right player, given    17/20 = 85.0%
      the right team
    guessing would give    33.3%
```

(That is a real held-out block from the reference match. Note the sample size:
20 throws means every throw is worth 5 percentage points, so a held-out figure
swings a lot. The all-throws number for the same run is 74.1%.)

`--holdout 0.3` splits the throws the same way every run. Tune against the
"tune" block; only read the "held out" block at the end. Quoting a number you
tuned against measures nothing.

The report also breaks accuracy down **per half** — a big gap between the two
almost always means a wrong `--half`.

---

## Step 5 — Look at it

```bash
python scripts/review_predictions.py --config config/mygame.json
python scripts/review_predictions.py --config config/mygame.json --only-wrong
python scripts/review_predictions.py --config config/mygame.json --only-uncertain 0.4
python scripts/review_predictions.py --config config/mygame.json --export review/
```

The throw plays with two reference panels beside it: what the pipeline chose,
and what you labelled. Green means right, red wrong. One glance tells you which
half failed — wrong team, or the wrong teammate.

`--only-uncertain 0.4` is the useful one when you have no labels: it shows the
calls the pipeline itself is least sure of, which is where the errors are.

---

## A different match

Nothing is tuned to any one game. Write a new config and repeat:

```bash
cp config/example.json config/othergame.json      # edit roster + paths
python scripts/run_pipeline.py --config config/othergame.json --half 1
```

Teams, names, home zones, court and timestamps all come from that game's own
setup files.

---

## What "working" looks like

| | |
|---|---|
| Correct team | around **97%** |
| Correct thrower | around **74%** with `--release-source csv`, around **50%** video-only |
| Confidence | means something: the calls at 0.7 and above are mostly right. `evaluate.py` prints accuracy split by confidence so you can check that on your own footage |
| Posture vs the end | 90%+ agreement between the posture team and the end the ball came from. Lower means either the half is wrong or one team is too far from the camera for the kit colours to separate — the answer's team comes from position, so it survives, but it is worth knowing |

Around 77% is close to the ceiling, and the reason is arithmetic rather than
effort — see [RESULTS.md](RESULTS.md) before trying to push past it.

---

## When something is wrong

| What you see | What to do |
|---|---|
| `no gallery found at ...` | run step 1 |
| `no court file at ...` (a warning, not fatal) | run step 2; without it crowd and referees are considered players and the half-time flip is only approximate |
| `the OSNet appearance model is not installed` | `pip install torchreid torch tensorboard`, or use `--appearance dinov2` |
| `this gallery has no home zones` | it was built by an older version — re-run step 1 to get position-based naming (~40% → ~70%) |
| Lots of wrong **teams** | the court polygon is too loose and referees are being counted. Re-run step 2, tighter |
| Teams the wrong way round throughout | the gallery was tagged in the other half. Re-run step 1 with the other `--half`, or pass `--gallery-half` |
| One half much worse than the other | `half2_start` is wrong, or `--half` does not match the footage. See [HALVES.md](HALVES.md) |
| The wrong **teammate**, often | the known hard case: two players whose home zones overlap. More gallery images will not fix it — [RESULTS.md](RESULTS.md) explains why |
| Some throws get **no call** | nobody was detected in that window (players occluded). Those come out with no thrower rather than a guess; the run summary counts them |
| `posture vs the end` below 75% | either `--half` is wrong for this footage, or one team is too far from the camera for the kit colours to separate. The answer's team comes from court position and survives that, but a video-only run (`--release-source thrower`) would not |

Add `--verbose` to any script to see the internals, including the framework
output that is normally hidden.
