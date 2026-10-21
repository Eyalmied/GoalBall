# Config files

One file per match. Everything the scripts need to find is in here, so no
command line is longer than it has to be and nothing about a specific game
lives in the source code.

```bash
cp config/example.json config/mygame.json
# edit it, then:
python scripts/run_pipeline.py --config config/mygame.json
```

| key | what it is |
|---|---|
| `name` | free text, shown in the run header |
| `video` | the match video |
| `events` | the upstream throw detector's CSV (only the timestamps are used) |
| `gallery_dir` | where the reference gallery is written / read |
| `court_file` | where the four court corners are written / read |
| `labels` | ground-truth CSV, if you label throws to measure accuracy |
| `predictions` | where the results go |
| `thrower_model` | a goalball-trained detector whose class 0 is `throwing_player`. This is how the thrower is found by default. Defaults to `../Model Weights/best.pt`, the project's own weights. Without it the pipeline falls back to movement and posture |
| `pose_model` | YOLO weight file used to find the people on court. A bare name like `yolov8m-pose.pt` lets ultralytics download it on first use, which is what both example configs do. This is the **generic COCO** model, not a fine-tuned goalball detector |
| `half2_start` | **Required.** Video timestamp where the second half begins, e.g. `"22:00"`. A match is assumed to have two halves, and this is the single source of truth for where it splits — tagging, the court check, labelling and prediction all read it from here. Omit it only if the video is a single half, and then pass `--half 1` or `--half 2` |
| `roster` | the six players: id, name, team. The team labels are what predictions report. |

Relative paths are resolved **against the config file itself**, so a config can
sit anywhere and still point at its data. Any key can be overridden on the
command line (`--video`, `--gallery-dir`, ...).

`paralympics24_isr_chn.json` is the worked example used throughout the docs; it
points at the files one level up from this repo folder.

> **Note on that example's gallery.** It points at a gallery built before home
> zones existed, so it runs but falls back to naming players by appearance —
> the weaker path, and the run says so loudly. To get the position-based
> numbers, rebuild it once:
>
> ```bash
> python scripts/build_gallery.py --config config/paralympics24_isr_chn.json >        --half 1 --rounds 5
> ```
>
> That rewrites the same folder with home zones and records the half.
