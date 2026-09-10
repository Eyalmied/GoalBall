# Goalball Sports Analytics

Computer-vision pipeline for Paralympic Goalball: YOLOv8 player/ball detection → LSTM throw-outcome prediction → Flask dashboard.

---

## Repository Structure

```
GoalBall/
├── predict_pipeline_with_YAMNet.py   ← Main inference script (YAMNet crowd-noise scoring)
├── predict_pipeline.py               ← Inference script without YAMNet
├── requirements.txt
│
├── Model Weights/                    ← Deployment model files (used by both pipelines)
│   ├── best.pt                       ← YOLOv8 deployment weights (~50 MB)
│   ├── final_model.pt                ← LSTM 23f baseline model (~2.4 MB)
│   ├── scaler.pkl                    ← StandardScaler for 23f model
│   ├── final_model_yamnet.pt         ← LSTM 32f YAMNet model (~2.5 MB)
│   ├── scaler_yamnet.pkl             ← StandardScaler for 32f model
│   └── yamnet_config.json            ← YAMNet crowd class config (n_crowd=8, indices)
│
├── ball+players_tuning10/            ← Pre-trained base checkpoint; YOLO fine-tuning starts from here
│   └── weights/best.pt
│
├── app/                              ← Flask analytics dashboard
│   ├── app.py
│   ├── config.py                     ← Data paths (relative, no editing needed)
│   ├── requirements.txt
│   ├── static/
│   ├── templates/
│   └── data/                         ← Game data (Excel + CSV)
│       ├── GoalPredictions_AllGames.csv
│       └── Paralkympics2024/
│           └── <GAME>/outputs/<GAME>_Throws_data.xlsx
│
├── goalball-thrower-id/              ← Who threw each ball (see below); takes the throw
│   │                                    timestamps from this pipeline's CSV
│   ├── README.md
│   ├── scripts/run_pipeline.py       ← One command, start to finish
│   ├── goalball/                     ← The library behind it
│   ├── docs/                         ← Usage, design, results, halves, output format
│   └── config/                       ← One JSON per match (paths + roster)
│
└── Train Model/
    ├── CNN YOLOv8 Finetune/          ← YOLO annotation pipeline + two independent training options
    │   ├── 1_mov_to_frames.py        ← Data prep step 1: extract frames from video for CVAT
    │   ├── 2_class2to32.py           ← Data prep step 2: remap class IDs + populate full_data/
    │   ├── yolo_loocv.py             ← Training option A: 7-fold LOOCV — evaluate model performance
    │   ├── yolo_final.py             ← Training option B: train on all data → Model Weights/best.pt
    │   ├── yolo_cnn_predict_2.py     ← LSTM data tool: use trained YOLO to label game videos → CSVs
    │   ├── yolo_cnn_LIVE.py          ← LSTM data tool: same as above but for a live camera feed
    │   └── full_data/full_data.yaml
    │
    └── LSTM Training/                ← LSTM training — two independent model variants
        ├── data_preperation with attention.ipynb
        ├── data_preperation with attention only goals.ipynb
        │
        ├── Baseline LSTM/            ← 23 visual features only (no audio)
        │   ├── k-fold.py             ← LOOCV → mean acc 0.776
        │   └── train_final.py        ← Train all data → Model Weights/final_model.pt
        │
        └── LSTM+YAMNet/              ← 32 features: 23 visual + 8 YAMNet + yam_rel
            ├── extract_audio_features.py  ← Step 0 (run once): extract YAMNet features → *_yamnet.csv
            ├── k-fold.py             ← LOOCV → mean acc 0.800
            └── train_final.py        ← Train all data → Model Weights/final_model_yamnet.pt
```

---

## Quick Start — Run the Prediction Pipeline

### Requirements

```bash
pip install -r requirements.txt
```

`ffmpeg` is also required for YAMNet crowd-noise scoring (it extracts audio from video files). It is a system tool — install it once with **one** of the following, then restart your terminal:

| Environment | Command |
|-------------|---------|
| Conda (recommended) | `conda install -c conda-forge ffmpeg` |
| Windows (winget) | `winget install ffmpeg` |
| Windows (Chocolatey) | `choco install ffmpeg` |

Verify it works: `ffmpeg -version`. If you skip this step the pipeline still runs but every Crowd Noise Score will be 0.0.

### Run on a recorded game video

```bash
python predict_pipeline_with_YAMNet.py
```

The script prompts you interactively for:

1. Path to the `.mp4` / `.mov` video file
2. Lower-side team name (team playing at the bottom of the screen pre-halftime)
3. Upper-side team name
4. Halftime timestamp (`hh:mm:ss` or `mm:ss`)
5. **8-click goal-zone calibration** — click the corners of both goal frames in this order:

```
lower right bottom → lower right top → lower left bottom → lower left top
upper left bottom  → upper left top  → upper right bottom → upper right top
```

After YOLO processing you validate each detected segment (press `1` = correct, `0` = false positive) and label each "to" segment with the outcome (`g` = goal, `b` = block, `o` = out).

### Run on a live camera feed

When prompted for the video path, enter `LIVE` (or a camera index such as `0`, `1`, `2`):

```
Video path  (or 'LIVE' / camera index 0, 1, 2… for live camera): LIVE
```

**Differences from recorded-video mode:**

| Step | Live mode behaviour |
|------|---------------------|
| Halftime | No timestamp prompt — press **T** in the preview window at the real halftime whistle; teams switch sides automatically |
| Preview | Always shown during detection (YOLO bounding boxes drawn in real time) |
| Stop | Press **Q** in the preview window — recording stops and the full LSTM analysis pipeline runs immediately |
| Calibration | Uses the first frame grabbed from the camera |
| Audio / YAMNet | Microphone is recorded in parallel via `sounddevice` and scored by YAMNet exactly like a video file (included in `requirements.txt`) |

The recorded video is saved as `Pipeline_Outputs/<GAME>/<GAME>_live_recording.mp4` alongside the usual outputs.

**Outputs** (written to `Pipeline_Outputs/<GAME>/`):

| File | Contents |
|------|----------|
| `<GAME>_Throws_data_predicted.xlsx` | Per-throw summary: zones, teams, LSTM prediction, Crowd Noise Score |
| `<GAME>_Throws_lstm_inference.csv` | Per-frame feature CSV for LSTM (re)training |

---

## Flask Dashboard

```bash
cd app
pip install -r requirements.txt
python app.py
# Opens at http://127.0.0.1:5000
```

Pages:

| Route | Description |
|-------|-------------|
| `/` | Overview — filter by game / team |
| `/goals` | Goal predictions table + zone heatmap + release-time chart |
| `/throws` | Per-game throw explorer with from/to zone breakdown |

The dashboard reads data from `app/data/` — the folder is already populated with the 6 Paris 2024 Paralympic games. To add a new game, run the prediction pipeline and copy the resulting `_Throws_data.xlsx` into `app/data/Paralkympics2024/<GAME>/outputs/` then add the LSTM-predicted goals to `GoalPredictions_AllGames.csv`.

---

## Train Model — CNN YOLOv8 Finetune

Located in `Train Model/CNN YOLOv8 Finetune/`.

This folder has three distinct purposes — data preparation, model training, and LSTM data generation — which are independent of each other.

### Data preparation (run once per new game, in order)

**Step 1 — Extract frames**

Edit the two path variables at the top of `1_mov_to_frames.py` then run it. Saves one JPEG per second named `<GAME>_F_image<N>.jpg` — the `_F_image` marker lets `yolo_loocv.py` group images by game.

```bash
python "Train Model/CNN YOLOv8 Finetune/1_mov_to_frames.py"
```

**Step 2 — Annotate in CVAT and remap class IDs**

Upload the frames to [CVAT](https://www.cvat.ai/) and annotate:
- Class 0 → `throwing_player`
- Class 1 → `defending_player`
- Class 32 → `sports_ball` (CVAT exports this as class 2 — the next script fixes it)

Export as **YOLO format** (images + `.txt` labels), then run:

```bash
python "Train Model/CNN YOLOv8 Finetune/2_class2to32.py"
```

Remaps class 2 → 32 and copies everything into `full_data/images/train/` and `full_data/labels/train/`.

### Model training (two independent options — pick one or both)

**Option A — Evaluate via LOOCV**

```bash
python "Train Model/CNN YOLOv8 Finetune/yolo_loocv.py"
```

Runs 7-fold leave-one-game-out cross-validation to measure generalisation performance. Produces metrics and per-fold results in `yolo_runs/loocv_results.csv`. Does **not** produce a deployment model.

**Option B — Train the deployment model**

```bash
python "Train Model/CNN YOLOv8 Finetune/yolo_final.py"
```

Trains on **all** annotated images and automatically copies the best checkpoint to `Model Weights/best.pt`, ready to deploy in the pipeline immediately.

**Key hyperparameters** (top of both scripts):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `TRAIN_EPOCHS` | 60 | Early-stopped via `PATIENCE=20` |
| `IMG_SIZE` | 960 | Higher = better small-object detection |
| `BATCH` | 8 | Reduce to 4 if GPU OOM |
| `LR0` | 0.0001 | Gentle LR for incremental fine-tuning |
| `FREEZE` | 10 | Freeze first 10 backbone layers |
| `DEVICE` | 0 | GPU index; `'cpu'` for no-GPU machines |

**LOOCV results (Paris 2024, 7 folds):** mean mAP@0.5 = **0.974**

### Generating LSTM training data from game videos

`yolo_cnn_predict_2.py` and `yolo_cnn_LIVE.py` are **not** YOLO training tools — they use the already-trained YOLO model (`Model Weights/best.pt`) to process a game video, let you interactively validate and label each detected throw segment, and output the per-frame CSVs that the LSTM trains on.

```bash
python "Train Model/CNN YOLOv8 Finetune/yolo_cnn_predict_2.py"   # recorded video
python "Train Model/CNN YOLOv8 Finetune/yolo_cnn_LIVE.py"        # live camera
```

Both write:
- `<GAME>_Throws_data.xlsx` — per-throw summary
- `<GAME>_Throws_lstm_training.csv` — per-frame feature CSV for LSTM training

Edit `game`, `video_path`, `excel_output_path`, and `lstm_csv_path` at the top of the file before running.

---

## Train Model — LSTM Training

Located in `Train Model/LSTM Training/`.

These scripts train the throw-outcome classifier. They require per-game LSTM training CSVs produced by the prediction pipeline.

**Configure paths** at the top of each script:

```python
DATA_ROOT = Path(r"C:\path\to\Paralkympics2024")          # folder with per-game sub-dirs
GOALS_DIR = Path(r"C:\path\to\Goals_Paralympics\outputs") # goal-clip CSV folder
```

There are **two independent LSTM variants**. Train whichever you need, or both. `predict_pipeline_with_YAMNet.py` loads both at startup and auto-selects between them at runtime.

### Option A — Baseline LSTM (23 visual features, no audio)

```bash
# Evaluate:
python "Train Model/LSTM Training/Baseline LSTM/k-fold.py"

# Train deployment model → Model Weights/final_model.pt
python "Train Model/LSTM Training/Baseline LSTM/train_final.py"
```

**Input:** 23 visual features per frame (ball/thrower/defender x,y,w,h,conf + velocity + visibility flags + relative time)  
**LOOCV result:** mean accuracy = **0.776**

### Option B — LSTM+YAMNet (32 features: visual + crowd audio)

**Step 0 — Extract audio features (run once before k-fold or train_final):**

```bash
python "Train Model/LSTM Training/LSTM+YAMNet/extract_audio_features.py"
```

Runs YAMNet on every game video and goal clip. Requires `ffmpeg` + `tensorflow` / `tensorflow-hub`. Runtime: ~5–10 min. Outputs `*_yamnet.csv` files alongside the scripts — loaded automatically by the training scripts.

```bash
# Evaluate:
python "Train Model/LSTM Training/LSTM+YAMNet/k-fold.py"

# Train deployment model → Model Weights/final_model_yamnet.pt + scaler_yamnet.pkl
python "Train Model/LSTM Training/LSTM+YAMNet/train_final.py"
```

**Input:** 32 features = 23 visual + 8 YAMNet crowd-class probs (`yam_0..yam_7`) + 1 relative crowd score (`yam_rel`)  
**`yam_rel`:** sum of 8 crowd probs per frame ÷ max across the full game — volume-invariant, peak always = 1.0  
**LOOCV results:**

| Model | Mean Acc | g1 Precision | g1 Recall | g1 F1 |
|-------|----------|-------------|-----------|-------|
| Baseline (23f) | 0.776 | 0.407 | 0.422 | 0.414 |
| 31f (+ yam_0..7) | 0.782 | 0.625 | 0.612 | 0.619 |
| **32f (+ yam_rel)** | **0.800** | 0.574 | **0.633** | 0.602 |

**Both models share the same BiLSTM architecture** (128 hidden, 2 layers, self-attention, focal loss) — only the input dimension differs.  
**Automatic routing in `predict_pipeline_with_YAMNet.py`:** if the game's peak YAMNet crowd score > 0.1 → 32f model is used; otherwise → 23f baseline. No manual selection needed.

---

## Thrower Identification — who threw each ball

`goalball-thrower-id/` answers a different question from the rest of this repo:
not *what happened to the throw*, but *which player made it*.

It takes only the **throw timestamps** from this pipeline's
`..._Throws_data_predicted.csv` and works out the rest from the video.

Goalball defeats the usual approaches — both teams wear identical uniforms,
every player wears blackout eyeshades, and at broadcast distance a jersey
number is about twenty pixels tall and usually behind the net. So instead it
identifies a player by **where the throw came from**, matched against home
zones recorded once by hand, and the team falls out of the name because the two
teams stand at opposite ends of the court. Posture (defenders throw themselves
flat to block, attackers stay on their feet) picks the thrower out of the
players on court.

| Measured on 85 hand-labelled throws | |
|---|---|
| Correct team | **97.6%** |
| Correct thrower | **74.1%** |
| Strictly video-only (no CSV field but the timestamps) | ~50% |
| Guessing | 17% |

```bash
cd goalball-thrower-id
pip install -r requirements.txt
pip install torch torchreid tensorboard      # the appearance model
python scripts/selftest.py                   # checks the install and the geometry

python scripts/run_pipeline.py --config config/paralympics24_isr_chn.json \
       --half auto --half2-start 22:00
```

That runs every step, skipping any already done: tag the players once (a
window, ~5 minutes), click the four court corners once, predict every throw
(~1 s each on a laptop CPU), and measure the accuracy if any throws have been
labelled.

**Teams change ends at half-time**, which moves every player's home zone to the
opposite end of the court — so the half is an explicit input (`--half 1`,
`--half 2`, or `--half auto --half2-start MM:SS`). Player ids, names and teams
never change; only the positions do.

Output is one JSON record per throw with the player, the team, a confidence in
three readable parts, and the reasoning behind the call. Confidence is
informative — calls at 0.70 and above are right 77.6% of the time against
61.1% below — so weak calls can be reviewed rather than trusted.

| | |
|---|---|
| [goalball-thrower-id/README.md](goalball-thrower-id/README.md) | overview |
| [docs/USAGE.md](goalball-thrower-id/docs/USAGE.md) | step by step, every key press and flag |
| [docs/HALVES.md](goalball-thrower-id/docs/HALVES.md) | halves and changing ends |
| [docs/DESIGN.md](goalball-thrower-id/docs/DESIGN.md) | how it works and why |
| [docs/RESULTS.md](goalball-thrower-id/docs/RESULTS.md) | what was measured, and the ceiling |
| [docs/OUTPUT.md](goalball-thrower-id/docs/OUTPUT.md) | every field in the output file |

It shares no code with the pipelines above and needs no changes to them: the
two generic COCO YOLO models it uses (people/skeletons and the ball) are
downloaded on first run, separate from `Model Weights/`.

---

## Model Files

All deployment weights live in `Model Weights/`:

| File | Description | Size |
|------|-------------|------|
| `Model Weights/best.pt` | YOLOv8 deployment weights | ~50 MB |
| `Model Weights/final_model.pt` | LSTM 23f baseline model (used when no crowd audio) | ~2.4 MB |
| `Model Weights/scaler.pkl` | StandardScaler for 23f model | <1 KB |
| `Model Weights/final_model_yamnet.pt` | LSTM 32f YAMNet model (used when crowd audio detected) | ~2.5 MB |
| `Model Weights/scaler_yamnet.pkl` | StandardScaler for 32f model | <2 KB |
| `Model Weights/yamnet_config.json` | YAMNet crowd class indices (n_crowd=8, crowd_idx list) | <1 KB |

`train_final.py` saves directly to `Model Weights/final_model_yamnet.pt` (and `scaler_yamnet.pkl`). `yolo_final.py` copies the best checkpoint to `Model Weights/best.pt` automatically after training.

The prediction pipeline automatically routes each game to the correct LSTM model: if the peak YAMNet crowd score across the full game exceeds 0.1, the 32f YAMNet model is used; otherwise the 23f baseline is used. The two models are loaded at startup — no manual selection required.

---

## Dependencies

```bash
pip install ultralytics opencv-python pandas numpy openpyxl torch scikit-learn \
            torchmetrics matplotlib seaborn Flask plotly tensorflow tensorflow-hub
# ffmpeg (system package) for YAMNet audio extraction
```
