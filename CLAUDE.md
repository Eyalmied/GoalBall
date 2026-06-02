# CLAUDE.md — GoalBall Analytics Platform

This file is the authoritative guide for any LLM (Claude Code or other) working in this repository. It documents every file, its purpose, how to run it, what inputs it expects, what it outputs, and how the pieces fit together. Read this file fully before making any changes or answering operational questions.

---

## Project Overview

**GoalBall Analytics** is a Paralympic sport analytics platform built for a master's thesis at Tel Aviv University. It automates match analysis for the sport of Goalball (a Paralympic team sport where players throw a ball toward the opponent's goal) using three stacked AI components:

1. **YOLOv8 object detector** — detects the ball, throwing player, and defending players frame-by-frame.
2. **Bidirectional LSTM classifier** — predicts the outcome of each throw: *goal*, *block*, or *out*.
3. **YAMNet audio scorer** (optional) — scores crowd reaction noise per throw using audio extracted from the video.

A **Flask web dashboard** visualises the aggregated results across games.

### Research Context
- Dataset: 6 Paris 2024 Paralympic Goalball games.
- YOLO evaluation: 7-fold leave-one-game-out cross-validation, mean mAP@0.5 = **0.974**.
- LSTM evaluation: 6-fold leave-one-game-out cross-validation, mean accuracy = **0.800** (32f YAMNet model; 23f baseline = 0.776).
- Final deployment model: trained on all 6 games + goal-clip augmentation.

---

## Directory Structure

```
GoalBall/
│
├── predict_pipeline.py                  ← MAIN INFERENCE (no audio)
├── predict_pipeline_with_YAMNet.py      ← MAIN INFERENCE + crowd-noise scoring
├── requirements.txt                     ← Python dependencies (root)
│
├── Model Weights/                       ← Deployment model files
│   ├── best.pt                          ← YOLOv8 weights (~50 MB)
│   ├── final_model.pt                   ← LSTM 23f baseline weights (~2.4 MB)
│   ├── scaler.pkl                       ← StandardScaler for 23f model (~1.4 KB)
│   ├── final_model_yamnet.pt            ← LSTM 32f YAMNet weights (~2.5 MB)
│   ├── scaler_yamnet.pkl                ← StandardScaler for 32f model (~1.7 KB)
│   └── yamnet_config.json              ← YAMNet crowd class config (n_crowd=8, crowd_idx list)
│
├── app/                                 ← Flask web dashboard
│   ├── app.py                           ← Flask routes
│   ├── config.py                        ← File-path config for the app
│   ├── requirements.txt                 ← App-only dependencies
│   ├── static/
│   │   ├── css/styles.css
│   │   └── images/{goal.jpg, throw.jpg}
│   ├── templates/
│   │   ├── base.html                    ← Bootstrap 5 base layout + nav
│   │   ├── index.html                   ← Welcome page
│   │   ├── goals.html                   ← Goal predictions page
│   │   └── throws.html                  ← Per-game throw explorer
│   └── data/
│       ├── GoalPredictions_AllGames.csv ← Aggregated goal records (all games)
│       └── Paralkympics2024/
│           └── {GAME}/outputs/
│               └── {GAME}_Throws_data.xlsx
│
├── Train Model/
│   ├── CNN YOLOv8 Finetune/             ← YOLO training & annotation scripts
│   │   ├── 1_mov_to_frames.py           ← Step 1: extract 1-FPS frames from video
│   │   ├── 2_class2to32.py              ← Step 2: remap CVAT class IDs, populate dataset
│   │   ├── yolo_final.py                ← Step 3a: train YOLO on all data → best.pt
│   │   ├── yolo_loocv.py                ← Step 3b: 7-fold LOOCV evaluation of YOLO
│   │   ├── yolo_cnn_predict_2.py        ← Step 4: annotate a game video (recorded)
│   │   ├── yolo_cnn_LIVE.py             ← Step 4 (live): annotate from camera feed
│   │   ├── full_data/
│   │   │   ├── images/train/            ← All annotated JPEG frames
│   │   │   ├── labels/train/            ← YOLO .txt labels (remapped class IDs)
│   │   │   └── full_data.yaml           ← YOLO dataset config template
│   │   └── yolo_runs/                   ← Training run outputs (auto-created)
│   │
│   └── LSTM Training/                   ← Two independent LSTM variants
│       ├── data_preperation with attention.ipynb
│       ├── data_preperation with attention only goals.ipynb
│       │
│       ├── Baseline LSTM/               ← 23-feature model (visual only, no audio)
│       │   ├── k-fold.py                ← Step 5b option A: LOOCV → mean acc 0.776
│       │   └── train_final.py           ← Step 5a option A: train → final_model.pt
│       │
│       └── LSTM+YAMNet/                 ← 32-feature model (visual + YAMNet crowd audio)
│           ├── extract_audio_features.py ← Prerequisite: YAMNet features → *_yamnet.csv
│           ├── k-fold.py                ← Step 5b option B: LOOCV → mean acc 0.800
│           └── train_final.py           ← Step 5a option B: train → final_model_yamnet.pt
│
└── ball+players_tuning10/               ← Pre-trained YOLO checkpoint (base for fine-tuning)
    ├── weights/best.pt
    └── args.yaml
```

---

## Environment & Dependencies

### Python Version
Python 3.9+ required.

### Install All Dependencies
```bash
pip install -r requirements.txt
```

### Root `requirements.txt`
```
ultralytics        # YOLOv8 training and inference
opencv-python      # Video/image I/O and calibration UI
pandas             # CSV/Excel handling
numpy              # Numeric ops, masking, feature computation
openpyxl           # Excel file creation with embedded hyperlinks
torch              # PyTorch (LSTM training & inference)
scikit-learn       # StandardScaler for feature normalization
torchmetrics       # Precision/Recall/F1 metrics (k-fold.py only)
matplotlib         # Confusion matrix plots (k-fold.py only)
seaborn            # Confusion matrix heatmaps (k-fold.py only)
Flask==3.0.3       # Web dashboard
plotly==5.24.1     # Interactive charts in dashboard
tensorflow         # YAMNet audio scoring (optional, graceful fallback)
tensorflow-hub     # YAMNet model download (optional)
sounddevice        # Microphone recording in live-camera mode (optional)
```

### App-Only `app/requirements.txt`
```
Flask==3.0.3
pandas==2.2.2
plotly==5.24.1
openpyxl==3.1.5
```

### System Dependencies
- **ffmpeg**: Required only for `predict_pipeline_with_YAMNet.py` (audio extraction from video). Install once using whichever method matches your environment, then restart your terminal:

  | Environment | Command |
  |-------------|---------|
  | Conda (recommended) | `conda install -c conda-forge ffmpeg` |
  | Windows (winget) | `winget install ffmpeg` |
  | Windows (Chocolatey) | `choco install ffmpeg` |

  Verify the install worked: `ffmpeg -version`. If you skip this step the pipeline still runs but every Crowd Noise Score will be 0.0.
- **NVIDIA GPU + CUDA**: Optional but strongly recommended for YOLO inference speed. Both pipeline scripts auto-detect CUDA and fall back to CPU silently.

---

## Model Weights

All three files live in `Model Weights/` and are committed to the repository.

| File | Size | Purpose | Required for |
|------|------|---------|-------------|
| `best.pt` | ~50 MB | YOLOv8n fine-tuned for ball (class 32) + throwing player (class 0) + defending player (class 1) detection | Both pipeline scripts, `yolo_cnn_predict_2.py`, `yolo_cnn_LIVE.py` |
| `final_model.pt` | ~2.4 MB | Bidirectional LSTM (128 hidden × 2 layers), 23-feature input — used when no crowd audio detected | Both pipeline scripts |
| `scaler.pkl` | ~1.4 KB | Fitted `sklearn.StandardScaler` for 23 LSTM input features | Both pipeline scripts |
| `final_model_yamnet.pt` | ~2.5 MB | Bidirectional LSTM (128 hidden × 2 layers), 32-feature input (23 visual + 8 YAMNet + yam_rel) — used when crowd audio detected | `predict_pipeline_with_YAMNet.py` |
| `scaler_yamnet.pkl` | ~1.7 KB | Fitted `sklearn.StandardScaler` for all 32 LSTM input features | `predict_pipeline_with_YAMNet.py` |
| `yamnet_config.json` | <1 KB | YAMNet crowd class indices (`n_crowd`, `crowd_idx` list) — determines which of YAMNet's 521 classes are used | `predict_pipeline_with_YAMNet.py`, `extract_audio_features.py` |

All six files are committed to the repository. End-users do not need any training data to run inference.

**Routing logic (automatic):** `predict_pipeline_with_YAMNet.py` loads both LSTM models at startup. After audio extraction, it runs YAMNet on the full waveform and computes `_game_max_crowd = max(patch_scores)`. If `_game_max_crowd > 0.1`, the 32f YAMNet model is used; otherwise the 23f baseline. The threshold 0.1 has a clean empirical gap: real tournament games peak > 0.99; silent recordings peak < 0.007.

---

## YOLO Class Definitions

| Class ID | Name | Notes |
|----------|------|-------|
| 0 | `throwing_player` | The player on the attacking side |
| 1 | `defending_player` | Any player on the defending side |
| 2–31 | COCO classes | Inherited from COCO base weights; largely irrelevant at inference |
| 32 | `sports_ball` | The Goalball (remapped from CVAT class 2 by `2_class2to32.py`) |

---

## LSTM Class Definitions

The LSTM outputs 6 classes encoding outcome × validity:

| Class ID | Label | Outcome | Validity | Meaning |
|----------|-------|---------|---------|---------|
| 0 | `o1` | out | valid | Out-of-bounds throw; real detection |
| 1 | `o0` | out | false | LSTM believes this is a detection artifact, not a real throw |
| 2 | `g1` | goal | valid | Goal scored; real detection |
| 3 | `g0` | goal | false | Goal prediction on suspected artifact |
| 4 | `b1` | block | valid | Ball blocked; real detection |
| 5 | `b0` | block | false | Block prediction on suspected artifact |

`FALSE_LABELS = {'o0', 'g0', 'b0'}` — rows with these predicted classes can be dropped automatically by setting `DROP_FALSE_THROWS = True` at the top of either pipeline script.

---

## Zone System

Each Goalball goal frame is divided into **9 horizontal zones** (1 = far left, 9 = far right, from the thrower's perspective). There are two goals: `lower` (bottom of screen / Team A attacking upward) and `upper` (top of screen / Team B attacking downward).

Zone computation:
1. User clicks 8 goal-frame corners during calibration (4 per goal frame).
2. Script defines two quadrilaterals (`lower_poly`, `upper_poly`).
3. For each throw, `which_goal(pt)` determines which goal the thrower is targeting (polygon point-in-test, fallback to Y-coordinate).
4. `compute_zone(P, A, B)` projects the thrower's position onto the goal's horizontal line and maps to zones 1–9.

---

## LSTM Input Features

There are two feature sets. The pipeline automatically selects the correct one.

### 23-Feature Baseline (used when no crowd audio)

Used by `predict_pipeline.py` (always) and `predict_pipeline_with_YAMNet.py` (when `_game_max_crowd ≤ 0.1`).

```python
FEATURE_COLS = [
    "segment_type",      # 1 = 'to' segment, 0 = 'from' segment
    "rel_t",             # relative frame time within throw (0.0 – 1.0)
    "gap",               # frames since last ball detection
    "defender_seen",     # 1 if defender bbox exists in this frame
    "thrower_seen",      # 1 if thrower bbox exists in this frame
    "ball_seen",         # 1 if ball bbox exists in this frame
    "ball_x",            # ball center X, normalized to [0, 1] by frame width
    "ball_y",            # ball center Y, normalized to [0, 1] by frame height
    "ball_dx",           # ball velocity X (frame-to-frame difference)
    "ball_dy",           # ball velocity Y
    "ball_w",            # ball bbox width (normalized)
    "ball_h",            # ball bbox height (normalized)
    "ball_conf",         # YOLO detection confidence for ball
    "thrower_x",         # thrower center X (normalized)
    "thrower_y",         # thrower center Y (normalized)
    "thrower_w",         # thrower bbox width (normalized)
    "thrower_h",         # thrower bbox height (normalized)
    "thrower_conf",      # YOLO confidence for thrower
    "defender_x",        # defender center X (normalized)
    "defender_y",        # defender center Y (normalized)
    "defender_w",        # defender bbox width (normalized)
    "defender_h",        # defender bbox height (normalized)
    "defender_conf",     # YOLO confidence for defender
]
```

### 32-Feature YAMNet Model (used when crowd audio detected)

Used by `predict_pipeline_with_YAMNet.py` when `_game_max_crowd > 0.1`. The 9 audio features are appended after the 23 visual features:

```python
YAMNET_COLS = [
    "yam_0",   # YAMNet crowd class 0 probability
    "yam_1",   # YAMNet crowd class 1 probability
    ...
    "yam_7",   # YAMNet crowd class 7 probability  (8 crowd-class probs total)
    "yam_rel", # sum(yam_0..7) per frame / max(sum) across all frames in game
               # volume-invariant relative crowd score; peak = 1.0 at loudest moment
]
FEATURE_COLS = VISUAL_COLS + YAMNET_COLS  # 23 + 8 + 1 = 32 total
```

**`yam_rel` motivation:** YAMNet uses a log mel spectrogram so its output shifts with recording volume — a quiet real crowd scores lower in absolute terms than a loud one. `yam_rel` normalises this: the loudest crowd moment in any game always scores 1.0, making the feature comparable across recordings with different volume levels.

Missing detections are **forward-filled** then **backward-filled**; remaining NaN after filling → 0.

---

## Running: Inference Pipeline

### `predict_pipeline.py` — Full inference, no audio scoring

**When to use:** You have a recorded game video (`.mp4`, `.MOV`, `.avi`, etc.) and want automatic LSTM-predicted outcomes with no crowd-noise scoring.

```bash
python predict_pipeline.py
```

**Interactive prompts (in order):**

| Prompt | Example input | Notes |
|--------|--------------|-------|
| Game name | `TUR_-_BRA_3-1` | Used as the output folder name under `Pipeline_Outputs/`. No spaces. |
| Video path | `C:\Users\USER\Videos\game.mp4` | Absolute path. Or type `LIVE` for camera mode. Or type a camera index `0`, `1`, `2`... |
| Lower-side team | `ISR` | Abbreviated team name; the team attacking toward the top of the screen in the first half. |
| Upper-side team | `CAN` | The team attacking toward the bottom of the screen in the first half. |
| Halftime timestamp | `47:12` | `mm:ss` or `hh:mm:ss`. After this timestamp, teams switch sides. |

**8-click goal-zone calibration (after the above prompts):**

A calibration frame from the video is displayed. Click exactly 8 points in this strict order:

```
Click 1: Lower goal — right bottom corner of goal frame
Click 2: Lower goal — right top corner of goal frame
Click 3: Lower goal — left bottom corner of goal frame
Click 4: Lower goal — left top corner of goal frame
Click 5: Upper goal — left bottom corner of goal frame
Click 6: Upper goal — left top corner of goal frame
Click 7: Upper goal — right bottom corner of goal frame
Click 8: Upper goal — right top corner of goal frame
```

Press any key after clicking to proceed.

**Per-segment interaction (after detection pass):**

For each detected throw segment, the script replays the frames with YOLO overlays:
- Green rectangle = throwing player
- Red rectangle = defending player
- Cyan circle = ball

User presses:
- `1` → this is a real throw (keep)
- `0` → false positive (discard)

**Live-camera mode** (if video input is `LIVE` or a camera index):
- Press `T` during live detection to mark halftime.
- Press `Q` to stop recording and proceed to analysis.
- Video is saved to `Pipeline_Outputs/{GAME}/{GAME}_live_recording.mp4`.

**Key configuration flags** (edit at top of script):

```python
SHOW_THROWS = True              # replay each throw with bboxes after analysis
DROP_FALSE_THROWS = False       # drop o0/g0/b0 predictions from Excel output
MERGE_DUPLICATE_THROWS = True   # merge phantom-split duplicate throw detections
MERGE_PHANTOM_GAP_SECONDS = 3   # merge threshold in seconds
```

**Outputs** (written to `Pipeline_Outputs/{GAME}/`):

| File | Description |
|------|-------------|
| `{GAME}_Throws_data_predicted.xlsx` | Per-throw summary: zones, teams, LSTM prediction, confidence, ball detection %, timestamps |
| `{GAME}_Throws_lstm_inference.csv` | Per-frame feature CSV (same format as training CSVs; useful for LSTM retraining) |
| `links/` | `.bat` files that open VLC at the throw's timestamp (Windows only) |

---

### `predict_pipeline_with_YAMNet.py` — Full inference + crowd-noise scoring

**When to use:** Same as above, but you also want a `Crowd Noise Score` column (0.0–1.0) per throw, reflecting audience reaction intensity.

```bash
python predict_pipeline_with_YAMNet.py
```

**Identical to `predict_pipeline.py` in all prompts and flags.** The only difference is the audio processing step that runs after LSTM inference.

**Additional requirements:**
- `ffmpeg` installed and on PATH (or in the active conda environment).
- `tensorflow` and `tensorflow-hub` installed (see `requirements.txt`).
- On first run, YAMNet downloads ~13 MB from TensorFlow Hub.

**Audio processing:**
1. `ffmpeg` extracts mono 16 kHz WAV from the video file.
2. YAMNet runs once on the full waveform → `_yamnet_patch_scores` (n_patches × 8) cached in memory.
3. **Model routing:** `_game_max_crowd = max(_yamnet_patch_scores)`. If > 0.1 → use 32f YAMNet model (`final_model_yamnet.pt` + `scaler_yamnet.pkl`); else → use 23f baseline. Threshold 0.1 has a clean empirical gap (real tournament games peak > 0.99; silent recordings peak < 0.007).
4. **LSTM features (32f path):** `yamnet_per_frame_features(frame_numbers)` maps frame numbers to patch indices via `patch_idx = round(frame / fps / 0.48)`, returns (n_frames, 9) array of `[yam_0..7, yam_rel]` where `yam_rel = yam_sum / game_max_sum`.
5. **Crowd Noise Score column:** for each throw, time window `[TO_start, TO_end + 3s]` is scored using the cached patch scores — no second YAMNet pass needed. Score = mean probability of crowd-class probs across the window.
6. Crowd keywords matched: `["cheer", "applause", "crowd", "yell", "shout", "whoop", "scream"]`.

**Graceful fallback:** If `ffmpeg` is missing, TensorFlow is not installed, or audio extraction fails, all crowd-noise scores are set to `0.0` and the pipeline continues without error.

**Additional Excel column:** `Crowd Noise Score` (float 0.0–1.0) appended to the throw summary.

---

## Running: Flask Web Dashboard

```bash
cd app
pip install -r requirements.txt
python app.py
# Opens at http://127.0.0.1:5000
```

**Data that must be present** (already in repo for Paris 2024):
- `app/data/GoalPredictions_AllGames.csv`
- `app/data/Paralkympics2024/{GAME}/outputs/{GAME}_Throws_data.xlsx` (6 games)

**Routes:**

| URL | Page | Filters (query params) | Content |
|-----|------|----------------------|---------|
| `/` | Welcome / Index | — | Intro, navigation |
| `/goals` | Goal Predictions | `games`, `tteams`, `dteams` | KPI cards, scatter plot (release time), zone heatmap |
| `/throws` | Throw Explorer | `games`, `teams` | Throw count KPIs, from-zone heatmap, to-zone heatmap |

**All filters are multi-select** (hold Ctrl/Cmd to select multiple options).

**Team abbreviation map used by the app:**
```python
{"ISR": "Israel", "CAN": "Canada", "TUR": "Turkey", "CHI": "China", "BRA": "Brazil"}
```

**To point the app at different data**, edit `app/config.py`:
```python
GOALS_CSV = str(_HERE / "data" / "GoalPredictions_AllGames.csv")
THROWS_ROOT = str(_HERE / "data" / "Paralkympics2024")
THROWS_PATTERN = r"**/*_Throws_data.xlsx"
```

---

## Training: YOLO Object Detector

### Prerequisites
- Annotated game frames in CVAT format (YOLO export), with class 2 = ball, class 0 = thrower, class 1 = defender.
- Pre-trained base checkpoint at `ball+players_tuning10/weights/best.pt` (already committed).

### Step 1 — Extract Frames from Video

**Script:** `Train Model/CNN YOLOv8 Finetune/1_mov_to_frames.py`

**Configure at top of file:**
```python
game         = "Gold_TUR-ISR_W_8-4_Euro_2023"   # prefix for output filenames
video_path   = r"C:\path\to\your\video.mpg"
output_folder = r"C:\path\to\frames_output"
```

**Run:**
```bash
python "Train Model/CNN YOLOv8 Finetune/1_mov_to_frames.py"
```

**Output:** One JPEG per second of video, named `{game}_F_image{N}.jpg`, saved to `output_folder`.
Then upload these frames to CVAT for manual annotation.

---

### Step 2 — Remap Class IDs and Populate Dataset

**Script:** `Train Model/CNN YOLOv8 Finetune/2_class2to32.py`

Upload the frames to [CVAT](https://www.cvat.ai/) and annotate with three classes: class 0 = `throwing_player`, class 1 = `defending_player`, class 32 = `sports_ball` (CVAT exports the ball as class 2 — this script remaps it). Export from CVAT as **YOLO format** (images + `.txt` labels), then run this script to remap class 2 → class 32 and copy images + labels into the training dataset.

**Configure at top of file:**
```python
LABELS_FOLDER = r"C:\path\to\cvat_export\labels"
IMAGES_FOLDER = r"C:\path\to\cvat_export\images"
```

**Run:**
```bash
python "Train Model/CNN YOLOv8 Finetune/2_class2to32.py"
```

**Output:**
- Images → `Train Model/CNN YOLOv8 Finetune/full_data/images/train/`
- Labels → `Train Model/CNN YOLOv8 Finetune/full_data/labels/train/`

---

### Step 3a — Train YOLO (All Data → Deployment Model)

**Script:** `Train Model/CNN YOLOv8 Finetune/yolo_final.py`

**Configure at top of file:**
```python
TRAIN_EPOCHS = 60
IMG_SIZE     = 960
BATCH        = 8
LR0          = 0.0001   # gentle LR for incremental fine-tuning
FREEZE       = 10       # freeze first 10 backbone layers
PATIENCE     = 20       # early-stopping patience (epochs without improvement)
DEVICE       = 0        # GPU index; set to 'cpu' for CPU-only training
```

**Run:**
```bash
python "Train Model/CNN YOLOv8 Finetune/yolo_final.py"
```

**What it does:**
1. Globs all images from `full_data/images/train/`.
2. Writes a `train.txt` with absolute paths.
3. Creates a `data.yaml` with 33 classes.
4. Loads `ball+players_tuning10/weights/best.pt` as the starting checkpoint.
5. Trains with Ultralytics API.
6. Copies best checkpoint to `Model Weights/best.pt`.

**Output:** `Model Weights/best.pt` (deployment model), `yolo_runs/yolo_final/results.csv` (training curve).

---

### Step 3b — Evaluate YOLO (7-fold LOOCV)

**Script:** `Train Model/CNN YOLOv8 Finetune/yolo_loocv.py`

**Configure at top of file** (same hyperparameters as `yolo_final.py`):
```python
TRAIN_EPOCHS = 60
IMG_SIZE     = 960
BATCH        = 8
LR0          = 0.0001
FREEZE       = 10
```

**Run:**
```bash
python "Train Model/CNN YOLOv8 Finetune/yolo_loocv.py"
```

**Strategy:** Images are grouped by game prefix (the part before `_F_image`). For each of 7 games: that game's images form the test fold; all other games + 80% of ungrouped images form the training fold.

**Output:** `yolo_runs/loocv_results.csv` — per-fold metrics (mAP@0.5, mAP@0.5:0.95, precision, recall). **Reported result: mean mAP@0.5 = 0.974**.

---

### Step 4 — Annotate a Recorded Game Video (for LSTM Training Data)

**Script:** `Train Model/CNN YOLOv8 Finetune/yolo_cnn_predict_2.py`

This script runs YOLO on a recorded video and has the user **manually label each throw's outcome** (`g`/`b`/`o`), generating CSVs for LSTM training.

**Configure at top of file:**
```python
game            = "TUR_-_BRA_3-1"
video_path      = f'Thesis/Paralympics2024/{game}/{game}.MOV'
excel_output_path = f'Thesis/Paralympics2024/{game}/outputs/{game}_Throws_data.xlsx'
lstm_csv_path   = f'Thesis/Paralympics2024/{game}/outputs/{game}_Throws_lstm_training.csv'
```

**Run:**
```bash
python "Train Model/CNN YOLOv8 Finetune/yolo_cnn_predict_2.py"
```

**Interactive flow:**
1. Calibration frame displayed → press any key.
2. Enter: lower team name, upper team name, halftime timestamp.
3. 8-click goal-zone calibration (same order as pipeline scripts above).
4. For each `'from'` segment: press `1` (real) or `0` (false positive).
5. For each `'to'` segment: press `1` or `0`, then if `1`: press `g` (goal), `b` (block), or `o` (out).

**Key difference from pipeline:** Outcome labels come from the user, not the LSTM. This produces ground-truth training data.

**Output:**
- `{game}_Throws_data.xlsx` — per-throw summary with manually assigned outcomes.
- `{game}_Throws_lstm_training.csv` — per-frame feature CSV with user labels.

---

### Step 4 (Live) — Annotate from Live Camera Feed

**Script:** `Train Model/CNN YOLOv8 Finetune/yolo_cnn_LIVE.py`

Same as `yolo_cnn_predict_2.py` but reads from a camera instead of a file.

**Run:**
```bash
python "Train Model/CNN YOLOv8 Finetune/yolo_cnn_LIVE.py"
```

**Additional interactions:**
- Press `T` during live detection to mark halftime.
- Press `Q` to stop recording and proceed to interactive segment labeling.

Camera feed is saved to `Pipeline_Outputs/{GAME}/{GAME}_live_recording.mp4` in real time. If `sounddevice` is installed, microphone audio is recorded and saved as WAV alongside the video.

---

## Training: LSTM Throw-Outcome Classifier

There are **two independent model variants** under `Train Model/LSTM Training/`. Each has its own subfolder with its own `k-fold.py` and `train_final.py`. `predict_pipeline_with_YAMNet.py` loads both at startup and routes automatically.

---

### Baseline LSTM — `Train Model/LSTM Training/Baseline LSTM/`

23 visual features, no audio. Used by the pipeline when crowd audio is absent or silent.

**Step 5a (Baseline) — Train deployment model:**

**Script:** `Train Model/LSTM Training/Baseline LSTM/train_final.py`

**Configure at top of file:**
```python
DATA_ROOT = Path(r"C:\path\to\Paralkympics2024")
GOALS_DIR = Path(r"C:\path\to\Goals_Paralympics\outputs")
# SAVE_PATH is auto-resolved to GoalBall/Model Weights/final_model.pt
```

```bash
python "Train Model/LSTM Training/Baseline LSTM/train_final.py"
```

**Output:** `Model Weights/final_model.pt` (save path auto-resolved via `SCRIPT_DIR.parent.parent.parent`).

**Step 5b (Baseline) — Evaluate via LOOCV:**

```bash
python "Train Model/LSTM Training/Baseline LSTM/k-fold.py"
```

**Reported result: mean accuracy = 0.776**

---

### LSTM+YAMNet — `Train Model/LSTM Training/LSTM+YAMNet/`

32 features: 23 visual + 8 YAMNet crowd probs + 1 `yam_rel`. Used by the pipeline when crowd audio is detected.

### Step 5 Prerequisite — Extract YAMNet Audio Features (run once)

**Script:** `Train Model/LSTM Training/LSTM+YAMNet/extract_audio_features.py`

Runs YAMNet on every game video and goal clip; writes one `*_yamnet.csv` per source CSV in the same `LSTM+YAMNet/` folder. `k-fold.py` and `train_final.py` load these CSVs automatically; if a file is missing they fall back to zeros with a warning.

**Requirements:** `tensorflow`, `tensorflow-hub`, `ffmpeg` on PATH. Runtime: ~5–10 minutes.

```bash
python "Train Model/LSTM Training/LSTM+YAMNet/extract_audio_features.py"
```

**Configure at top of file:**
```python
DATA_ROOT  = Path(r"C:\path\to\Paralkympics2024")
GOALS_ROOT = Path(r"C:\path\to\Goals_Paralympics")
```

---

### Step 5a (YAMNet) — Train LSTM+YAMNet Deployment Model

**Script:** `Train Model/LSTM Training/LSTM+YAMNet/train_final.py`

**Configure at top of file:**
```python
DATA_ROOT = Path(r"C:\path\to\Paralkympics2024")
GOALS_DIR = Path(r"C:\path\to\Goals_Paralympics\outputs")
# SAVE_MODEL and SAVE_SCALER are auto-resolved to GoalBall/Model Weights/
```

GAMES = [
    "ISR_-_CAN_5-1", "TUR_-_BRA_3-1", "TUR_-_ISR_5-4",
    "ISR_-_BRA_8-4", "CHI_-_TUR_7-5", "BRA_-_TUR_3-3"
]

N_EPOCHS = 100
PATIENCE = 10       # early stopping (validation accuracy)
BATCH    = 32
LR       = 1e-3
WD       = 1e-4
GAMMA    = 2        # focal loss gamma
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
```

**Run:**
```bash
python "Train Model/LSTM Training/LSTM+YAMNet/train_final.py"
```

**Data loading:**
1. Reads `{GAME}/outputs/{GAME}_Throws_lstm_training.csv` for each of the 6 games.
2. Reads all `*.csv` from `GOALS_DIR` (goal-clip augmentation data).
3. Normalizes coordinates by video resolution (reads video with `cv2` to get dimensions, falls back to 1920×1080).
4. Augments `g1` class via horizontal flip (`mirror_g1()`) to mitigate class imbalance.

**Class mapping:**
```python
CLS = {
    ('o', 1): 0,   # out, valid
    ('o', 0): 1,   # out, false positive
    ('g', 1): 2,   # goal, valid
    ('g', 0): 3,   # goal, false positive
    ('b', 1): 4,   # block, valid
    ('b', 0): 5,   # block, false positive
}
```

**Model architecture:**
```python
class ThrowLSTM(nn.Module):
    n = 32          # input features (23 visual + 8 YAMNet crowd probs + 1 yam_rel)
    h = 128         # hidden size per LSTM direction
    l = 2           # LSTM layers
    c = 6           # output classes

    self.lstm = nn.LSTM(n, h, l, batch_first=True, bidirectional=True, dropout=0.3)
    self.attn = nn.Sequential(Linear(h*2, 64), Tanh(), Dropout(0.2), Linear(64, 1))
    self.head = nn.Sequential(Linear(h*2, 128), ReLU(), Dropout(0.4), Linear(128, c))
```

**Training details:**
- **Loss:** Focal loss — `(1 − p_t)^γ × CE` — with per-class weights (squared inverse frequency + ×2 boost for goal class).
- **Sampler:** `WeightedRandomSampler` with 3× oversampling for minority classes.
- **LR scheduler:** `ReduceLROnPlateau` (factor=0.3, patience=3).
- **Early stopping:** Patience=10 on validation accuracy.

**Output:** `Model Weights/final_model_yamnet.pt` + `Model Weights/scaler_yamnet.pkl` (best checkpoint, auto-resolved to `GoalBall/Model Weights/`).

Console prints per epoch: accuracy, macro-F1, goal-class precision, goal-class recall.

---

### Step 5b (YAMNet) — Evaluate via LOOCV

**Script:** `Train Model/LSTM Training/LSTM+YAMNet/k-fold.py`

**Configure at top of file** (same paths as `train_final.py`):
```python
DATA_ROOT = Path(r"C:\path\to\Paralkympics2024")
GOALS_DIR = Path(r"C:\path\to\Goals_Paralympics\outputs")
```

**Run:**
```bash
python "Train Model/LSTM Training/LSTM+YAMNet/k-fold.py"
```

**Strategy:** For each of 6 games: that game is the test fold; all other 5 games + goal clips form the training fold. A fresh LSTM is trained per fold (30 epochs, patience=10).

**Output:**
- `best_fold_{fold_idx}.pt` — per-fold checkpoint.
- Console: per-fold accuracy, macro-F1, weighted-F1.
- `plot_confusion_matrix_heatmap.png` — aggregated confusion matrix across all folds.
- `plot_metrics_per_class.png` — bar chart of per-class metrics.

**Reported results (32f YAMNet model):**

| Model | Mean Acc | g1 Precision | g1 Recall | g1 F1 |
|-------|----------|-------------|-----------|-------|
| Baseline 23f | 0.776 | 0.407 | 0.422 | 0.414 |
| 31f (+yam_0..7) | 0.782 | 0.625 | 0.612 | 0.619 |
| **32f (+yam_rel)** | **0.800** | 0.574 | **0.633** | 0.602 |

---

## Data File Schemas

### `{GAME}_Throws_lstm_training.csv` (generated by annotation scripts)

This CSV is the **ground-truth training data** for the LSTM. Each row is one frame within a detected throw segment.

| Column | Type | Description |
|--------|------|-------------|
| `segment_id` | int | 1-based segment index |
| `segment_type` | str | `'from'` or `'to'` |
| `frame` | int | Frame number (1-based) within the video |
| `label` | int | 1 = user confirmed real; 0 = user marked false positive |
| `outcome` | str | `'g'`, `'b'`, `'o'` (manual), or blank for `'from'` segments |
| `ball_x` | float | YOLO ball center X (raw pixels); NaN if undetected |
| `ball_y` | float | YOLO ball center Y |
| `ball_w` | float | Bbox width (pixels) |
| `ball_h` | float | Bbox height (pixels) |
| `ball_conf` | float | YOLO detection confidence 0–1 |
| `thrower_x/y/w/h/conf` | float | Same for throwing player |
| `defender_x/y/w/h/conf` | float | Same for defending player |

### `{GAME}_Throws_data.xlsx` / `{GAME}_Throws_data_predicted.xlsx` (per-throw summary)

One row per throw, written to Excel by any pipeline script.

| Column | Description |
|--------|-------------|
| Throw Number | Sequential index |
| Start Time | Video timestamp (`hh:mm:ss`) |
| From Zone | `"Lower N"` or `"Upper N"` (zone 1–9) |
| To Zone | Same format |
| Throwing Team | Team abbreviation |
| Defending Team | Team abbreviation |
| Predicted Outcome | `goal` / `block` / `out` (LSTM) — or user-labeled in annotation scripts |
| Predicted Class | Full class label: `g1`, `g0`, `b1`, `b0`, `o1`, `o0` |
| Prediction Confidence | Float 0–1 |
| Throw Length (frames) | Number of frames in the TO segment |
| Ball Detection Rate | Fraction of frames where ball was detected |
| Crowd Noise Score | Float 0–1 (YAMNet version only; absent or 0 otherwise) |
| From Coord | Raw pixel coordinate of throw origin |
| To Coord | Raw pixel coordinate of throw destination |

### `app/data/GoalPredictions_AllGames.csv`

Aggregated goal records across all games, used by the `/goals` Flask route.

| Column | Description |
|--------|-------------|
| Game | Game identifier string |
| Throw Number | Sequential index |
| From Zone | Zone string |
| To Zone | Zone string |
| Throwing Team | Abbreviation |
| Defending Team | Abbreviation |
| Release Time | Timestamp `mm:ss` or `hh:mm:ss` |

---

## Full Pipeline: End-to-End Workflow

### For a New Game (Training Data Collection)

```
1. Record game video
2. python 1_mov_to_frames.py         → extracts 1 FPS JPEGs
3. Annotate in CVAT (manual)
4. Export from CVAT as YOLO format
5. python 2_class2to32.py            → remaps class IDs, populates full_data/
6. python yolo_final.py              → trains/updates YOLO → best.pt
7. python yolo_cnn_predict_2.py      → runs YOLO, user labels outcomes
                                        → {GAME}_Throws_lstm_training.csv

8a. (Baseline model)
    python "Baseline LSTM/train_final.py"   → Model Weights/final_model.pt

8b. (YAMNet model — requires audio)
    python "LSTM+YAMNet/extract_audio_features.py"  → *_yamnet.csv (run once)
    python "LSTM+YAMNet/train_final.py"             → Model Weights/final_model_yamnet.pt

    predict_pipeline_with_YAMNet.py auto-selects between the two at runtime.
    (Both models can coexist in Model Weights/ — train both for full coverage.)
```

### For Analyzing a New Game (Inference, No New Training)

```
1. python predict_pipeline_with_YAMNet.py
   (or predict_pipeline.py if no audio needed)
   → interactive prompts: game name, video path, teams, halftime, calibration
   → per-segment validity check (1/0 keypresses)
   → Pipeline_Outputs/{GAME}/{GAME}_Throws_data_predicted.xlsx
```

### For Viewing Results in the Dashboard

```
1. Copy {GAME}_Throws_data_predicted.xlsx to app/data/Paralkympics2024/{GAME}/outputs/
2. Append goal rows to app/data/GoalPredictions_AllGames.csv
3. cd app && python app.py
4. Open http://127.0.0.1:5000
```

---

## Key Algorithms

### Segment Detection and Pairing
1. YOLO runs on every frame; the "active class" for each frame is whichever player class (0 or 1) has the highest detection confidence when the ball (class 32) is also detected.
2. Frames are grouped into consecutive runs of the same active class → each run is a segment (`'from'` = thrower active; `'to'` = defender active).
3. Consecutive same-type segments are collapsed into one.
4. `'to'` segments are extended by **80 frames** beyond the detected class-change to capture the outcome moment (ball crossing goal / blocked).
5. Segments are paired sequentially: `from_1 → to_1`, `from_2 → to_2`, etc.

### Phantom-Split Deduplication
When YOLO briefly misclassifies a mid-flight ball frame as the opposite team, it creates two `from` segments for the same physical throw. Detection criterion: two `from` segments whose start frames are within `MERGE_PHANTOM_GAP_SECONDS × FPS` of each other **and** have the same throwing team (polygon test). Resolution: keep the **later** throw (better TO coverage), drop the earlier one.

### Halftime Side Swap
Before `halftime_frame`, the `lower_poly` team is the attacking side when the ball is in the lower goal area. After `halftime_frame`, teams swap. `which_goal(pt)` does a point-in-polygon test; the team assignment is then looked up by half and goal.

---

## Common Issues and Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `scaler.pkl not found, fitting new scaler...` then crash | Training CSV paths don't exist | Either ship `scaler.pkl` with the repo (already committed) or set `GOALBALL_TRAIN_ROOT` / `GOALBALL_GOALS_DIR` env vars to point at the training data |
| `ffmpeg not found` in YAMNet pipeline | ffmpeg not installed | `conda install ffmpeg` or install via OS package manager; all scores default to 0.0 and pipeline continues |
| YOLO runs very slowly | No GPU detected | Normal on CPU; set `DEVICE=0` and ensure CUDA PyTorch is installed |
| `CUDA out of memory` during YOLO training | Batch too large for GPU | Reduce `BATCH = 8` to `4` or `2` in training scripts |
| Calibration click order wrong | User clicks in wrong order | Must follow the strict 8-click order documented above; restart the script to redo |
| Excel file locked | Excel has the file open | Close Excel, then re-run the pipeline |
| Throws not merging as expected | `MERGE_PHANTOM_GAP_SECONDS` too low | Increase from 3 to 5–10 seconds in the CONFIG block at top of pipeline scripts |

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10–3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | — (CPU OK) | 8 GB (YOLO training), 4 GB (inference) |
| Disk | 200 MB (models + code) | 50 GB (with all game videos) |
| OS | Windows 10+ / Linux / macOS | Windows 11 (tested) |
| ffmpeg | Not needed for basic inference | Required for YAMNet version |

The inference pipeline and Flask app run fully on CPU. GPU is only needed for faster YOLO inference and for YOLO/LSTM training.
