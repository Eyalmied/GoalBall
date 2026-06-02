"""
extract_audio_features.py  — Step 0 of LSTM+YAMNet training (run once).

For each training CSV + its corresponding video:
  1. Extracts the full audio track to a 16 kHz mono WAV (via ffmpeg).
  2. Runs YAMNet on the full waveform -> per-patch (521,) probability vectors.
  3. Maps each video frame in the CSV to its nearest YAMNet patch.
  4. Keeps only the N_CROWD crowd-related class probabilities per frame.
  5. Adds yam_rel = sum(yam_0..N) / max(sum) across the full clip (volume-invariant).
  6. Saves an augmented CSV in this directory with columns yam_0..yam_{N-1} + yam_rel.

Outputs go to the same folder as this script (LSTM+YAMNet/):
  - yamnet_config.json    <- crowd class indices, read by k-fold.py and train_final.py
  - {GAME}_yamnet.csv     <- augmented per-frame CSVs

Run this ONCE before k-fold.py / train_final.py.
Runtime: ~5-10 min for 6 games + 17 goal clips.

Requirements: tensorflow, tensorflow-hub, ffmpeg on PATH (or in conda env).
Run with:  C:/Users/USER/anaconda3/envs/thesis_gpu/python.exe extract_audio_features.py
"""

import sys, json, csv, wave, tempfile, subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# ─── PATHS ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_ROOT  = Path(r"C:\Users\USER\Desktop\Thesis\Paralkympics2024")
GOALS_ROOT = Path(r"C:\Users\USER\Desktop\Thesis\Goals_Paralympics")
GOALS_DIR  = GOALS_ROOT / "outputs"

GAMES = [
    "ISR_-_CAN_5-1",
    "TUR_-_BRA_3-1",
    "TUR_-_ISR_5-4",
    "ISR_-_BRA_8-4",
    "CHI_-_TUR_7-5",
    "BRA_-_TUR_3-3",
]

# Video paths for the 6 main games (all .mov in their game subfolder)
GAME_VIDEOS = {
    game: DATA_ROOT / game / f"{game}.mov" for game in GAMES
}

YAMNET_CROWD_KEYWORDS = ["cheer", "applause", "crowd", "yell", "shout", "whoop", "scream"]
YAMNET_HOP_S = 0.48   # YAMNet patch hop in seconds (0.96 s window, 50% overlap)
# ──────────────────────────────────────────────────────────────────────────────


def load_yamnet():
    try:
        import tensorflow_hub as hub
    except ImportError:
        print("[ERROR] tensorflow-hub not installed.")
        print("  Run: pip install tensorflow tensorflow-hub")
        sys.exit(1)

    print("[YAMNet] Loading model from TF Hub (~13 MB download on first run)...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode()

    crowd_idx = []
    with open(class_map_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["display_name"].lower()
            if any(k in name for k in YAMNET_CROWD_KEYWORDS):
                crowd_idx.append(int(row["index"]))

    print(f"[YAMNet] Matched {len(crowd_idx)} crowd class indices: {crowd_idx}")

    cfg = {"n_crowd": len(crowd_idx), "crowd_idx": crowd_idx}
    with open(SCRIPT_DIR / "yamnet_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[YAMNet] Config saved -> yamnet_config.json\n")

    return model, crowd_idx


def find_ffmpeg() -> str:
    conda_ff = Path(sys.executable).parent / "Library" / "bin" / "ffmpeg.exe"
    return str(conda_ff) if conda_ff.exists() else "ffmpeg"


def extract_audio_16k(video_path: Path, wav_path: Path) -> bool:
    cmd = [find_ffmpeg(), "-y", "-i", str(video_path),
           "-ar", "16000", "-ac", "1", "-f", "wav", str(wav_path)]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [ERROR] ffmpeg failed: {e}")
        return False


def load_wav_16k(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def get_video_fps(video_path: Path) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 25.0
    except Exception:
        return 25.0


def process_one(csv_path: Path, video_path: Path,
                yamnet_model, crowd_idx: list) -> pd.DataFrame:
    n_crowd  = len(crowd_idx)
    yam_cols = [f"yam_{i}" for i in range(n_crowd)]

    df  = pd.read_csv(csv_path)
    fps = get_video_fps(video_path)
    print(f"  Video : {video_path}  ({fps:.1f} fps)")
    print(f"  CSV   : {len(df)} rows")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    if not extract_audio_16k(video_path, wav_path):
        print("  [WARN] Audio extraction failed — filling with 0.0")
        for col in yam_cols:
            df[col] = 0.0
        df["yam_rel"] = 0.0
        wav_path.unlink(missing_ok=True)
        return df

    waveform = load_wav_16k(wav_path)
    wav_path.unlink(missing_ok=True)
    print(f"  Audio : {len(waveform)/16000:.1f}s at 16 kHz")

    scores, _, _ = yamnet_model(waveform)
    scores_np    = scores.numpy()           # (n_patches, 521)
    n_patches    = scores_np.shape[0]
    print(f"  Patches: {n_patches}")

    frame_indices = df["frame"].values.astype(float)
    time_s        = frame_indices / fps
    patch_indices = np.clip(
        np.round(time_s / YAMNET_HOP_S).astype(int), 0, n_patches - 1
    )

    # Keep only the N_CROWD crowd-class columns (not all 521)
    frame_yamnet = scores_np[patch_indices][:, crowd_idx]   # (n_frames, n_crowd)
    for i, col in enumerate(yam_cols):
        df[col] = frame_yamnet[:, i].astype(np.float32)

    # Game-relative crowd score: sum of 8 crowd probs per frame / max such sum in this clip.
    # Makes the feature volume-invariant: the loudest crowd moment always = 1.0,
    # so a quiet recording of real crowd looks identical to a loud recording.
    yam_sum      = frame_yamnet.sum(axis=1)          # (n_frames,)
    game_max_sum = float(yam_sum.max())
    df["yam_rel"] = (yam_sum / (game_max_sum + 1e-8)).astype(np.float32)

    print(f"  Max crowd score this clip : {frame_yamnet.max():.4f}")
    print(f"  Max yam_sum / yam_rel=1.0 : {game_max_sum:.4f}")
    return df


def main():
    yamnet_model, crowd_idx = load_yamnet()
    n_crowd = len(crowd_idx)

    # Build full job list: (csv_path, video_path, out_csv_path, label)
    jobs = []

    # 6 main games
    for game in GAMES:
        csv_path   = DATA_ROOT / game / "outputs" / f"{game}_Throws_lstm_training.csv"
        video_path = GAME_VIDEOS[game]
        out_path   = SCRIPT_DIR / f"{game}_yamnet.csv"
        jobs.append((csv_path, video_path, out_path, game))

    # 17 goal clips — video stem matches CSV game_tag exactly
    for csv_path in sorted(GOALS_DIR.glob("*.csv")):
        game_tag   = csv_path.stem.split("_Throws")[0]
        video_path = GOALS_ROOT / f"{game_tag}.mp4"
        out_path   = SCRIPT_DIR / f"{game_tag}_yamnet.csv"
        jobs.append((csv_path, video_path, out_path, game_tag))

    # Print status table
    print(f"\n{'='*65}")
    print(f"JOBS: {len(jobs)} total  |  {n_crowd} crowd-class features per frame")
    print(f"{'='*65}")
    any_missing = False
    for csv_path, video_path, out_path, tag in jobs:
        csv_ok  = "✅" if csv_path.exists()   else "❌"
        vid_ok  = "✅" if video_path.exists() else "❌"
        done    = "⏩" if out_path.exists()   else "  "
        print(f"  {done} {csv_ok}csv {vid_ok}video  {tag}")
        if not video_path.exists():
            any_missing = True

    if any_missing:
        print("\n[!] Some videos not found — those clips will use audio=0.0 (zeros).")
    print()

    processed = skipped = failed = 0
    for csv_path, video_path, out_path, tag in jobs:
        if out_path.exists():
            print(f"[SKIP] {tag}")
            skipped += 1
            continue
        if not csv_path.exists():
            print(f"[SKIP] {tag} — CSV not found")
            failed += 1
            continue

        print(f"\n[{tag}]")
        try:
            df_aug = process_one(csv_path, video_path, yamnet_model, crowd_idx)
            df_aug.to_csv(out_path, index=False)
            print(f"  Saved -> {out_path.name}  ({len(df_aug)} rows, {n_crowd} new cols)")
            processed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print(f"\n{'='*65}")
    print(f"Done.  Processed={processed}  Skipped={skipped}  Failed={failed}")
    print(f"Output dir: {SCRIPT_DIR}")
    print(f"\nNext: python k-fold.py  or  python train_final.py")


if __name__ == "__main__":
    main()
