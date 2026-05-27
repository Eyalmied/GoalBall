"""
YOLOv8 final deployment model — train on ALL available data.

Mirrors the LOOCV setup (same hyperparameters, same base model) but uses
every annotated image for training with no holdout.  The resulting weights
are the model to load in predict_pipeline_with_YAMNet.py for inference on new games.

Output
------
  Train Model/CNN YOLOv8 Finetune/yolo_runs/yolo_final/weights/best.pt   ← use this
  Train Model/CNN YOLOv8 Finetune/yolo_runs/yolo_final/results.csv       ← training curve

Run
---
  python "Train Model/CNN YOLOv8 Finetune/yolo_final.py"
"""

import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

_HERE     = Path(__file__).resolve().parent         # Train Model/CNN YOLOv8 Finetune/
_REPO     = _HERE.parent.parent                     # GoalBall/

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT  = _HERE / "full_data"
BASE_MODEL = _REPO / "ball+players_tuning10" / "weights" / "best.pt"
OUT_DIR    = _HERE / "yolo_runs"

NC = 33
NAMES = {
    0: "throwing_player",  1: "defending_player", 2: "car",
    3: "motorcycle",       4: "airplane",          5: "bus",
    6: "train",            7: "truck",             8: "boat",
    9: "traffic light",   10: "fire hydrant",     11: "stop sign",
    12: "parking meter",  13: "bench",            14: "bird",
    15: "cat",            16: "dog",              17: "horse",
    18: "sheep",          19: "cow",              20: "elephant",
    21: "bear",           22: "zebra",            23: "giraffe",
    24: "backpack",       25: "umbrella",         26: "handbag",
    27: "tie",            28: "suitcase",         29: "frisbee",
    30: "skis",           31: "snowboard",        32: "sports_ball",
}

TRAIN_EPOCHS = 60
IMG_SIZE     = 960
BATCH        = 8
LR0          = 0.0001
FREEZE       = 10
PATIENCE     = 20
SEED         = 42
DEVICE       = 0       # set to 'cpu' if no GPU
# ──────────────────────────────────────────────────────────────────────────────


def main():
    assert DATA_ROOT.exists(), f"Data folder not found: {DATA_ROOT}"
    assert BASE_MODEL.exists(), f"Base model not found: {BASE_MODEL}"

    # Collect every image from both train/ and val/ sub-dirs
    all_images = []
    for split in ("train", "val"):
        d = DATA_ROOT / "images" / split
        if d.exists():
            all_images += list(d.glob("*.jpg")) + list(d.glob("*.png"))

    print(f"Total images for final training: {len(all_images)}")

    # Write a single train.txt that lists every image
    run_dir   = OUT_DIR / "yolo_final"
    train_txt = run_dir / "train.txt"
    data_yaml = run_dir / "data.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(train_txt, "w") as f:
        for p in all_images:
            f.write(str(p.resolve()) + "\n")

    # Use the same file for train and val so Ultralytics has something to
    # evaluate against each epoch (we care about training loss, not val mAP here)
    cfg = {
        "train": str(train_txt.resolve()),
        "val":   str(train_txt.resolve()),
        "nc":    NC,
        "names": NAMES,
    }
    with open(data_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    model  = YOLO(str(BASE_MODEL))
    result = model.train(
        data     = str(data_yaml),
        epochs   = TRAIN_EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        augment  = True,
        lr0      = LR0,
        freeze   = FREEZE,
        device   = DEVICE,
        project  = str(OUT_DIR),
        name     = "yolo_final",
        patience = PATIENCE,
        exist_ok = True,
        verbose  = True,
        workers  = 0,   # Windows: avoids CUDA DLL clash in spawned workers
        seed     = SEED,
    )

    best_pt    = OUT_DIR / "yolo_final" / "weights" / "best.pt"
    deploy_pt  = _REPO / "Model Weights" / "best.pt"
    shutil.copy(best_pt, deploy_pt)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Run output      → {best_pt}")
    print(f"Deployment copy → {deploy_pt}")
    rd = result.results_dict
    print(f"  mAP@0.5    : {rd.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:0.95: {rd.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision  : {rd.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall     : {rd.get('metrics/recall(B)', 0):.4f}")
    print("=" * 60)
    print(f"\nDeployment weights copied to: {_REPO / 'Model Weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
