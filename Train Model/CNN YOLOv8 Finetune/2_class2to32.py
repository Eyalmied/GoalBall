"""
Step 2: Remap ball class IDs (CVAT default class 2 → YOLO class 32 / sports_ball)
        and copy annotated images + labels into the full_data/ pool.

Background: CVAT exports the ball with class ID 2 by default. In YOLOv8's
COCO-pretrained backbone sports_ball is class 32. This script remaps every
label line starting with '2' to '32' so fine-tuning builds on the pretrained
ball detector.

All images land in full_data/images/train/ — yolo_loocv.py groups them by
game prefix and creates its own LOOCV splits on the fly. yolo_final.py uses
the entire pool with no holdout.

After running this script, proceed to:
  yolo_loocv.py   — 7-fold leave-one-game-out cross-validation
  yolo_final.py   — final model trained on ALL data
"""

import os
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent    # Train Model/CNN YOLOv8 Finetune/

# === CONFIGURE THESE ===
LABELS_FOLDER = r"C:\path\to\cvat_export\labels"   # CVAT export labels folder
IMAGES_FOLDER = r"C:\path\to\cvat_export\images"   # CVAT export images folder
# =======================

OUT_IMAGES = _HERE / "full_data" / "images" / "train"
OUT_LABELS = _HERE / "full_data" / "labels" / "train"
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)


def remap_and_copy_label(src_label: Path, dst_label: Path):
    lines = src_label.read_text().splitlines()
    remapped = []
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0] == "2":
            parts[0] = "32"
        remapped.append(" ".join(parts))
    dst_label.write_text("\n".join(remapped) + "\n")


label_files = list(Path(LABELS_FOLDER).glob("*.txt"))
copied = skipped = 0

for lbl in label_files:
    img = Path(IMAGES_FOLDER) / (lbl.stem + ".jpg")
    if not img.exists():
        print(f"  Warning: no image for {lbl.name}")
        skipped += 1
        continue

    remap_and_copy_label(lbl, OUT_LABELS / lbl.name)
    shutil.copy(img, OUT_IMAGES / img.name)
    copied += 1

print(f"Done: {copied} files copied, {skipped} skipped (image not found).")
print(f"Labels remapped (class 2 → 32).")
print(f"Images → {OUT_IMAGES}")
print(f"Labels → {OUT_LABELS}")
