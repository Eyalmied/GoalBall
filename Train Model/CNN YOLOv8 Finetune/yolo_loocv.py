"""
YOLOv8 7-fold LOOCV fine-tuning — local GPU version.

Split strategy
--------------
  Game prefix is extracted by splitting the filename on '_F_image':
    'TUR - BRA 3-1_F_image1088'        → game = 'TUR - BRA 3-1'
    'CAN_-_BRA_4-2_F_image012'         → game = 'CAN_-_BRA_4-2'
    'GOALS_Tokyo2024_ISR5_KOR4_F_image1' → game = 'GOALS_Tokyo2024_ISR5_KOR4'

  Files without '_F_image' (e.g. frame_00012.jpg) are ungrouped.

  Each LOOCV fold:
    val   = ALL images of the left-out game  +  20 % of ungrouped (fixed seed)
    train = ALL other games                  +  80 % of ungrouped (fixed seed)

  7 named games → 7 folds.

Run
---
  python "Train Model/CNN YOLOv8 Finetune/yolo_loocv.py"
"""

import csv
import random
import re
import sys
import yaml
from pathlib import Path

from ultralytics import YOLO

_HERE     = Path(__file__).resolve().parent         # Train Model/CNN YOLOv8 Finetune/
_REPO     = _HERE.parent.parent                     # GoalBall/

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = _HERE / "full_data"
BASE_MODEL  = _REPO / "ball+players_tuning10" / "weights" / "best.pt"
OUT_DIR     = _HERE / "yolo_runs"
RESULTS_CSV = OUT_DIR / "loocv_results.csv"

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
BATCH        = 8        # lower to 4 if you run out of VRAM
LR0          = 0.0001   # gentle LR for incremental fine-tuning
FREEZE       = 10       # freeze first 10 backbone layers
PATIENCE     = 20
SEED         = 42
DEVICE       = 0        # GPU index; set to 'cpu' to test without GPU
# ───────────────────────────────────────────────────────────────────────────────

SPLIT_MARKER = re.compile(r"_F_image", re.IGNORECASE)


def extract_game(stem: str) -> str | None:
    """Return game prefix or None if ungrouped."""
    parts = SPLIT_MARKER.split(stem, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip()
    return None


def pool_images(data_root: Path) -> list[Path]:
    imgs = []
    for split in ("train", "val"):
        d = data_root / "images" / split
        if d.exists():
            imgs += list(d.glob("*.jpg")) + list(d.glob("*.png"))
    return imgs


def group_by_game(images: list[Path]) -> tuple[dict[str, list[Path]], list[Path]]:
    games: dict[str, list[Path]] = {}
    ungrouped: list[Path] = []
    for p in images:
        key = extract_game(p.stem)
        if key:
            games.setdefault(key, []).append(p)
        else:
            ungrouped.append(p)
    return games, ungrouped


def write_txt(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in images:
            f.write(str(p.resolve()) + "\n")


def write_yaml(path: Path, train_txt: Path, val_txt: Path) -> None:
    cfg = {
        "train": str(train_txt.resolve()),
        "val":   str(val_txt.resolve()),
        "nc":    NC,
        "names": NAMES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def train_fold(fold_name: str, train_imgs: list[Path], val_imgs: list[Path]) -> dict:
    fold_dir  = OUT_DIR / fold_name
    train_txt = fold_dir / "train.txt"
    val_txt   = fold_dir / "val.txt"
    data_yaml = fold_dir / "data.yaml"

    write_txt(train_txt, train_imgs)
    write_txt(val_txt,   val_imgs)
    write_yaml(data_yaml, train_txt, val_txt)

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
        name     = fold_name,
        patience = PATIENCE,
        exist_ok = True,
        verbose  = True,
        workers  = 0,   # Windows fix: avoid CUDA DLL clash in spawned worker processes
    )

    rd = result.results_dict
    return {
        "mAP50":     round(rd.get("metrics/mAP50(B)",     0), 4),
        "mAP50-95":  round(rd.get("metrics/mAP50-95(B)",  0), 4),
        "precision": round(rd.get("metrics/precision(B)", 0), 4),
        "recall":    round(rd.get("metrics/recall(B)",    0), 4),
    }


def safe_name(s: str) -> str:
    """Make a game name safe for use as a folder name."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def main() -> None:
    if not DATA_ROOT.exists():
        sys.exit(f"\nData folder not found: {DATA_ROOT}")
    if not BASE_MODEL.exists():
        sys.exit(f"\nBase model not found: {BASE_MODEL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    # ── Pool ─────────────────────────────────────────────────────────────────
    print("Pooling all images …")
    all_images = pool_images(DATA_ROOT)
    print(f"  Total: {len(all_images)}")

    games, ungrouped = group_by_game(all_images)

    print(f"\nNamed-game groups ({len(games)}) — one left out per fold:")
    for g in sorted(games, key=lambda x: -len(games[x])):
        print(f"  [{len(games[g]):3d}]  {g}")

    print(f"\nUngrouped (frame_* etc.): {len(ungrouped)} images")
    print(f"  → fixed 80/20 split (seed={SEED}), same slice every fold")

    # Fixed ungrouped split — identical across all folds
    random.shuffle(ungrouped)
    cut       = int(0.8 * len(ungrouped))
    ung_train = ungrouped[:cut]
    ung_val   = ungrouped[cut:]

    game_names  = sorted(games.keys(), key=lambda x: -len(games[x]))
    all_results = []

    # ── LOOCV ────────────────────────────────────────────────────────────────
    for fold_idx, test_game in enumerate(game_names):
        fold_name = f"fold_{fold_idx + 1}_{safe_name(test_game)}"

        val_imgs   = games[test_game] + ung_val
        train_imgs = ung_train.copy()
        for g, imgs in games.items():
            if g != test_game:
                train_imgs.extend(imgs)

        print(f"\n{'=' * 65}")
        print(f"  Fold {fold_idx + 1}/{len(game_names)}  —  VAL game: {test_game}")
        print(f"  Train: {len(train_imgs)}  |  Val: {len(val_imgs)} "
              f"({len(games[test_game])} game + {len(ung_val)} ungrouped)")
        print(f"{'=' * 65}")

        metrics = train_fold(fold_name, train_imgs, val_imgs)
        row = {
            "fold":      fold_idx + 1,
            "test_game": test_game,
            "n_train":   len(train_imgs),
            "n_val":     len(val_imgs),
            **metrics,
        }
        all_results.append(row)
        print(f"  → mAP50={row['mAP50']}  mAP50-95={row['mAP50-95']}  "
              f"P={row['precision']}  R={row['recall']}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("LOOCV SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'Fold':<5} {'Test Game':<40} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7}")
    for r in all_results:
        print(f"{r['fold']:<5} {r['test_game']:<40} {r['mAP50']:>7.4f} "
              f"{r['mAP50-95']:>9.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")

    n = len(all_results)
    means = {k: round(sum(r[k] for r in all_results) / n, 4)
             for k in ("mAP50", "mAP50-95", "precision", "recall")}
    print(f"\n{'':5} {'MEAN':<40} {means['mAP50']:>7.4f} "
          f"{means['mAP50-95']:>9.4f} {means['precision']:>7.4f} {means['recall']:>7.4f}")

    fieldnames = ["fold", "test_game", "n_train", "n_val",
                  "mAP50", "mAP50-95", "precision", "recall"]
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nResults saved → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
