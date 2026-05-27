"""
Step 1: Extract 1-frame-per-second JPEGs from a game video for CVAT annotation.

After running this script, upload the output folder to CVAT, annotate the
ball (class 32 / sports_ball) and players (class 0 / throwing_player,
class 1 / defending_player), then export the annotations as YOLO-format .txt
files alongside the images.

Then run 2_class2to32_and_split.py to remap class IDs and build the
train/val dataset expected by yolo_loocv.py and yolo_final.py.
"""

import cv2
import os

# === CONFIGURE THESE ===
game         = "Gold_TUR-ISR_W_8-4_Euro_2023"
video_path   = r"C:\path\to\your\video.mpg"          # update to your video path
output_folder = r"C:\path\to\your\frames_output"      # where to save JPEGs
# =======================

prefix = f"{game}_F_image"   # filename prefix — keep this format for LOOCV grouping

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    raise ValueError("Failed to get FPS from video.")
frame_interval = int(fps)   # 1 frame per second

frame_index = 0
saved_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_index % frame_interval == 0:
        saved_count += 1
        filename = f"{prefix}{saved_count}.jpg"
        cv2.imwrite(os.path.join(output_folder, filename), frame)

    frame_index += 1

cap.release()
print(f"Done. Saved {saved_count} frames (1 per second) → {output_folder}")
