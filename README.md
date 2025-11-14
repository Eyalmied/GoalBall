_______________________________________
# Throw Detection & Labeling Pipeline
_______________________________________


------------Purpose--------------------------------
1. Detect ball, thrower, defender using a YOLO model.
2. Segment continuous ball motion into candidate "from" and "to" throw segments.
3. Manually validate segments and annotate outcomes.
4. Pair throws (from → to) with halftime side swap logic.
5. Export:
   - LSTM training CSV (per-frame features + user labels).
   - Excel summary of validated throws (zones, teams, coordinates).

## Dependencies
Install with:
bash:

!pip install ultralytics opencv-python pandas numpy openpyxl


------------Inputs----------------------------

Video: Path/To/Video.MOV or .mp4
Model: ball+players_tuning10/weights/best.pt
User interactive inputs:
- Lower team name (pre-halftime bottom side).
- Upper team name (pre-halftime top side).
- Halftime timestamp (hh:mm:ss or mm:ss).
- 8 calibration clicks (goal-frame corners).
After processing:
- Segment validation (keys: 1 = correct, 0 = false).
- Outcome for correct "to" segments: g (goal), b (block), o (out).

Calibration Click Order
lower right bottom
lower right top
lower left bottom
lower left top
upper left bottom
upper left top
upper right bottom
upper right top
These define two quadrilaterals (lower goal, upper goal) and their horizontal bars for 9 zone partitions.

----------------------------------------------------------------
