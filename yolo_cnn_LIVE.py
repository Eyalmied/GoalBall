# # yolo_cnn_LIVE.py
# # Requirements:
# #   pip install ultralytics opencv-python pandas numpy openpyxl
#
# from ultralytics import YOLO
# import cv2
# import pandas as pd
# import numpy as np
# import time, os, collections
# from pathlib import Path
#
# # ==========================
# # ===== USER SETTINGS  =====
# # ==========================
# MODEL_PATH   = "ball+players_tuning10/weights/best.pt"  # <- your trained YOLO weights
# OUTPUT_ROOT  = "Paralkympics2024"                       # <- base folder for outputs
# GAME_NAME    = "TEST1Lve"                               # <- choose a name per half
# CAM_INDEX    = 0                                        # <- webcam device index
#
# # Hyperparameters / knobs
# CFG = dict(
#     CONF_THRESHOLD   = 0.70,   # YOLO confidence threshold
#     MAX_GAP          = 2,      # frames to tolerate no ball while keeping current segment
#     EXTEND_TO_FRAMES = 80,     # how many frames to append to "to" segments for review/outcome
#     N_MEAN_FROM      = 5,      # average first N coords for "from" centroid
#     N_MEAN_TO        = 5,      # average last  N coords for "to" centroid
#     TARGET_FPS       = 30.0,   # FPS for temporary video used during review and timing
#     BALL_TRAIL_LEN   = 25,     # length of live ball trail
#     # suppress quick duplicate FROM after a TO on the same side (frames)
#     SUPPRESS_AFTER_TO_WINDOW_FRAMES = 45,
# )
#
# # Live overlay flags (toggles: D/Z/B keys)
# FLAGS = {
#     'SHOW_DETECTIONS': True,   # draw YOLO boxes + labels + conf
#     'SHOW_ZONES': True,        # draw goal polygons + zone bars + numbers
#     'SHOW_BALL_TRAIL': False,  # default OFF; toggle with B
# }
#
# # UI sizing (smaller/leaner annotations)
# UI_FONT_SCALE = 0.5
# UI_THICKNESS  = 1
# UI_ZONE_LINE_THICKNESS = 2
# UI_BOX_THICKNESS = 2
# UI_SMALL_FONT = 0.45
#
# # Class mapping used in your training:
# # 0 = thrower, 1 = defender, 32 = ball
# CLASS_THROWER = 0
# CLASS_DEFENDER= 1
# CLASS_BALL    = 32
# CLASS_NAMES   = {CLASS_THROWER: "thrower", CLASS_DEFENDER: "defender", CLASS_BALL: "ball"}
#
# # Resolve output paths
# GAME_DIR = Path(OUTPUT_ROOT) / GAME_NAME
# OUT_DIR  = GAME_DIR / "outputs"
# OUT_DIR.mkdir(parents=True, exist_ok=True)
#
# LSTM_CSV_PATH        = OUT_DIR / f"{GAME_NAME}_Throws_lstm_training.csv"
# THROWS_XLSX_PATH     = OUT_DIR / f"{GAME_NAME}_Throws_data.xlsx"
# THROWS_CSV_FALLBACK  = OUT_DIR / f"{GAME_NAME}_Throws_data.csv"
# TMP_VIDEO_PATH       = OUT_DIR / f"{GAME_NAME}_TmpCapture.avi"   # recorded live to allow review
#
# # ==========================
# # ======== HELPERS =========
# # ==========================
# def draw_text(img, text, org, scale=None, color=(255,255,255), thick=None):
#     if scale is None: scale = UI_FONT_SCALE
#     if thick is None: thick = UI_THICKNESS
#     cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
#     cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
#
# def compute_zone(P, A, B):
#     """Return zone index 1..9 along segment AB for point P (projected)."""
#     P, A, B = np.array(P, float), np.array(A, float), np.array(B, float)
#     u = np.dot(P - A, B - A) / (np.dot(B - A, B - A) + 1e-6)
#     u = float(np.clip(u, 0.0, 1.0))
#     # 9 equal bins along [0,1]; clamp to 0..8 then +1 => 1..9
#     idx = int(np.floor(u * 9.0))
#     idx = int(np.clip(idx, 0, 8))
#     return idx + 1
#
# def compute_mean_coord(coords, segment_type, n=10):
#     if not coords:
#         return np.array([np.nan, np.nan])
#     sel = coords[:n] if segment_type == 'from' else coords[-n:]
#     return np.mean(sel, axis=0)
#
# def which_goal(pt, lower_poly, upper_poly, frame_h):
#     if cv2.pointPolygonTest(lower_poly, tuple(pt), False) >= 0: return 'lower'
#     if cv2.pointPolygonTest(upper_poly, tuple(pt), False) >= 0: return 'upper'
#     return 'lower' if pt[1] > frame_h / 2 else 'upper'
#
# def ensure_window(win, w=None, h=None):
#     cv2.namedWindow(win, cv2.WINDOW_NORMAL)
#     if w and h:
#         cv2.resizeWindow(win, w, h)
#
# def draw_detections(ov, xyxy, cls_ids, confs):
#     for (x1, y1, x2, y2), c, cf in zip(xyxy, cls_ids, confs):
#         x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
#         color = (0, 200, 0) if c == CLASS_THROWER else (0, 0, 200) if c == CLASS_DEFENDER else (220, 220, 0)
#         cv2.rectangle(ov, (x1, y1), (x2, y2), color, UI_BOX_THICKNESS)
#         name = CLASS_NAMES.get(int(c), f"id{int(c)}")
#         draw_text(ov, f"{name} {cf:.2f}", (x1, max(0, y1 - 6)), UI_SMALL_FONT, color, UI_THICKNESS)
#
# def first_thrower_center(segment, frame_features, n=1):
#     """Centroid of thrower (class 0) in earliest n frames of a FROM segment."""
#     centres = []
#     for f in segment['frames'][:n]:
#         ff = frame_features[f - 1]
#         if ff['thrower_x'] is not None:
#             centres.append((ff['thrower_x'], ff['thrower_y']))
#     return np.mean(centres, axis=0) if centres else (np.nan, np.nan)
#
# def thrower_side_from_feat(feat, frame_h):
#     """Estimate side for a potential FROM using the thrower center if available; fallback to ball y."""
#     if feat['thrower_x'] is not None:
#         return 'lower' if feat['thrower_y'] > frame_h/2 else 'upper'
#     if feat['ball_y'] is not None:
#         return 'lower' if feat['ball_y'] > frame_h/2 else 'upper'
#     return None
#
# # ==========================
# # ======== SETUP ===========
# # ==========================
# model = YOLO(MODEL_PATH)
#
# # 1) Grab one frame from webcam for calibration
# cap = cv2.VideoCapture(CAM_INDEX)
# if not cap.isOpened():
#     raise RuntimeError(f"Cannot open webcam device index {CAM_INDEX}")
# time.sleep(0.5)
# ok, calib = cap.read()
# if not ok:
#     cap.release()
#     raise RuntimeError("Failed to read a frame from webcam for calibration.")
#
# ensure_window("Calibrate", 1280, 720)
# cv2.imshow("Calibrate", calib.copy())
#
# # 2) Get team names (fixed sides for this half)
# lower_team = input("Enter LOWER team name (e.g., ISR): ").strip()
# upper_team = input("Enter UPPER team name (e.g., CAN): ").strip()
#
# # 3) Click 8 goal-frame corners (same order)
# labels = [
#     "1) lower right bottom", "2) lower right top",
#     "3) lower left bottom",  "4) lower left top",
#     "5) upper left bottom",  "6) upper left top",
#     "7) upper right bottom", "8) upper right top",
# ]
# pts = []
# def on_click(evt, x, y, flags, param):
#     if evt == cv2.EVENT_LBUTTONDOWN and len(pts) < 8:
#         pts.append((x, y))
#         frame = calib.copy()
#         for i, p in enumerate(pts):
#             cv2.circle(frame, p, 4, (0,0,255), -1)
#             cv2.putText(frame, labels[i], (p[0] + 4, p[1] - 4),
#                         cv2.FONT_HERSHEY_SIMPLEX, UI_SMALL_FONT, (0,0,255), 1)
#         cv2.imshow("Calibrate", frame)
#
# cv2.setMouseCallback("Calibrate", on_click)
# print("Click the 8 points in order:")
# for l in labels: print(" ", l)
#
# while len(pts) < 8:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyWindow("Calibrate")
# if len(pts) < 8:
#     cap.release()
#     raise RuntimeError("Calibration aborted: 8 points were not selected.")
#
# p1,p2,p3,p4,p5,p6,p7,p8 = [np.array(p, int) for p in pts]
# # lower and upper goal polygons
# lower_poly = np.array([p1,p2,p4,p3], dtype=np.int32)
# upper_poly = np.array([p5,p6,p8,p7], dtype=np.int32)
#
# # bars (tick centers) for zone numbering
# low_rt, low_lt = p2, p4
# up_lt,  up_rt  = p6, p8
# low_bar = [low_rt + (low_lt - low_rt) * ((k + 0.5) / 9) for k in range(9)]
# up_bar  = [up_lt  + (up_rt  - up_lt)  * ((k + 0.5) / 9) for k in range(9)]
#
# frame_h, frame_w = calib.shape[:2]
#
# # ==========================
# # ========== LIVE ==========
# # ==========================
# print("\n--- LIVE MODE ---")
# print("Keys: [H] end half & review | [Q] abort | [D] detections on/off | [Z] zones on/off | [B] ball trail on/off")
#
# # Prepare temp video writer for review
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# writer = cv2.VideoWriter(str(TMP_VIDEO_PATH), fourcc, CFG['TARGET_FPS'], (frame_w, frame_h))
# if not writer.isOpened():
#     cap.release()
#     raise RuntimeError("Failed to open VideoWriter for temporary recording.")
#
# # Online buffers
# frame_idx = 0
# gap = 0
# cur = {'label': None, 'coords': [], 'frames': []}
# seq = []                     # raw segments alternating 'from' / 'to'
# frame_features = []          # per-frame features for LSTM CSV
# ball_trail = collections.deque(maxlen=CFG['BALL_TRAIL_LEN'])
#
# # state for suppressing duplicate FROM after a TO
# last_to_end_frame = -10**9
# last_to_side = None
#
# ensure_window("Live", 1280, 720)
# fps_smooth = 0.0
#
# while True:
#     ok, frame = cap.read()
#     if not ok:
#         print("[INFO] Webcam read returned false. Ending half.")
#         break
#
#     frame_idx += 1
#     t_infer0 = time.time()
#
#     # YOLO inference
#     res = model.predict(frame, save=False, conf=CFG['CONF_THRESHOLD'], verbose=False)[0]
#     boxes   = res.boxes
#     if boxes is not None and len(boxes) > 0:
#         cls_ids = boxes.cls.cpu().numpy()
#         xywh    = boxes.xywh.cpu().numpy()
#         xyxy    = boxes.xyxy.cpu().numpy()
#         confs   = boxes.conf.cpu().numpy()
#     else:
#         cls_ids = np.array([]); xywh = np.zeros((0,4)); xyxy = np.zeros((0,4)); confs = np.array([])
#
#     # pick label for segment logic (based on players presence)
#     players = [c for c in cls_ids if c in (CLASS_THROWER, CLASS_DEFENDER)]
#     label   = 1 if len(players) > 1 else (players[0] if len(players) == 1 else None)
#
#     # per-frame feature record
#     feat = dict(frame=frame_idx, label=label,
#                 ball_x=None, ball_y=None, ball_w=None, ball_h=None, ball_conf=None,
#                 thrower_x=None, thrower_y=None, thrower_w=None, thrower_h=None, thrower_conf=None,
#                 defender_x=None, defender_y=None, defender_w=None, defender_h=None, defender_conf=None)
#
#     # ball centers (for segment coords)
#     balls = []
#     for c,b,bb,cf in zip(cls_ids, xywh, xyxy, confs):
#         x,y,w,h = map(float, b)
#         if   c == CLASS_BALL     and feat['ball_x']    is None:
#             feat.update(ball_x=x,ball_y=y,ball_w=w,ball_h=h,ball_conf=float(cf))
#             balls.append(b[:2])  # center
#             if FLAGS['SHOW_BALL_TRAIL']:
#                 ball_trail.append((int(x), int(y)))
#         elif c == CLASS_THROWER  and feat['thrower_x'] is None:
#             feat.update(thrower_x=x,thrower_y=y,thrower_w=w,thrower_h=h,thrower_conf=float(cf))
#         elif c == CLASS_DEFENDER and feat['defender_x']is None:
#             feat.update(defender_x=x,defender_y=y,defender_w=w,defender_h=h,defender_conf=float(cf))
#     frame_features.append(feat)
#
#     # --------- segment builder with suppression ----------
#     if label is not None and balls:
#         new_label = label
#
#         # Suppress immediate FROM on the same side right after a TO
#         if new_label == CLASS_THROWER:
#             side_now = thrower_side_from_feat(feat, frame_h)
#             if (side_now is not None and
#                 last_to_side is not None and
#                 side_now == last_to_side and
#                 (frame_idx - last_to_end_frame) <= CFG['SUPPRESS_AFTER_TO_WINDOW_FRAMES']):
#                 # ignore this potential FROM; treat as transient noise
#                 pass
#             else:
#                 gap = 0
#                 if cur['label'] is None:
#                     cur = {'label': new_label, 'coords': [], 'frames': []}
#                 if cur['label'] == new_label:
#                     cur['coords'].append(balls[0]); cur['frames'].append(frame_idx)
#                 else:
#                     seg_type = 'from' if cur['label'] == CLASS_THROWER else 'to'
#                     if cur['frames']:
#                         if seg_type == 'to':
#                             last_to_end_frame = cur['frames'][-1]
#                             to_pt = compute_mean_coord(cur['coords'], 'to', n=CFG['N_MEAN_TO'])
#                             last_to_side = 'lower' if to_pt[1] > frame_h/2 else 'upper'
#                         seq.append({'type': seg_type, 'coords': cur['coords'], 'frames': cur['frames']})
#                     cur = {'label': new_label, 'coords': [balls[0]], 'frames': [frame_idx]}
#         else:
#             # defender (TO) or both players -> proceed normally
#             gap = 0
#             if cur['label'] is None:
#                 cur = {'label': new_label, 'coords': [], 'frames': []}
#             if cur['label'] == new_label:
#                 cur['coords'].append(balls[0]); cur['frames'].append(frame_idx)
#             else:
#                 seg_type = 'from' if cur['label'] == CLASS_THROWER else 'to'
#                 if cur['frames']:
#                     if seg_type == 'to':
#                         last_to_end_frame = cur['frames'][-1]
#                         to_pt = compute_mean_coord(cur['coords'], 'to', n=CFG['N_MEAN_TO'])
#                         last_to_side = 'lower' if to_pt[1] > frame_h/2 else 'upper'
#                     seq.append({'type': seg_type, 'coords': cur['coords'], 'frames': cur['frames']})
#                 cur = {'label': new_label, 'coords': [balls[0]], 'frames': [frame_idx]}
#     else:
#         # no players or no ball -> maybe close current after a gap
#         if cur['label'] is not None:
#             gap += 1
#             if gap > CFG['MAX_GAP']:
#                 seg_type = 'from' if cur['label'] == CLASS_THROWER else 'to'
#                 if cur['frames']:
#                     if seg_type == 'to':
#                         last_to_end_frame = cur['frames'][-1]
#                         to_pt = compute_mean_coord(cur['coords'], 'to', n=CFG['N_MEAN_TO'])
#                         last_to_side = 'lower' if to_pt[1] > frame_h/2 else 'upper'
#                     seq.append({'type': seg_type, 'coords': cur['coords'], 'frames': cur['frames']})
#                 cur = {'label': None, 'coords': [], 'frames': []}
#                 gap = 0
#     # ----------------------------------------------------
#
#     # write raw frame for later review/seek
#     writer.write(frame)
#
#     # ------- live overlay -------
#     ov = frame.copy()
#
#     # draw detections
#     if FLAGS['SHOW_DETECTIONS'] and len(cls_ids) > 0:
#         draw_detections(ov, xyxy, cls_ids, confs)
#
#     # draw goal polys, zones, numbers
#     if FLAGS['SHOW_ZONES']:
#         cv2.polylines(ov, [lower_poly], True, (0, 180, 0), UI_ZONE_LINE_THICKNESS)
#         cv2.polylines(ov, [upper_poly], True, (0, 180, 0), UI_ZONE_LINE_THICKNESS)
#         for k, pos in enumerate(low_bar, 1):
#             x, y = map(int, pos); draw_text(ov, str(k), (x-10, y+18), 0.6, (180,0,180), UI_THICKNESS)
#         cv2.line(ov, tuple(low_rt), tuple(low_lt), (180, 0, 180), UI_ZONE_LINE_THICKNESS)
#         for k, pos in enumerate(up_bar, 1):
#             x, y = map(int, pos); draw_text(ov, str(k), (x-10, y-8), 0.6, (180,0,180), UI_THICKNESS)
#         cv2.line(ov, tuple(up_lt), tuple(up_rt), (180, 0, 180), UI_ZONE_LINE_THICKNESS)
#
#     # ball trail + current ball dot
#     if FLAGS['SHOW_BALL_TRAIL'] and len(ball_trail) > 1:
#         for i in range(1, len(ball_trail)):
#             cv2.line(ov, ball_trail[i-1], ball_trail[i], (220,220,0), 2)
#     if feat['ball_x'] is not None:
#         bx, by = int(feat['ball_x']), int(feat['ball_y'])
#         cv2.circle(ov, (bx, by), 6, (220,220,0), -1)
#
#     # HUD
#     dt = time.time() - t_infer0
#     fps = 1.0 / max(dt, 1e-6)
#     fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps
#     seg_state = "none" if cur['label'] is None else ("FROM" if cur['label']==CLASS_THROWER else "TO")
#     draw_text(ov, f"FPS {fps_smooth:4.1f} | Segment: {seg_state}", (10, 26), 0.65, (255,255,255), UI_THICKNESS)
#     draw_text(ov, f"Lower: {lower_team} | Upper: {upper_team}", (10, 50), 0.65, (0,255,255), UI_THICKNESS)
#     draw_text(ov, "H=end half  Q=abort  D=det  Z=zones  B=trail", (10, frame_h - 16), 0.6, (255,255,0), UI_THICKNESS)
#
#     cv2.imshow("Live", ov)
#     key = cv2.waitKey(1) & 0xFF
#     if key in (ord('q'), ord('Q')):
#         print("[INFO] Aborted by user (Q).")
#         break
#     if key in (ord('h'), ord('H')):
#         print("[INFO] Half ended by user (H).")
#         break
#     if key in (ord('d'), ord('D')):
#         FLAGS['SHOW_DETECTIONS'] = not FLAGS['SHOW_DETECTIONS']
#     if key in (ord('z'), ord('Z')):
#         FLAGS['SHOW_ZONES'] = not FLAGS['SHOW_ZONES']
#     if key in (ord('b'), ord('B')):
#         FLAGS['SHOW_BALL_TRAIL'] = not FLAGS['SHOW_BALL_TRAIL']
#
# # close any open current segment
# if cur['label'] is not None and cur['frames']:
#     seg_type = 'from' if cur['label'] == CLASS_THROWER else 'to'
#     if seg_type == 'to':
#         last_to_end_frame = cur['frames'][-1]
#         to_pt = compute_mean_coord(cur['coords'], 'to', n=CFG['N_MEAN_TO'])
#         last_to_side = 'lower' if to_pt[1] > frame_h/2 else 'upper'
#     seq.append({'type': seg_type, 'coords': cur['coords'], 'frames': cur['frames']})
#
# # Done capturing
# cap.release()
# writer.release()
# cv2.destroyWindow("Live")
#
# # ==========================
# # ====== POSTPROCESS =======
# # ==========================
# # collapse consecutive same-type segments
# collapsed, grp = [], None
# for seg in seq:
#     if grp is None or grp['type'] != seg['type']:
#         grp = {'type': seg['type'], 'coords': seg['coords'][:], 'frames': seg['frames'][:]}
#         collapsed.append(grp)
#     else:
#         grp['coords'].extend(seg['coords'])
#         grp['frames'].extend(seg['frames'])
#
# # extend each "to" by N frames (bounded by last captured frame)
# max_frames = frame_idx
# for seg in collapsed:
#     if seg['type'] == 'to':
#         last = seg['frames'][-1]
#         seg['frames'].extend(range(last + 1, min(last + 1 + CFG['EXTEND_TO_FRAMES'], max_frames)))
#
# # re-map identities to stable ids post-collapse
# segment_id_of = {id(seg): idx + 1 for idx, seg in enumerate(collapsed)}
#
# # ==========================
# # ======== REVIEW UI =======
# # ==========================
# print("\n--- REVIEW / LABEL ---")
# print("Controls: SPACE=play/pause | ←/→=step (paused) | N=next seg | q=stop current seg | Esc=abort all")
# print("For each segment: press '1' (real throw) or '0' if false positive.")
# print("For 'to' segments marked as real, press 'g' goal / 'b' block / 'o' out.")
#
# training_records = []
# seg_cap = cv2.VideoCapture(str(TMP_VIDEO_PATH))
# if not seg_cap.isOpened():
#     raise RuntimeError("Failed to open temporary recording for review.")
#
# ensure_window("Label Segment", 1280, 720)
# delay_ms = max(1, int(1000.0 / max(1.0, CFG['TARGET_FPS'])))
#
# def show_frame_at(f_idx, overlay_fn):
#     seg_cap.set(cv2.CAP_PROP_POS_FRAMES, max(f_idx - 1, 0))
#     ret, img = seg_cap.read()
#     if not ret:
#         return False
#     ov = img.copy()
#     overlay_fn(ov)
#     cv2.imshow("Label Segment", ov)
#     return True
#
# for i, seg in enumerate(collapsed):
#     start, end = seg['frames'][0], seg['frames'][-1]
#     seg_type = seg['type']
#     print(f"\nSegment {i+1}/{len(collapsed)} — {seg_type} — frames {start}→{end}")
#
#     # Prepare overlay function (zones + per-frame detections)
#     def overlay(ov_img, f=None):
#         cv2.polylines(ov_img, [lower_poly], True, (0, 180, 0), UI_ZONE_LINE_THICKNESS)
#         cv2.polylines(ov_img, [upper_poly], True, (0, 180, 0), UI_ZONE_LINE_THICKNESS)
#         for k, pos in enumerate(low_bar, 1):
#             x, y = map(int, pos); draw_text(ov_img, str(k), (x-10, y+18), 0.6, (180,0,180), UI_THICKNESS)
#         cv2.line(ov_img, tuple(low_rt), tuple(low_lt), (180, 0, 180), UI_ZONE_LINE_THICKNESS)
#         for k, pos in enumerate(up_bar, 1):
#             x, y = map(int, pos); draw_text(ov_img, str(k), (x-10, y-8), 0.6, (180,0,180), UI_THICKNESS)
#         cv2.line(ov_img, tuple(up_lt), tuple(up_rt), (180, 0, 180), UI_ZONE_LINE_THICKNESS)
#
#         if f is not None and 0 < f <= len(frame_features):
#             ff = frame_features[f - 1]
#             if ff['ball_x'] is not None:
#                 cx, cy = int(ff['ball_x']), int(ff['ball_y'])
#                 cv2.circle(ov_img, (cx, cy), 6, (220,220,0), -1)
#                 draw_text(ov_img, "ball", (cx+8, cy+8), UI_SMALL_FONT, (220,220,0), UI_THICKNESS)
#             if ff['thrower_x'] is not None:
#                 x,y,w,h = ff['thrower_x'], ff['thrower_y'], ff['thrower_w'], ff['thrower_h']
#                 tl, br = (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2))
#                 cv2.rectangle(ov_img, tl, br, (0,200,0), UI_BOX_THICKNESS)
#                 draw_text(ov_img, "thrower", (tl[0], max(0, tl[1]-8)), UI_SMALL_FONT, (0,200,0), UI_THICKNESS)
#             if ff['defender_x'] is not None:
#                 x,y,w,h = ff['defender_x'], ff['defender_y'], ff['defender_w'], ff['defender_h']
#                 tl, br = (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2))
#                 cv2.rectangle(ov_img, tl, br, (0,0,200), UI_BOX_THICKNESS)
#                 draw_text(ov_img, "defender", (tl[0], max(0, tl[1]-8)), UI_SMALL_FONT, (0,0,200), UI_THICKNESS)
#
#         big_label = "THROW FROM" if seg_type == "from" else "THROW TO"
#         draw_text(ov_img, f"Segment {i+1}/{len(collapsed)}: {seg_type} ({start}-{end})", (10, 24), 0.65, (255,255,255), UI_THICKNESS)
#         draw_text(ov_img, big_label, (10, 48), 0.8, (0,255,255), UI_THICKNESS)
#         draw_text(ov_img, f"{lower_team} (Lower) / {upper_team} (Upper)", (10, 72), 0.65, (0,255,255), UI_THICKNESS)
#
#     # --- Playback controls ---
#     playing = True
#     fpos = 0
#     seg_frames = seg['frames'][:]  # local copy of frames in this segment
#
#     while True:
#         if playing:
#             if fpos >= len(seg_frames):
#                 break
#             f = seg_frames[fpos]
#             if not show_frame_at(f, lambda ov_: overlay(ov_, f)):
#                 break
#             key = cv2.waitKey(max(1, int(1000.0 / max(1.0, CFG['TARGET_FPS'])))) & 0xFF
#             fpos += 1
#         else:
#             # paused: stay on current fpos (clamped)
#             fpos = max(0, min(fpos, len(seg_frames)-1))
#             f = seg_frames[fpos]
#             if not show_frame_at(f, lambda ov_: overlay(ov_, f)):
#                 break
#             key = cv2.waitKey(0) & 0xFF
#
#         if key == 27:  # Esc => abort all review
#             print("[INFO] Review aborted.")
#             seg_cap.release()
#             cv2.destroyWindow("Label Segment")
#             raise SystemExit(0)
#         elif key == ord(' '):            # SPACE toggles play/pause
#             playing = not playing
#         elif key == ord('n') or key == ord('N'):
#             break                         # skip to next segment
#         elif key in (ord('q'), ord('Q')):
#             break                         # stop current segment playback, go to label
#         elif not playing:
#             # stepping only when paused
#             if key == 83:                 # Right arrow
#                 fpos = min(fpos + 1, len(seg_frames)-1)
#             elif key == 81:               # Left arrow
#                 fpos = max(fpos - 1, 0)
#
#     # ----- segment decision -----
#     print("Press '1' if this is a real throw, '0' if false positive.")
#     while True:
#         k = cv2.waitKey(0)
#         if k in (ord('1'), ord('0')):
#             seg_label = 1 if k == ord('1') else 0
#             break
#
#     outcome = ''
#     if seg_label == 1 and seg_type == 'to':
#         print("Enter outcome: 'g' for goal, 'b' for block, 'o' for out")
#         while True:
#             k = cv2.waitKey(0)
#             if k in (ord('g'), ord('b'), ord('o')):
#                 outcome = chr(k)
#                 break
#
#     # write training records for each frame in the segment
#     for f in seg_frames:
#         ff = frame_features[f - 1]
#         rec = {
#             'segment_id': i + 1,
#             'segment_type': seg_type,
#             'frame': f,
#             'label': seg_label,
#             'outcome': outcome
#         }
#         rec.update({k: v for k, v in ff.items() if k not in ('frame', 'label')})
#         training_records.append(rec)
#
# seg_cap.release()
# cv2.destroyWindow("Label Segment")
#
# # ==========================
# # ====== FILTER & SAVE =====
# # ==========================
# # Pair FROM -> next TO (keep orphan FROM at end)
# throws = []
# pending_from = None
# for seg in collapsed:
#     if seg['type'] == 'from':
#         pending_from = seg
#     elif seg['type'] == 'to':
#         throws.append({'num': len(throws)+1, 'from': pending_from, 'to': seg, 'incomplete': False})
#         pending_from = None
# if pending_from is not None:
#     throws.append({'num': len(throws)+1, 'from': pending_from, 'to': None, 'incomplete': True})
#
# # keep only segments marked as correct (and always keep orphan FROM)
# seg_labels = {rec['segment_id']: rec['label'] for rec in training_records}
# segment_id_of = {id(seg): idx + 1 for idx, seg in enumerate(collapsed)}  # re-map after collapse
# good_throws = []
# for t in throws:
#     if t['to'] is None:
#         good_throws.append(t)
#     else:
#         seg_id = segment_id_of[id(t['to'])]
#         if seg_labels.get(seg_id, 1) == 1:
#             good_throws.append(t)
# throws = good_throws
#
# # ==========================
# # ===== RESULTS TABLE  =====
# # ==========================
# results = []
# for t in throws:
#     num = t['num']
#
#     # ---------- FROM point & side ----------
#     fc = compute_mean_coord(t['from']['coords'], 'from', n=CFG['N_MEAN_FROM'])
#     # prefer detected thrower; fallback to ball cluster
#     thr_xy = first_thrower_center(t['from'], frame_features, n=1)
#     from_pt = thr_xy if not np.isnan(thr_xy[0]) else fc
#     from_side = which_goal(from_pt, lower_poly, upper_poly, frame_h)  # 'lower' or 'upper'
#
#     # teams by side (fixed this half)
#     throwing  = lower_team if from_side == 'lower' else upper_team
#     defending = upper_team if from_side == 'lower' else lower_team
#
#     # FROM zone by side
#     if from_side == 'lower':
#         fz = compute_zone(fc, low_rt, low_lt); fz_label = f"{fz} Lower"
#     else:
#         fz = compute_zone(fc, up_lt, up_rt);   fz_label = f"{fz} Upper"
#
#     # ---------- TO point & side ----------
#     if t['to'] is None:
#         tc = np.array([np.nan, np.nan], dtype=float)
#         tz_label = np.nan
#     else:
#         tc = compute_mean_coord(t['to']['coords'], 'to', n=CFG['N_MEAN_TO'])
#         to_side = which_goal(tc, lower_poly, upper_poly, frame_h)
#         defending_side = to_side  # use actual TO side, do NOT assume opposite
#         if defending_side == 'lower':
#             tz = compute_zone(tc, low_rt, low_lt); tz_label = f"{tz} Lower"
#         else:
#             tz = compute_zone(tc, up_lt, up_rt);   tz_label = f"{tz} Upper"
#
#     results.append({
#         'Throw Number':   num,
#         'From Zone':      fz_label,
#         'To Zone':        tz_label,
#         'Throwing Team':  throwing,
#         'Defending Team': defending,
#         'From Coord':     fc.tolist(),
#         'To Coord':       tc.tolist()
#     })
#
# # Save outputs
# pd.DataFrame(training_records).to_csv(LSTM_CSV_PATH, index=False)
#
# # Excel with graceful fallback if openpyxl missing in this interpreter
# try:
#     pd.DataFrame(results).to_excel(THROWS_XLSX_PATH, index=False)
#     print("\n=== DONE ===")
#     print(f"Saved LSTM training CSV → {LSTM_CSV_PATH}")
#     print(f"Saved Throws Excel      → {THROWS_XLSX_PATH}")
# except ModuleNotFoundError:
#     pd.DataFrame(results).to_csv(THROWS_CSV_FALLBACK, index=False)
#     print("\n[WARN] openpyxl is not available in this Python interpreter.")
#     print("       Wrote CSV fallback instead.")
#     print(f"Saved LSTM training CSV → {LSTM_CSV_PATH}")
#     print(f"Saved Throws CSV        → {THROWS_CSV_FALLBACK}")
#     print("Tip: run this script with the same Python where you installed openpyxl,")
#     print("     or `pip install openpyxl` in THIS interpreter.")








# yolo_cnn_LIVE.py
# Requirements:
#   pip install ultralytics opencv-python pandas numpy openpyxl

from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import time, os, collections
from pathlib import Path

# ==========================
# ===== USER SETTINGS  =====
# ==========================
MODEL_PATH   = "ball+players_tuning10/weights/best.pt"  # <- your trained YOLO weights
OUTPUT_ROOT  = "Paralkympics2024"                       # <- base folder for outputs
GAME_NAME    = "TEST1Lve"                               # <- choose a name per half
CAM_INDEX = 0                                 # <- webcam device index

# Hyperparameters / knobs
CFG = dict(
    CONF_THRESHOLD   = 0.70,   # YOLO confidence threshold
    MAX_GAP          = 2,      # frames to tolerate no ball while keeping current segment
    EXTEND_TO_FRAMES = 80,     # how many frames to append to "to" segments for review/outcome
    N_MEAN_FROM      = 5,      # average first N coords for "from" centroid
    N_MEAN_TO        = 5,      # average last  N coords for "to" centroid
    TARGET_FPS       = 30.0,   # FPS for temporary video used during review and timing
    BALL_TRAIL_LEN   = 25,     # length of live ball trail

    # Suppress quick FROM after a TO on same side
    SUPPRESS_AFTER_TO_WINDOW_FRAMES = 45,

    # NEW: de-dup similar FROM→TO pairs that cover the same play
    DEDUP_TIME_IOU        = 0.50,  # IoU of time ranges to consider duplicate
    DEDUP_MAX_START_DIFF  = 12,    # frames: |start1-start2| small => duplicate if sides match
    DEDUP_MAX_END_DIFF    = 20,    # frames: |end1-end2| small => duplicate if sides match
)

# Live overlay flags (toggles: D/Z/B keys)
FLAGS = {
    'SHOW_DETECTIONS': True,   # draw YOLO boxes + labels + conf
    'SHOW_ZONES': True,        # draw goal polygons + zone bars + numbers
    'SHOW_BALL_TRAIL': False,  # default OFF; toggle with B
}

# UI sizing (smaller/leaner annotations)
UI_FONT_SCALE = 0.5
UI_THICKNESS  = 1
UI_ZONE_LINE_THICKNESS = 2
UI_BOX_THICKNESS = 2
UI_SMALL_FONT = 0.45

# Class mapping used in your training:
# 0 = thrower, 1 = defender, 32 = ball
CLASS_THROWER = 0
CLASS_DEFENDER= 1
CLASS_BALL    = 32
CLASS_NAMES   = {CLASS_THROWER: "thrower", CLASS_DEFENDER: "defender", CLASS_BALL: "ball"}

# Resolve output paths
GAME_DIR = Path(OUTPUT_ROOT) / GAME_NAME
OUT_DIR  = GAME_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LSTM_CSV_PATH        = OUT_DIR / f"{GAME_NAME}_Throws_lstm_training.csv"
THROWS_XLSX_PATH     = OUT_DIR / f"{GAME_NAME}_Throws_data.xlsx"
THROWS_CSV_FALLBACK  = OUT_DIR / f"{GAME_NAME}_Throws_data.csv"
TMP_VIDEO_PATH       = OUT_DIR / f"{GAME_NAME}_TmpCapture.avi"   # recorded live to allow review

# ==========================
# ======== HELPERS =========
# ==========================
def draw_text(img, text, org, scale=None, color=(255,255,255), thick=None):
    if scale is None: scale = UI_FONT_SCALE
    if thick is None: thick = UI_THICKNESS
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def compute_zone(P, A, B):
    """Return zone index 1..9 along segment AB for point P (projected)."""
    P, A, B = np.array(P, float), np.array(A, float), np.array(B, float)
    u = np.dot(P - A, B - A) / (np.dot(B - A, B - A) + 1e-6)
    u = float(np.clip(u, 0.0, 1.0))
    idx = int(np.floor(u * 9.0))
    idx = int(np.clip(idx, 0, 8))
    return idx + 1

def compute_mean_coord(coords, segment_type, n=10):
    if not coords:
        return np.array([np.nan, np.nan])
    sel = coords[:n] if segment_type == 'from' else coords[-n:]
    return np.mean(sel, axis=0)

def which_goal(pt, lower_poly, upper_poly, frame_h):
    if cv2.pointPolygonTest(lower_poly, tuple(pt), False) >= 0: return 'lower'
    if cv2.pointPolygonTest(upper_poly, tuple(pt), False) >= 0: return 'upper'
    return 'lower' if pt[1] > frame_h / 2 else 'upper'

def ensure_window(win, w=None, h=None):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if w and h:
        cv2.resizeWindow(win, w, h)

def draw_detections(ov, xyxy, cls_ids, confs):
    for (x1, y1, x2, y2), c, cf in zip(xyxy, cls_ids, confs):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = (0, 200, 0) if c == CLASS_THROWER else (0, 0, 200) if c == CLASS_DEFENDER else (220, 220, 0)
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, UI_BOX_THICKNESS)
        name = CLASS_NAMES.get(int(c), f"id{int(c)}")
        draw_text(ov, f"{name} {cf:.2f}", (x1, max(0, y1 - 6)), UI_SMALL_FONT, color, UI_THICKNESS)

def first_thrower_center(segment, frame_features, n=1):
    """Centroid of thrower (class 0) in earliest n frames of a FROM segment."""
    centres = []
    for f in segment['frames'][:n]:
        ff = frame_features[f - 1]
        if ff['thrower_x'] is not None:
            centres.append((ff['thrower_x'], ff['thrower_y']))
    return np.mean(centres, axis=0) if centres else (np.nan, np.nan)

def thrower_side_from_feat(feat, frame_h):
    """Estimate side for a potential FROM using the thrower center if available; fallback to ball y."""
    if feat['thrower_x'] is not None:
        return 'lower' if feat['thrower_y'] > frame_h/2 else 'upper'
    if feat['ball_y'] is not None:
        return 'lower' if feat['ball_y'] > frame_h/2 else 'upper'
    return None

def interval_iou(a0, a1, b0, b1):
    """IoU for 1D intervals [a0,a1], [b0,b1] (inclusive indices are fine)."""
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(1e-6, (a1 - a0) + (b1 - b0) - inter)
    return inter / union

# ==========================
# ======== SETUP ===========
# ==========================
model = YOLO(MODEL_PATH)

# 1) Grab one frame from webcam for calibration
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open webcam device index {CAM_INDEX}")
time.sleep(0.5)
ok, calib = cap.read()
if not ok:
    cap.release()
    raise RuntimeError("Failed to read a frame from webcam for calibration.")

ensure_window("Calibrate", 1280, 720)
cv2.imshow("Calibrate", calib.copy())

# 2) Get team names (fixed sides for this half)
lower_team = input("Enter LOWER team name (e.g., ISR): ").strip()
upper_team = input("Enter UPPER team name (e.g., CAN): ").strip()

# 3) Click 8 goal-frame corners (same order)
labels = [
    "1) lower right bottom", "2) lower right top",
    "3) lower left bottom",  "4) lower left top",
    "5) upper left bottom",  "6) upper left top",
    "7) upper right bottom", "8) upper right top",
]
pts = []
def on_click(evt, x, y, flags, param):
    if evt == cv2.EVENT_LBUTTONDOWN and len(pts) < 8:
        pts.append((x, y))
        frame = calib.copy()
        for i, p in enumerate(pts):
            cv2.circle(frame, p, 4, (0,0,255), -1)
            cv2.putText(frame, labels[i], (p[0] + 4, p[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, UI_SMALL_FONT, (0,0,255), 1)
        cv2.imshow("Calibrate", frame)

cv2.setMouseCallback("Calibrate", on_click)
print("Click the 8 points in order:")
for l in labels: print(" ", l)

while len(pts) < 8:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Calibrate")
if len(pts) < 8:
    cap.release()
    raise RuntimeError("Calibration aborted: 8 points were not selected.")

p1,p2,p3,p4,p5,p6,p7,p8 = [np.array(p, int) for p in pts]
# lower and upper goal polygons
lower_poly = np.array([p1,p2,p4,p3], dtype=np.int32)
upper_poly = np.array([p5,p6,p8,p7], dtype=np.int32)

# bars (tick centers) for zone numbering
low_rt, low_lt = p2, p4
up_lt,  up_rt  = p6, p8
low_bar = [low_rt + (low_lt - low_rt) * ((k + 0.5) / 9) for k in range(9)]
up_bar  = [up_lt  + (up_rt  - up_lt)  * ((k + 0.5) / 9) for k in range(9)]

frame_h, frame_w = calib.shape[:2]

# ==========================
# ========== LIVE ==========
# ==========================
print("\n--- LIVE MODE ---")
print("Keys: [H] end half & review | [Q] abort | [D] detections on/off | [Z] zones on/off | [B] ball trail on/off")

# Prepare temp video writer for review
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(str(TMP_VIDEO_PATH), fourcc, CFG['TARGET_FPS'], (frame_w, frame_h))
if not writer.isOpened():
    cap.release()
    raise RuntimeError("Failed to open VideoWriter for temporary recording.")

# Online buffers
frame_idx = 0
gap = 0
cur = {'label': None, 'coords': [], 'frames': []}
seq = []                     # raw segments alternating 'from' / 'to'
frame_features = []          # per-frame features for LSTM CSV
ball_trail = collections.deque(maxlen=CFG['BALL_TRAIL_LEN'])

# suppressing duplicate FROM after a TO
last_to_end_frame = -10**9
last_to_side = None

ensure_window("Live", 1280, 720)
fps_smooth = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("[INFO] Webcam read returned false. Ending half.")
        break

    frame_idx += 1
    t_infer0 = time.time()

    # YOLO inference
    res = model.predict(frame, save=False, conf=CFG['CONF_THRESHOLD'], verbose=False)[0]
    boxes   = res.boxes
    if boxes is not None and len(boxes) > 0:
        cls_ids = boxes.cls.cpu().numpy()
        xywh    = boxes.xywh.cpu().numpy()
        xyxy    = boxes.xyxy.cpu().numpy()
        confs   = boxes.conf.cpu().numpy()
    else:
        cls_ids = np.array([]); xywh = np.zeros((0,4)); xyxy = np.zeros((0,4)); confs = np.array([])

    # pick label for segment logic (based on players presence)
    players = [c for c in cls_ids if c in (CLASS_THROWER, CLASS_DEFENDER)]
    label   = 1 if len(players) > 1 else (players[0] if len(players) == 1 else None)

    # per-frame feature record
    feat = dict(frame=frame_idx, label=label,
                ball_x=None, ball_y=None, ball_w=None, ball_h=None, ball_conf=None,
                thrower_x=None, thrower_y=None, thrower_w=None, thrower_h=None, thrower_conf=None,
                defender_x=None, defender_y=None, defender_w=None, defender_h=None, defender_conf=None)

    # ball centers (for segment coords)
    balls = []
    for c,b,bb,cf in zip(cls_ids, xywh, xyxy, confs):
        x,y,w,h = map(float, b)
        if   c == CLASS_BALL     and feat['ball_x']    is None:
            feat.update(ball_x=x,ball_y=y,ball_w=w,ball_h=h,ball_conf=float(cf))
            balls.append(b[:2])  # center
            if FLAGS['SHOW_BALL_TRAIL']:
                ball_trail.append((int(x), int(y)))
        elif c == CLASS_THROWER  and feat['thrower_x'] is None:
            feat.update(thrower_x=x,thrower_y=y,thrower_w=w,thrower_h=h,thrower_conf=float(cf))
        elif c == CLASS_DEFENDER and feat['defender_x']is None:
            feat.update(defender_x=x,defender_y=y,defender_w=w,defender_h=h,defender_conf=float(cf))
    frame_features.append(feat)

    # --------- segment builder with suppression ----------
    if label is not None and balls:
        new_label = label

        # Suppress immediate FROM on the same side right after a TO
        if new_label == CLASS_THROWER:
            side_now = thrower_side_from_feat(feat, frame_h)
            if (side_now is not None and
                last_to_side is not None and
                side_now == last_to_side and
                (frame_idx - last_to_end_frame) <= CFG['SUPPRESS_AFTER_TO_WINDOW_FRAMES']):
                pass  # ignore transient FROM
            else:
                gap = 0
                if cur['label'] is None:
                    cur = {'label': new_label, 'coords': [], 'frames': []}
                if cur['label'] == new_label:
                    cur['coords'].append(balls[0]); cur['frames'].append(frame_idx)
                else:
                    seg_type = 'from' if cur['label'] == CLASS_THROWER else 'to'
                    if cur['frames']:
                        if seg_type == 'to':
                            last_to_end_frame = cur['frames'][-1]
                            to_pt = compute_mean_coord(cur['coords'], 'to', n=CFG['N_MEAN_TO'])
                            last_to_side = 'lower' if to_pt[1] > frame_h/2 else 'upper'
                        seq.append({'type': seg_type, 'coords': cur['coords'], 'frames': cur['frames']})
                    cur = {'label': new_label, 'coords': [balls[0]], 'frames': [frame_idx]}
        else:
            # defender (TO) or both players -> proceed normally
            gap = 0
            if cur['label'] is None:
                cur = {'label': new_label, 'coords': [], 'frames': []}
            if cur['label'] == new_label:
                cur['coords'].append(balls[0]); cur['frames'].append(frame_idx)
            else:
                seg_type = 'from' if cur['label'] == CLASS_THROWER else 'to'
                if cur['frames']:
                    if seg_type == 'to':
                        last_to_end_frame = cur['frames'][-1]
                        to_pt = compute_mean_coord(cur['coords'], 'to', n=CFG['N_MEAN_TO'])
                        last_to_side = 'lower' if to_pt[1] > frame_h/2 else 'upper'
                    seq.append({'type': seg_type, 'coords': cur['coords'], 'frames': cur['frames']})
                cur = {'label': new_label, 'coords': [balls[0]], 'frames': [frame_idx]}
    else:
        # no players or no ball -> maybe close current after a gap
        if cur['label'] is not None:
            gap += 1
            if gap > CFG['MAX_GAP']:
                seg_type = 'from' if cur['label'] == CLASS_THROWER else 'to'
                if cur['frames']:
                    if seg_type == 'to':
                        last_to_end_frame = cur['frames'][-1]
                        to_pt = compute_mean_coord(cur['coords'], 'to', n=CFG['N_MEAN_TO'])
                        last_to_side = 'lower' if to_pt[1] > frame_h/2 else 'upper'
                    seq.append({'type': seg_type, 'coords': cur['coords'], 'frames': cur['frames']})
                cur = {'label': None, 'coords': [], 'frames': []}
                gap = 0
    # ----------------------------------------------------

    # write raw frame for later review/seek
    writer.write(frame)

    # ------- live overlay -------
    ov = frame.copy()

    # draw detections
    if FLAGS['SHOW_DETECTIONS'] and len(cls_ids) > 0:
        draw_detections(ov, xyxy, cls_ids, confs)

    # draw goal polys, zones, numbers
    if FLAGS['SHOW_ZONES']:
        cv2.polylines(ov, [lower_poly], True, (0, 180, 0), UI_ZONE_LINE_THICKNESS)
        cv2.polylines(ov, [upper_poly], True, (0, 180, 0), UI_ZONE_LINE_THICKNESS)
        for k, pos in enumerate(low_bar, 1):
            x, y = map(int, pos); draw_text(ov, str(k), (x-10, y+18), 0.6, (180,0,180), UI_THICKNESS)
        cv2.line(ov, tuple(low_rt), tuple(low_lt), (180, 0, 180), UI_ZONE_LINE_THICKNESS)
        for k, pos in enumerate(up_bar, 1):
            x, y = map(int, pos); draw_text(ov, str(k), (x-10, y-8), 0.6, (180,0,180), UI_THICKNESS)
        cv2.line(ov, tuple(up_lt), tuple(up_rt), (180, 0, 180), UI_ZONE_LINE_THICKNESS)

    # ball trail + current ball dot
    if FLAGS['SHOW_BALL_TRAIL'] and len(ball_trail) > 1:
        for i in range(1, len(ball_trail)):
            cv2.line(ov, ball_trail[i-1], ball_trail[i], (220,220,0), 2)
    if feat['ball_x'] is not None:
        bx, by = int(feat['ball_x']), int(feat['ball_y'])
        cv2.circle(ov, (bx, by), 6, (220,220,0), -1)

    # HUD
    dt = time.time() - t_infer0
    fps = 1.0 / max(dt, 1e-6)
    fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps
    seg_state = "none" if cur['label'] is None else ("FROM" if cur['label']==CLASS_THROWER else "TO")
    draw_text(ov, f"FPS {fps_smooth:4.1f} | Segment: {seg_state}", (10, 26), 0.65, (255,255,255), UI_THICKNESS)
    draw_text(ov, f"Lower: {lower_team} | Upper: {upper_team}", (10, 50), 0.65, (0,255,255), UI_THICKNESS)
    draw_text(ov, "H=end half  Q=abort  D=det  Z=zones  B=trail", (10, frame_h - 16), 0.6, (255,255,0), UI_THICKNESS)

    cv2.imshow("Live", ov)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        print("[INFO] Aborted by user (Q).")
        break
    if key in (ord('h'), ord('H')):
        print("[INFO] Half ended by user (H).")
        break
    if key in (ord('d'), ord('D')):
        FLAGS['SHOW_DETECTIONS'] = not FLAGS['SHOW_DETECTIONS']
    if key in (ord('z'), ord('Z')):
        FLAGS['SHOW_ZONES'] = not FLAGS['SHOW_ZONES']
    if key in (ord('b'), ord('B')):
        FLAGS['SHOW_BALL_TRAIL'] = not FLAGS['SHOW_BALL_TRAIL']

# close any open current segment
if cur['label'] is not None and cur['frames']:
    seg_type = 'from' if cur['label'] == CLASS_THROWER else 'to'
    if seg_type == 'to':
        last_to_end_frame = cur['frames'][-1]
        to_pt = compute_mean_coord(cur['coords'], 'to', n=CFG['N_MEAN_TO'])
        last_to_side = 'lower' if to_pt[1] > frame_h/2 else 'upper'
    seq.append({'type': seg_type, 'coords': cur['coords'], 'frames': cur['frames']})

# Done capturing
cap.release()
writer.release()
cv2.destroyWindow("Live")

# ==========================
# ====== POSTPROCESS =======
# ==========================
# collapse consecutive same-type segments
collapsed, grp = [], None
for seg in seq:
    if grp is None or grp['type'] != seg['type']:
        grp = {'type': seg['type'], 'coords': seg['coords'][:], 'frames': seg['frames'][:]}
        collapsed.append(grp)
    else:
        grp['coords'].extend(seg['coords'])
        grp['frames'].extend(seg['frames'])

# extend each "to" by N frames (bounded by last captured frame)
max_frames = frame_idx
for seg in collapsed:
    if seg['type'] == 'to':
        last = seg['frames'][-1]
        seg['frames'].extend(range(last + 1, min(last + 1 + CFG['EXTEND_TO_FRAMES'], max_frames)))

# ---------- NEW: pair-level de-dup BEFORE review ----------
pairs = []
pending_from_idx = None
for idx, seg in enumerate(collapsed):
    if seg['type'] == 'from':
        pending_from_idx = idx
    elif seg['type'] == 'to' and pending_from_idx is not None:
        fseg = collapsed[pending_from_idx]
        tseg = seg
        start = fseg['frames'][0]
        end   = tseg['frames'][-1]
        # coords and sides
        fc = compute_mean_coord(fseg['coords'], 'from', n=CFG['N_MEAN_FROM'])
        thr_xy = first_thrower_center(fseg, frame_features, n=1)
        from_pt = thr_xy if not np.isnan(thr_xy[0]) else fc
        from_side = which_goal(from_pt, lower_poly, upper_poly, frame_h)

        tc = compute_mean_coord(tseg['coords'], 'to', n=CFG['N_MEAN_TO'])
        to_side = which_goal(tc, lower_poly, upper_poly, frame_h)

        pairs.append(dict(
            from_idx=pending_from_idx, to_idx=idx,
            start=start, end=end,
            from_side=from_side, to_side=to_side
        ))
        pending_from_idx = None

# Keep only non-duplicate pairs
accepted_pairs = []
for p in pairs:
    dup = False
    for q in accepted_pairs:
        same_sides = (p['from_side']==q['from_side']) and (p['to_side']==q['to_side'])
        iou = interval_iou(p['start'], p['end'], q['start'], q['end'])
        close_starts = abs(p['start'] - q['start']) <= CFG['DEDUP_MAX_START_DIFF']
        close_ends   = abs(p['end']   - q['end'])   <= CFG['DEDUP_MAX_END_DIFF']
        if same_sides and (iou >= CFG['DEDUP_TIME_IOU'] or (close_starts and close_ends)):
            dup = True
            break
    if not dup:
        accepted_pairs.append(p)

# Build keep set of segment indices (from accepted pairs) + keep orphans
keep_from = {p['from_idx'] for p in accepted_pairs}
keep_to   = {p['to_idx']   for p in accepted_pairs}
paired_from = {p['from_idx'] for p in pairs}
paired_to   = {p['to_idx']   for p in pairs}

collapsed_filtered = []
for idx, seg in enumerate(collapsed):
    if idx in keep_from or idx in keep_to:
        collapsed_filtered.append(seg)
    else:
        # keep orphan FROM (not paired at all); drop duplicate pairs
        if seg['type']=='from' and (idx not in paired_from):
            collapsed_filtered.append(seg)

collapsed = collapsed_filtered
# ---------------------------------------------------------

# re-map identities to stable ids post-filter
segment_id_of = {id(seg): idx + 1 for idx, seg in enumerate(collapsed)}

# ==========================
# ======== REVIEW UI =======
# ==========================
print("\n--- REVIEW / LABEL ---")
print("Controls: SPACE=play/pause | ←/→=step (paused) | N=next seg | q=stop current seg | Esc=abort all")
print("For each segment: press '1' (real throw) or '0' if false positive.")
print("For 'to' segments marked as real, press 'g' goal / 'b' block / 'o' out.")

training_records = []
seg_cap = cv2.VideoCapture(str(TMP_VIDEO_PATH))
if not seg_cap.isOpened():
    raise RuntimeError("Failed to open temporary recording for review.")

ensure_window("Label Segment", 1280, 720)
delay_ms = max(1, int(1000.0 / max(1.0, CFG['TARGET_FPS'])))

def show_frame_at(f_idx, overlay_fn):
    seg_cap.set(cv2.CAP_PROP_POS_FRAMES, max(f_idx - 1, 0))
    ret, img = seg_cap.read()
    if not ret:
        return False
    ov = img.copy()
    overlay_fn(ov)
    cv2.imshow("Label Segment", ov)
    return True

for i, seg in enumerate(collapsed):
    start, end = seg['frames'][0], seg['frames'][-1]
    seg_type = seg['type']
    print(f"\nSegment {i+1}/{len(collapsed)} — {seg_type} — frames {start}→{end}")

    # Prepare overlay function (zones + per-frame detections)
    def overlay(ov_img, f=None):
        cv2.polylines(ov_img, [lower_poly], True, (0, 180, 0), UI_ZONE_LINE_THICKNESS)
        cv2.polylines(ov_img, [upper_poly], True, (0, 180, 0), UI_ZONE_LINE_THICKNESS)
        for k, pos in enumerate(low_bar, 1):
            x, y = map(int, pos); draw_text(ov_img, str(k), (x-10, y+18), 0.6, (180,0,180), UI_THICKNESS)
        cv2.line(ov_img, tuple(low_rt), tuple(low_lt), (180, 0, 180), UI_ZONE_LINE_THICKNESS)
        for k, pos in enumerate(up_bar, 1):
            x, y = map(int, pos); draw_text(ov_img, str(k), (x-10, y-8), 0.6, (180,0,180), UI_THICKNESS)
        cv2.line(ov_img, tuple(up_lt), tuple(up_rt), (180, 0, 180), UI_ZONE_LINE_THICKNESS)

        if f is not None and 0 < f <= len(frame_features):
            ff = frame_features[f - 1]
            if ff['ball_x'] is not None:
                cx, cy = int(ff['ball_x']), int(ff['ball_y'])
                cv2.circle(ov_img, (cx, cy), 6, (220,220,0), -1)
                draw_text(ov_img, "ball", (cx+8, cy+8), UI_SMALL_FONT, (220,220,0), UI_THICKNESS)
            if ff['thrower_x'] is not None:
                x,y,w,h = ff['thrower_x'], ff['thrower_y'], ff['thrower_w'], ff['thrower_h']
                tl, br = (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2))
                cv2.rectangle(ov_img, tl, br, (0,200,0), UI_BOX_THICKNESS)
                draw_text(ov_img, "thrower", (tl[0], max(0, tl[1]-8)), UI_SMALL_FONT, (0,200,0), UI_THICKNESS)
            if ff['defender_x'] is not None:
                x,y,w,h = ff['defender_x'], ff['defender_y'], ff['defender_w'], ff['defender_h']
                tl, br = (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2))
                cv2.rectangle(ov_img, tl, br, (0,0,200), UI_BOX_THICKNESS)
                draw_text(ov_img, "defender", (tl[0], max(0, tl[1]-8)), UI_SMALL_FONT, (0,0,200), UI_THICKNESS)

        big_label = "THROW FROM" if seg_type == "from" else "THROW TO"
        draw_text(ov_img, f"Segment {i+1}/{len(collapsed)}: {seg_type} ({start}-{end})", (10, 24), 0.65, (255,255,255), UI_THICKNESS)
        draw_text(ov_img, big_label, (10, 48), 0.8, (0,255,255), UI_THICKNESS)
        draw_text(ov_img, f"{lower_team} (Lower) / {upper_team} (Upper)", (10, 72), 0.65, (0,255,255), UI_THICKNESS)

    # --- Playback controls ---
    playing = True
    fpos = 0
    seg_frames = seg['frames'][:]  # local copy of frames in this segment

    while True:
        if playing:
            if fpos >= len(seg_frames):
                break
            f = seg_frames[fpos]
            if not show_frame_at(f, lambda ov_: overlay(ov_, f)):
                break
            key = cv2.waitKey(max(1, int(1000.0 / max(1.0, CFG['TARGET_FPS'])))) & 0xFF
            fpos += 1
        else:
            fpos = max(0, min(fpos, len(seg_frames)-1))
            f = seg_frames[fpos]
            if not show_frame_at(f, lambda ov_: overlay(ov_, f)):
                break
            key = cv2.waitKey(0) & 0xFF

        if key == 27:  # Esc => abort all review
            print("[INFO] Review aborted.")
            seg_cap.release()
            cv2.destroyWindow("Label Segment")
            raise SystemExit(0)
        elif key == ord(' '):            # SPACE toggles play/pause
            playing = not playing
        elif key == ord('n') or key == ord('N'):
            break                         # skip to next segment
        elif key in (ord('q'), ord('Q')):
            break                         # stop current segment playback, go to label
        elif not playing:
            if key == 83:                 # Right arrow
                fpos = min(fpos + 1, len(seg_frames)-1)
            elif key == 81:               # Left arrow
                fpos = max(fpos - 1, 0)

    # ----- segment decision -----
    print("Press '1' if this is a real throw, '0' if false positive.")
    while True:
        k = cv2.waitKey(0)
        if k in (ord('1'), ord('0')):
            seg_label = 1 if k == ord('1') else 0
            break

    outcome = ''
    if seg_label == 1 and seg_type == 'to':
        print("Enter outcome: 'g' for goal, 'b' for block, 'o' for out")
        while True:
            k = cv2.waitKey(0)
            if k in (ord('g'), ord('b'), ord('o')):
                outcome = chr(k)
                break

    # write training records for each frame in the segment
    for f in seg_frames:
        ff = frame_features[f - 1]
        rec = {
            'segment_id': i + 1,
            'segment_type': seg_type,
            'frame': f,
            'label': seg_label,
            'outcome': outcome
        }
        rec.update({k: v for k, v in ff.items() if k not in ('frame', 'label')})
        training_records.append(rec)

seg_cap.release()
cv2.destroyWindow("Label Segment")

# ==========================
# ====== FILTER & SAVE =====
# ==========================
# Pair FROM -> next TO (keep orphan FROM at end)
throws = []
pending_from = None
for seg in collapsed:
    if seg['type'] == 'from':
        pending_from = seg
    elif seg['type'] == 'to':
        throws.append({'num': len(throws)+1, 'from': pending_from, 'to': seg, 'incomplete': False})
        pending_from = None
if pending_from is not None:
    throws.append({'num': len(throws)+1, 'from': pending_from, 'to': None, 'incomplete': True})

# keep only segments marked as correct (and always keep orphan FROM)
seg_labels = {rec['segment_id']: rec['label'] for rec in training_records}
segment_id_of = {id(seg): idx + 1 for idx, seg in enumerate(collapsed)}  # re-map after filter
good_throws = []
for t in throws:
    if t['to'] is None:
        good_throws.append(t)
    else:
        seg_id = segment_id_of[id(t['to'])]
        if seg_labels.get(seg_id, 1) == 1:
            good_throws.append(t)
throws = good_throws

# ==========================
# ===== RESULTS TABLE  =====
# ==========================
results = []
for t in throws:
    num = t['num']

    # ---------- FROM point & side ----------
    fc = compute_mean_coord(t['from']['coords'], 'from', n=CFG['N_MEAN_FROM'])
    # prefer detected thrower; fallback to ball cluster
    thr_xy = first_thrower_center(t['from'], frame_features, n=1)
    from_pt = thr_xy if not np.isnan(thr_xy[0]) else fc
    from_side = which_goal(from_pt, lower_poly, upper_poly, frame_h)  # 'lower' or 'upper'

    # teams by side (fixed this half)
    throwing  = lower_team if from_side == 'lower' else upper_team
    defending = upper_team if from_side == 'lower' else lower_team

    # FROM zone by side
    if from_side == 'lower':
        fz = compute_zone(fc, low_rt, low_lt); fz_label = f"{fz} Lower"
    else:
        fz = compute_zone(fc, up_lt, up_rt);   fz_label = f"{fz} Upper"

    # ---------- TO point & side ----------
    if t['to'] is None:
        tc = np.array([np.nan, np.nan], dtype=float)
        tz_label = np.nan
    else:
        tc = compute_mean_coord(t['to']['coords'], 'to', n=CFG['N_MEAN_TO'])
        to_side = which_goal(tc, lower_poly, upper_poly, frame_h)
        if to_side == 'lower':
            tz = compute_zone(tc, low_rt, low_lt); tz_label = f"{tz} Lower"
        else:
            tz = compute_zone(tc, up_lt, up_rt);   tz_label = f"{tz} Upper"

    results.append({
        'Throw Number':   num,
        'From Zone':      fz_label,
        'To Zone':        tz_label,
        'Throwing Team':  throwing,
        'Defending Team': defending,
        'From Coord':     fc.tolist(),
        'To Coord':       tc.tolist()
    })

# Save outputs
pd.DataFrame(training_records).to_csv(LSTM_CSV_PATH, index=False)

# Excel with graceful fallback if openpyxl missing in this interpreter
try:
    pd.DataFrame(results).to_excel(THROWS_XLSX_PATH, index=False)
    print("\n=== DONE ===")
    print(f"Saved LSTM training CSV → {LSTM_CSV_PATH}")
    print(f"Saved Throws Excel      → {THROWS_XLSX_PATH}")
except ModuleNotFoundError:
    pd.DataFrame(results).to_csv(THROWS_CSV_FALLBACK, index=False)
    print("\n[WARN] openpyxl is not available in this Python interpreter.")
    print("       Wrote CSV fallback instead.")
    print(f"Saved LSTM training CSV → {LSTM_CSV_PATH}")
    print(f"Saved Throws CSV        → {THROWS_CSV_FALLBACK}")
    print("Tip: run this script with the same Python where you installed openpyxl,")
    print("     or `pip install openpyxl` in THIS interpreter.")
