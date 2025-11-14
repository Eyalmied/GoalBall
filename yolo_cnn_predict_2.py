#!pip install ultralytics opencv-python pandas numpy openpyxl

from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import os
import time

game = "TUR_-_BRA_3-1"
# — your model & file paths —
model = YOLO('Thesis/ball+players_tuning10/weights/best.pt')
video_path = f'Thesis/Paralympics2024/{game}/{game}.MOV'
excel_output_path = f'Thesis/Paralympics2024/{game}/outputs/{game}_Throws_data.xlsx'
lstm_csv_path   = f'Thesis/Paralympics2024/{game}/outputs/{game}_Throws_lstm_training.csv'

os.makedirs(os.path.dirname(excel_output_path), exist_ok=True)
os.makedirs(os.path.dirname(lstm_csv_path), exist_ok=True)

# — grab one frame for calibration clicks —
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video_path}")
ret, calib = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to read calibration frame")

cv2.namedWindow("Calibration Frame", cv2.WINDOW_NORMAL)
cv2.imshow("Calibration Frame", calib)
print("Press any key once you have reviewed the calibration frame...")
cv2.waitKey(0)
cv2.destroyWindow("Calibration Frame")

# ===== User input for team names =====
lower_team = input("Enter Lower Team name (e.g., ISR): ")
upper_team = input("Enter Upper Team name (e.g., CAN): ")

# ===== User input for halftime time =====
halftime_str = input("Enter halftime time (hh:mm:ss or mm:ss): ") 
parts = halftime_str.split(':')
if len(parts) == 2:
    hours = 0
    minutes, seconds = map(int, parts)
elif len(parts) == 3:
    hours, minutes, seconds = map(int, parts)
else:
    raise ValueError("Invalid time format. Please use hh:mm:ss or mm:ss")

total_seconds = hours * 3600 + minutes * 60 + seconds
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"[INFO] Detected FPS: {fps:.2f}")

halftime_frame = int(total_seconds * fps)
print(f"Halftime frame (based on {total_seconds}s and {fps:.2f} fps): {halftime_frame}")

# — click 8 goal-frame corners —
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
        cv2.circle(calib, (x, y), 5, (0,0,255), -1)
        cv2.putText(calib, labels[len(pts)-1], (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imshow("Calibrate", calib)

cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
cv2.imshow("Calibrate", calib)
cv2.setMouseCallback("Calibrate", on_click)
print("Now click in order the following points:")
for l in labels: print(" ", l)
while len(pts) < 8:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# unpack & define
p1,p2,p3,p4,p5,p6,p7,p8 = [np.array(p) for p in pts]
low_rt, low_lt = p2, p4
up_lt,  up_rt  = p6, p8
lower_poly = np.array([p1,p2,p4,p3], dtype=np.int32)
upper_poly = np.array([p5,p6,p8,p7], dtype=np.int32)

def which_goal(pt):
    if cv2.pointPolygonTest(lower_poly, tuple(pt), False) >= 0: return 'lower'
    if cv2.pointPolygonTest(upper_poly, tuple(pt), False) >= 0: return 'upper'
    return 'lower' if pt[1] > calib.shape[0]/2 else 'upper'

def compute_zone(P,A,B):
    P,A,B = np.array(P), np.array(A), np.array(B)
    u = np.dot(P-A, B-A) / (np.dot(B-A, B-A)+1e-6)
    u = float(np.clip(u,0,1))
    return int(u*9)+1

def compute_mean_coord(coords, segment_type, n=10):
    if not coords: return np.array([np.nan, np.nan])
    sel = coords[:n] if segment_type=='from' else coords[-n:]
    return np.mean(sel, axis=0)

def first_thrower_center(segment, n=1):
    """
    Return the centroid (x, y) of the thrower in the earliest `n`
    frames of a *from* segment.
    If none found, returns (np.nan, np.nan).
    """
    centres = []
    for f in segment['frames'][:n]:
        ff = frame_features[f - 1]          # frame_features already filled later
        if ff['thrower_x'] is not None:     # class-0 box exists on that frame
            centres.append((ff['thrower_x'], ff['thrower_y']))
    return np.mean(centres, axis=0) if centres else (np.nan, np.nan)


# — detection & feature extraction —
cap = cv2.VideoCapture(video_path)
seq, cur = [], {'label':None,'coords':[],'frames':[]}
gap,max_gap,fn = 0,2,0
frame_features = []

while True:
    ret, frame = cap.read()
    if not ret: break
    fn += 1
    res = model.predict(frame, save=False, conf=0.7, verbose=False)[0]
    cls  = res.boxes.cls.cpu().numpy()
    xy   = res.boxes.xywh.cpu().numpy()
    confs= res.boxes.conf.cpu().numpy()

    players = [c for c in cls if c in (0,1)]
    label   = 1 if len(players)>1 else (players[0] if players else None)

    feat = dict(frame=fn, label=label,
                ball_x=None,ball_y=None,ball_w=None,ball_h=None,ball_conf=None,
                thrower_x=None,thrower_y=None,thrower_w=None,thrower_h=None,thrower_conf=None,
                defender_x=None,defender_y=None,defender_w=None,defender_h=None,defender_conf=None)
    for c,b,cf in zip(cls,xy,confs):
        x,y,w,h = map(float,b)
        if   c==32 and feat['ball_x']   is None: feat.update(ball_x=x,ball_y=y,ball_w=w,ball_h=h,ball_conf=float(cf))
        elif c==0  and feat['thrower_x']is None: feat.update(thrower_x=x,thrower_y=y,thrower_w=w,thrower_h=h,thrower_conf=float(cf))
        elif c==1  and feat['defender_x']is None: feat.update(defender_x=x,defender_y=y,defender_w=w,defender_h=h,defender_conf=float(cf))
    frame_features.append(feat)

    balls = [b[:2] for c,b in zip(cls,xy) if c==32]
    if label is not None and balls:
        gap=0
        if cur['label'] is None:
            cur={'label':label,'coords':[],'frames':[]}
        if cur['label']==label:
            cur['coords'].append(balls[0]); cur['frames'].append(fn)
        else:
            seq.append({'type':'from' if cur['label']==0 else 'to',
                        'coords':cur['coords'],'frames':cur['frames']})
            cur={'label':label,'coords':[balls[0]],'frames':[fn]}
    else:
        if cur['label'] is not None:
            gap+=1
            if gap>max_gap:
                seq.append({'type':'from' if cur['label']==0 else 'to',
                            'coords':cur['coords'],'frames':cur['frames']})
                cur={'label':None,'coords':[],'frames':[]}
                gap=0

cap.release()
if cur['label'] is not None and cur['frames']:
    seq.append({'type':'from' if cur['label']==0 else 'to','coords':cur['coords'],'frames':cur['frames']})

# — collapse & extend —
collapsed, grp = [], None
for seg in seq:
    if grp is None or grp['type']!=seg['type']:
        grp={'type':seg['type'],'coords':seg['coords'][:],'frames':seg['frames'][:]}
        collapsed.append(grp)
    else:
        grp['coords'].extend(seg['coords'])
        grp['frames'].extend(seg['frames'])

cap = cv2.VideoCapture(video_path)
max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
for seg in collapsed:
    if seg['type']=='to':
        last=seg['frames'][-1]
        seg['frames'].extend(range(last + 1, min(last + 81, max_frames)))
# === map every segment object to its 1-based ID (used later) ===
segment_id_of = {id(seg): idx + 1 for idx, seg in enumerate(collapsed)}

# — interactive labeling —
training_records=[]
seg_cap=cv2.VideoCapture(video_path)
cv2.namedWindow("Label Segment",cv2.WINDOW_NORMAL)


for i,seg in enumerate(collapsed):
    start,end=seg['frames'][0],seg['frames'][-1]
    print(f"\nSegment {i+1}/{len(collapsed)} — {seg['type']} — frames {start}→{end}")
    for f in seg['frames']:
        seg_cap.set(cv2.CAP_PROP_POS_FRAMES,f)
        ret,img=seg_cap.read()
        if not ret: continue
        ov,ff=img.copy(),frame_features[f-1]
        if ff['ball_x']  is not None:
            cx,cy=int(ff['ball_x']),int(ff['ball_y'])
            cv2.circle(ov,(cx,cy),8,(255,255,0),-1)
            cv2.putText(ov,"ball",(cx+10,cy+10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
        if ff['thrower_x']is not None:
            x,y,w,h=ff['thrower_x'],ff['thrower_y'],ff['thrower_w'],ff['thrower_h']
            tl,br=(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2))
            cv2.rectangle(ov,tl,br,(0,255,0),2)
            cv2.putText(ov,"thrower",(tl[0],tl[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        if ff['defender_x']is not None:
            x,y,w,h=ff['defender_x'],ff['defender_y'],ff['defender_w'],ff['defender_h']
            tl,br=(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2))
            cv2.rectangle(ov,tl,br,(0,0,255),2)
            cv2.putText(ov,"defender",(tl[0],tl[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        info=f"Segment {i+1}/{len(collapsed)}: {seg['type']} ({start}-{end})"
        cv2.putText(ov,info,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        big_label="THROW FROM" if seg['type']=="from" else "THROW TO"
        end_fr=seg['frames'][-1]
        tl,tu = (upper_team,lower_team) if end_fr>halftime_frame else (lower_team,upper_team)
        cv2.putText(ov,big_label,(10,70),cv2.FONT_HERSHEY_DUPLEX,1.2,(0,255,255),3)
        cv2.putText(ov,f"{tl} / {tu}",(10,110),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),2)
        cv2.imshow("Label Segment",ov)
        if cv2.waitKey(50)&0xFF==ord('q'): break

    print("Press '1' if correct throw, '0' if false positive")
    while True:
        k=cv2.waitKey(0)
        if k in (ord('1'),ord('0')):
            seg_label = 1 if k==ord('1') else 0
            break

    outcome=''
    if seg_label==1 and seg['type']=='to':
        print("Enter outcome: 'g' for goal, 'b' for block, 'o' for out")
        while True:
            k=cv2.waitKey(0)
            if k in (ord('g'),ord('b'),ord('o')):
                outcome=chr(k); break

    for f in seg['frames']:
        ff=frame_features[f-1]
        rec={'segment_id':i+1,'segment_type':seg['type'],'frame':f,'label':seg_label,'outcome':outcome}
        rec.update({k:v for k,v in ff.items() if k not in ('frame','label')})
        training_records.append(rec)

seg_cap.release()
cv2.destroyAllWindows()

# save LSTM CSV
pd.DataFrame(training_records).to_csv(lstm_csv_path,index=False)
print(f"Saved LSTM training CSV → {lstm_csv_path}")

# — build & visualize throws, then save Excel —

cap = cv2.VideoCapture(video_path)
# throws = []
current_lower, current_upper = lower_team, upper_team
#
# # ————— REPLACED PAIRING LOGIC —————
# # find each 'from' and pair it with the very next 'to'
# i = 0
# while i < len(collapsed):
#     if collapsed[i]['type'] == 'from':
#         j = i + 1
#         while j < len(collapsed) and collapsed[j]['type'] != 'to':
#             j += 1
#         if j <= len(collapsed):
#             throws.append({
#                 'num': len(throws) + 1,
#                 'from': collapsed[i],
#                 'to':   collapsed[j]
#             })
#             i = j + 1
#             continue
#     i += 10

# ---------- robust FROM-TO pairing (keeps orphans) ----------
throws = []
pending_from = None                                 # last 'from' we saw

for seg in collapsed:
    if seg['type'] == 'from':
        pending_from = seg

    elif seg['type'] == 'to':
        # normal paired throw
        throws.append({
            'num': len(throws) + 1,
            'from': pending_from,
            'to':   seg,
            'incomplete': False
        })
        pending_from = None                         # reset for next pair

# video finished — did we end on a lone 'from'?
if pending_from is not None:
    throws.append({
        'num': len(throws) + 1,
        'from': pending_from,
        'to':   None,                               # the missing “to”
        'incomplete': True
    })
# -----------------------------------------------------------



# filter by user labels
# --- keep only segments user marked as correct (or orphan FROMs) ---
seg_labels = {rec['segment_id']: rec['label'] for rec in training_records}

good_throws = []
for t in throws:
    if t['to'] is None:  # orphaned FROM → always keep
        good_throws.append(t)
    else:
        seg_id = segment_id_of[id(t['to'])]  # lookup by identity
        if seg_labels.get(seg_id, 1) == 1:   # keep only if user pressed “1”
            good_throws.append(t)

throws = good_throws

# --- precompute zone tick positions along the two goal lines ---
low_bar = [low_rt + (low_lt - low_rt) * ((k + 0.5) / 9) for k in range(9)]
up_bar  = [up_lt  + (up_rt  - up_lt)  * ((k + 0.5) / 9) for k in range(9)]

results = []
cap = cv2.VideoCapture(video_path)
current_lower, current_upper = lower_team, upper_team  # start as entered

for t in throws:
    num = t['num']

    # ---------- FROM side (always exists) ----------
    fc = compute_mean_coord(t['from']['coords'], 'from', n=5)

    # prefer detected thrower; fallback to ball cluster
    thrower_xy = first_thrower_center(t['from'])
    side = which_goal(thrower_xy) if not np.isnan(thrower_xy[0]) else which_goal(fc)   # 'lower' or 'upper'

    # ---------- half-time swap of TEAM NAMES on sides ----------
    end_fr = t['to']['frames'][-1] if t['to'] is not None else t['from']['frames'][-1]
    if end_fr > halftime_frame:
        current_lower, current_upper = upper_team, lower_team
    else:
        current_lower, current_upper = lower_team, upper_team

    # ---------- teams by side (not by constants) ----------
    throwing  = current_lower if side == 'lower' else current_upper
    defending = current_upper if side == 'lower' else current_lower

    # ---------- FROM zone by side ----------
    if side == 'lower':
        fz = compute_zone(fc, low_rt, low_lt)
        fz_label = f"{fz} Lower"
    else:
        fz = compute_zone(fc, up_lt, up_rt)
        fz_label = f"{fz} Upper"

    # ---------- TO side (may be missing) ----------
    if t['to'] is None:  # orphaned throw
        tc        = np.array([np.nan, np.nan], dtype=float)
        tz_label  = np.nan
        complete  = False
    else:
        tc = compute_mean_coord(t['to']['coords'], 'to', n=5)
        defending_side = 'upper' if side == 'lower' else 'lower'
        if defending_side == 'lower':
            tz = compute_zone(tc, low_rt, low_lt)
            tz_label = f"{tz} Lower"
        else:
            tz = compute_zone(tc, up_lt, up_rt)
            tz_label = f"{tz} Upper"
        complete = True

    # ---------- visualisation ----------
    if complete:  # (delete this line to also show orphans)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(end_fr - 1, 0))
        ret, img = cap.read()
        if ret:
            # 1) goal rectangles (green)
            cv2.polylines(img, [lower_poly], True, (0, 255, 0), 2)
            cv2.polylines(img, [upper_poly], True, (0, 255, 0), 2)

            # 2) magenta zone bars & numbers
            for k, pos in enumerate(low_bar, 1):
                x, y = map(int, pos)
                cv2.putText(img, str(k), (x - 12, y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)
            cv2.line(img, tuple(low_rt), tuple(low_lt), (255, 0, 255), 3)

            for k, pos in enumerate(up_bar, 1):
                x, y = map(int, pos)
                cv2.putText(img, str(k), (x - 12, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)
            cv2.line(img, tuple(up_lt), tuple(up_rt), (255, 0, 255), 3)

            # 3) ball positions & labels
            fx, fy = map(int, fc)
            cv2.circle(img, (fx, fy), 10, (0, 255, 0), -1)
            cv2.putText(img, f"Throw {num} from {fz_label}", (fx + 5, fy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if not np.isnan(tc).any():  # only if TO exists
                tx, ty = map(int, tc)
                cv2.circle(img, (tx, ty), 10, (0, 0, 255), -1)
                cv2.putText(img, f"to {tz_label}", (tx + 5, ty - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(img, f"Defending: {defending}", (tx + 5, ty - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(img, f"Throwing: {throwing}", (fx + 5, fy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # show frame
            cv2.imshow(f"Throw {num}", img)
            print(f"Displaying Throw {num}. Press any key to continue …")
            cv2.waitKey(0)
            cv2.destroyWindow(f"Throw {num}")

    # ---------- store result ----------
    results.append({
        'Throw Number':   num,
        'From Zone':      fz_label,
        'To Zone':        tz_label,
        'Throwing Team':  throwing,
        'Defending Team': defending,
        'From Coord':     fc.tolist(),
        'To Coord':       tc.tolist()
    })


cap.release()

pd.DataFrame(results).to_excel(excel_output_path, index=False)
print(f"Saved Throws Excel → {excel_output_path}")

