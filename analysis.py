import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rtmlib import Wholebody, draw_skeleton
import json
import time


video_path = "/home/topiary-pc/Projects/MovesMethod/Example1.mov"
video_loader = cv2.VideoCapture(video_path)

model = Wholebody(mode="lightweight", backend="onnxruntime", device="cuda")

SHOW = False            # True: draw + show every frame in a window during the first pass (slower). q quits.
PROGRESS_EVERY = 100    # print a progress line every N frames so a headless run does not look stuck

n_frame = 0
TIMES = dict(detect=0.0, pose=0.0, geometry=0.0, draw=0.0)   # accumulated seconds per stage
t_pass1 = time.perf_counter()


def detect_facing_position(k):
    '''
    We'll use the ratio between the torso's length (mid shoulder point to mid hip point) and the shoulder's width to identify whether the person is facing front or not. 
    And we're using the toe's direction in relative to the heels for left vs right.
    '''

    shoulder_midpoint = (k[6] + k[5])/2 
    hip_midpoint = (k[12] + k[11])/2 

    torso_height = np.linalg.norm(shoulder_midpoint - hip_midpoint)

    shoulder_width = abs(k[5][0] - k[6][0])

    ratio = shoulder_width/torso_height

    if ratio > 0.45:
        if k[5][0] > k[6][0]:
            return "FRONT"
        else:
            return "BACK"
    
    toe_dx = ((k[17][0] - k[19][0]) + (k[20][0] - k[22][0])) / 2
    return "RIGHT" if toe_dx < 0 else "LEFT"
        
    

def angle_between(u, v):
    '''
    Just a simple dot product angle formula.
    '''
    c = np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return angle


def calc_shoulders_angle(hip_point, shoulder_point, elbow_point):
    '''
    Receives the coordinates for the hip, shoulder, and the elbow.
    Calculates the angle of the shoulder with respect to the other two points and returns in degrees.
    '''

    vec_hip = hip_point - shoulder_point
    vec_elbow = elbow_point - shoulder_point

    shoulder_flex = angle_between(vec_hip, vec_elbow)

    return shoulder_flex



def calc_hips_angle(hip_point, knee_point, shoulder_point):
    '''
    Receives the coord for the hip, knee and the shoulder.
    Calculates the angle of the hip with respect to knee and shoulder.
    '''

    vec_knee = knee_point - hip_point
    vec_shoulder = shoulder_point - hip_point

    hips_flex = angle_between(vec_knee, vec_shoulder)

    return hips_flex


def calc_knee_angle(knee_point, hip_point, ankle_point):

    vec_ankle = ankle_point - knee_point
    vec_hip = hip_point - knee_point

    knee_flex = angle_between(vec_ankle, vec_hip)

    return knee_flex


def calc_ankle_angle(knee_point, ankle_point, heel_point, big_toe_point, small_toe_point):
    '''
    Receives the coords for the knee, ankle, heel, big toe and small toe.
    Returns (raw_angle, foot_pitch) in degrees.
    '''

    vec_knee = knee_point - ankle_point
    toe_mid = (big_toe_point + small_toe_point)/2
    vec_foot = toe_mid - heel_point

    ankle_raw = angle_between(vec_foot, vec_knee)
    foot_pitch = np.degrees(np.arctan2(vec_foot[1], abs(vec_foot[0])))    # heel->toe dy > 0 in image coords when the heel is raised; abs(dx) makes it facing-independent

    return ankle_raw, foot_pitch


# Raw joint angles -> clinical convention. One table per exercise; the angle functions stay generic.
# Overhead squat: shoulder flexion reads directly (arm down = 0), hip/knee flexion close from straight (180),
# ankle dorsiflexion is measured from the foot perpendicular to the shin (90).
OVERHEAD_SQUAT = {
    "shoulder": lambda a: a,
    "hip":      lambda a: 180 - a,
    "knee":     lambda a: 180 - a,
    "ankle":    lambda a: 90 - a,
}

SIDE = {"RIGHT": dict(shoulder=5, elbow=7, hip=11, knee=13, ankle=15, big_toe=17, small_toe=18, heel=19),
        "LEFT": dict(shoulder=6, elbow=8, hip=12, knee=14, ankle=16, big_toe=20, small_toe=21, heel=22)}

def score_overhead_squat(k, s, view, conv=OVERHEAD_SQUAT, thr=0.3):
    '''
    Scores the overhead squat when the person is facing left or right only.
    Also the keypoint score should be above the threshold.
    '''
    if view not in SIDE:
        return None

    i = SIDE[view]
    ok = lambda *idx: all(s[j] > thr for j in idx)
    P = lambda name: k[i[name]]

    out = dict(view=view)
    out["shoulder"] = conv["shoulder"](calc_shoulders_angle(P("hip"), P("shoulder"), P("elbow"))) if ok(i["hip"], i["shoulder"], i["elbow"]) else np.nan
    out["hip"] = conv["hip"](calc_hips_angle(P('hip'), P('knee'), P("shoulder"))) if ok(i["shoulder"], i["hip"], i["knee"]) else np.nan
    out["knee"] = conv["knee"](calc_knee_angle(P("knee"), P("hip"), P("ankle"))) if ok(i['hip'], i['knee'], i['ankle']) else np.nan
    if ok(i["knee"], i["ankle"], i["heel"], i["big_toe"], i["small_toe"]):
        ankle_raw, out["foot_pitch"] = calc_ankle_angle(P("knee"), P("ankle"), P("heel"), P("big_toe"), P("small_toe"))
        out["ankle"] = conv["ankle"](ankle_raw)
    else:
        out["ankle"], out["foot_pitch"] = np.nan, np.nan

    return out
    


    


fps = video_loader.get(cv2.CAP_PROP_FPS) or 30.0
rows = []                                   # one dict per frame, turned into a dataframe after the loop
kp_store = {}                               # frame -> (keypoints, scores), reused for the output video

# where each angle is written on the frame: (joint keypoint name, label)
ANGLE_ANCHOR = {"shoulder": "shoulder", "hip": "hip", "knee": "knee", "ankle": "ankle"}


def draw_angles(img, k, res, view):
    '''
    Writes each angle next to its joint on the camera-facing side, plus the foot pitch under the heel.
    Also draws the skeleton lines the angle was computed from.
    '''
    i = SIDE[view]
    P = lambda name: tuple(map(int, k[i[name]]))

    for name, anchor in ANGLE_ANCHOR.items():
        val = res.get(name, np.nan)
        x, y = P(anchor)
        txt = f"{name} {val:.0f}" if not np.isnan(val) else f"{name} --"
        colour = (0, 255, 0) if not np.isnan(val) else (0, 0, 255)
        cv2.circle(img, (x, y), 6, colour, -1, cv2.LINE_AA)
        cv2.putText(img, txt, (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)   # outline
        cv2.putText(img, txt, (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

    pitch = res.get("foot_pitch", np.nan)
    if not np.isnan(pitch):
        hx, hy = P("heel")
        cv2.putText(img, f"pitch {pitch:.0f}", (hx + 12, hy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, f"pitch {pitch:.0f}", (hx + 12, hy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA)


while True:
    ok, img = video_loader.read()
    if not ok: break

    n_frame += 1
    t0 = time.perf_counter()
    dets = model.det_model(img) #for the bbox of the people in the n_frame
    TIMES["detect"] += time.perf_counter() - t0
    if len(dets) == 0:
        rows.append(dict(frame=n_frame, t=n_frame / fps, view=None))
        if SHOW:
            cv2.imshow("frame", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue

    box = max(dets, key=lambda b: (b[2] - b[0])*(b[3]-b[1]))

    t0 = time.perf_counter()
    keypoints, scores = model.pose_model(img, bboxes=[box])
    TIMES["pose"] += time.perf_counter() - t0
    k, s = keypoints[0], scores[0]
    kp_store[n_frame] = (keypoints, scores)

    t0 = time.perf_counter()
    facing_pos = detect_facing_position(k)
    res = score_overhead_squat(k, s, facing_pos)          # None for FRONT / BACK
    TIMES["geometry"] += time.perf_counter() - t0

    rows.append(dict(frame=n_frame, t=n_frame / fps, **(res if res else dict(view=facing_pos))))

    if n_frame % PROGRESS_EVERY == 0:
        print(f"frame {n_frame}  {facing_pos}", flush=True)

    if SHOW:
        t0 = time.perf_counter()
        img = draw_skeleton(img, keypoints, scores, kpt_thr=0.3)
        if res:
            draw_angles(img, k, res, facing_pos)
        cv2.putText(img, f"{facing_pos} FACING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        TIMES["draw"] += time.perf_counter() - t0
        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


video_loader.release()
if SHOW:
    cv2.destroyAllWindows()
TIMES["pass1_total"] = time.perf_counter() - t_pass1

df = pd.DataFrame(rows)
df.to_csv("overhead_squat_frames.csv", index=False)


# ============================================================ scoring: bands + hold detection
import json

BANDS = {"shoulder": [145, 150, 161.5, 170],
         "hip":      [102.2, 112.1, 122.0, 131.9],
         "knee":     [106.7, 114.4, 123.1, 134.3],
         "ankle":    [30.1, 33.3, 36.0, 39.1]}
LABELS = ["High Risk", "Below Avg", "Average", "Above Avg", "Elite"]

def band(joint, value):
    if np.isnan(value):
        return "n/a"
    return LABELS[np.searchsorted(BANDS[joint], value, side="right")]


# pose definition in absolute terms, so it does not matter how much of the video is something else
POSE = dict(knee_min=90, hip_min=70, shoulder_min=120, max_speed=3.0)   # degrees, deg/frame
MIN_HOLD_S    = 1.0
MERGE_GAP_S   = 0.5
HEEL_LIFT_DEG = 12.0

def smooth(col, w=5):
    return col.rolling(w, center=True, min_periods=1).median()

def is_overhead_squat(d):
    '''Boolean series: frame looks like a held overhead squat.'''
    knee = smooth(d["knee"]); hip = smooth(d["hip"]); sh = smooth(d["shoulder"])
    speed = knee.diff().abs().rolling(3, min_periods=1).mean()
    return ((knee >= POSE["knee_min"]) & (hip >= POSE["hip_min"]) &
            (sh >= POSE["shoulder_min"]) & (speed < POSE["max_speed"])).fillna(False)

def runs_of(mask, min_len, merge_gap):
    '''(start, end) index pairs for consecutive True stretches; short gaps are bridged, short runs dropped.'''
    runs, start = [], None
    for idx, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if runs and start - runs[-1][1] < merge_gap:
                runs[-1] = (runs[-1][0], idx)
            else:
                runs.append((start, idx))
            start = None
    return [r for r in runs if r[1] - r[0] >= min_len]


def score_hold(d, hold, fps):
    # standing reference: the 5 s before this hold, knee under 20, feet visible
    pre = d.iloc[max(0, hold.index[0] - int(5 * fps)):hold.index[0]]
    standing = pre[(pre.knee < 20) & pre.foot_pitch.notna()]
    heel_lift = hold.foot_pitch.median() - standing.foot_pitch.median() if len(standing) else np.nan

    r = {}
    for j in ("shoulder", "hip", "knee", "ankle"):
        med = hold[j].median()
        r[j] = dict(value=round(float(med), 1),
                    band=band(j, med),
                    iqr=round(float(hold[j].quantile(0.75) - hold[j].quantile(0.25)), 1),
                    valid=round(float(hold[j].notna().mean()), 2))
    if np.isnan(heel_lift):
        r["ankle"]["band"] = "n/a (no standing reference)"
    elif heel_lift > HEEL_LIFT_DEG:
        r["ankle"]["band"] = "n/a (heel lift)"
    r["heel_lift"] = None if np.isnan(heel_lift) else round(float(heel_lift), 1)
    r["frames"] = [int(hold.frame.iloc[0]), int(hold.frame.iloc[-1])]
    r["duration_s"] = round(len(hold) / fps, 1)
    return r


t0 = time.perf_counter()
results = {}
for view in ("LEFT", "RIGHT"):
    d = df[df.view == view].reset_index(drop=True)
    if d.empty:
        continue
    mask = is_overhead_squat(d)
    holds = []
    for a, b in runs_of(mask, int(MIN_HOLD_S * fps), int(MERGE_GAP_S * fps)):
        hold = d.iloc[a:b]
        # trim the settle-in / rise-out edges: keep frames near this hold's own peak
        hold = hold[smooth(hold["knee"]) >= 0.85 * hold["knee"].quantile(0.98)]
        holds.append(score_hold(d, hold, fps))
    results[view] = holds

TIMES["scoring"] = time.perf_counter() - t0

print(json.dumps(results, indent=2))


# VIDEO OVERLAY
BAND_COLOUR = {"High Risk": (0, 0, 255), "Below Avg": (0, 128, 255), "Average": (0, 255, 255),
               "Above Avg": (0, 255, 0), "Elite": (255, 200, 0)}

# frame -> scored hold, so every frame inside a hold knows its bands
frame_to_hold = {}
for view, holds in results.items():
    for r in holds:
        for f in range(r["frames"][0], r["frames"][1] + 1):
            frame_to_hold[f] = (view, r)

per_frame = df.set_index("frame")

video_loader = cv2.VideoCapture(video_path)
W = int(video_loader.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(video_loader.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("overhead_squat_scored.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

t0 = time.perf_counter()
t0 = time.perf_counter()
n_frame = 0
while True:
    ok, img = video_loader.read()
    if not ok: break
    n_frame += 1

    row = per_frame.loc[n_frame]
    view = row["view"] if isinstance(row["view"], str) else "NO PERSON"   # row.view would be the Series method

    if n_frame in kp_store:
        keypoints, scores = kp_store[n_frame]
        img = draw_skeleton(img, keypoints, scores, kpt_thr=0.3)
        if view in SIDE:
            draw_angles(img, keypoints[0], row.to_dict(), view)

    cv2.putText(img, f"{view} FACING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    if n_frame in frame_to_hold:
        _, r = frame_to_hold[n_frame]
        cv2.rectangle(img, (10, 60), (W - 10, 200), (0, 0, 0), -1)
        cv2.putText(img, "HOLD", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y = 118
        for joint in ("shoulder", "hip", "knee", "ankle"):
            txt = f"{joint:<9}{r[joint]['value']:>6.1f}   {r[joint]['band']}"
            cv2.putText(img, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        BAND_COLOUR.get(r[joint]["band"], (200, 200, 200)), 2, cv2.LINE_AA)
            y += 26
        if r["heel_lift"] is not None:
            cv2.putText(img, f"heel lift {r['heel_lift']:.0f} deg", (W - 230, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if r["heel_lift"] > 12 else (0, 255, 0), 2, cv2.LINE_AA)

    writer.write(img)

video_loader.release()
writer.release()
TIMES["video_out"] = time.perf_counter() - t0
print("wrote overhead_squat_scored.mp4")


# ============================================================ timing summary
n_scored = len(kp_store)                      # frames where a person was found and the pose model ran
video_s  = n_frame / fps
print(f"\n{'stage':<12}{'total s':>10}{'ms/frame':>12}")
for stage in ("detect", "pose", "geometry", "draw"):
    print(f"{stage:<12}{TIMES[stage]:>10.2f}{1000 * TIMES[stage] / max(n_scored, 1):>12.2f}")
per_frame_ms = 1000 * (TIMES["detect"] + TIMES["pose"] + TIMES["geometry"]) / max(n_scored, 1)
print(f"{'model+geom':<12}{'':>10}{per_frame_ms:>12.2f}   -> {1000 / per_frame_ms:.1f} fps possible without drawing")
print(f"{'pass1 total':<12}{TIMES['pass1_total']:>10.2f}   ({TIMES['pass1_total'] / video_s:.2f}x video length, {n_frame} frames)")
print(f"{'scoring':<12}{TIMES['scoring']:>10.3f}")
print(f"{'video out':<12}{TIMES['video_out']:>10.2f}")

results["timing"] = dict(frames=n_frame, scored_frames=n_scored, video_s=round(video_s, 1),
                         ms_per_frame={k: round(1000 * TIMES[k] / max(n_scored, 1), 2) for k in ("detect", "pose", "geometry", "draw")},
                         realtime_factor=round(TIMES["pass1_total"] / video_s, 2),
                         totals_s={k: round(v, 2) for k, v in TIMES.items()})
with open("overhead_squat_results.json", "w") as f:
    json.dump(results, f, indent=2)
