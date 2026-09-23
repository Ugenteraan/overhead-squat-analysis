import cv2
import numpy as np
import matplotlib.pyplot as plt
from rtmlib import Wholebody, draw_skeleton


video_path = "/home/topiary-pc/Projects/MovesMethod/Example1.mov"
video_loader = cv2.VideoCapture(video_path)

model = Wholebody(mode="lightweight", backend="onnxruntime", device="cpu")

n_frame = 0


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
      raw_angle:    angle between the shin (ankle->knee) and the sole (heel->toe mid). 90 when the foot is
                    perpendicular to the shin. The exercise-specific conversion (e.g. 90 - raw) is applied by the scorer.
      foot_pitch:   angle of the sole against the image horizontal, positive when the heel is above the toes.
                    Used to detect heel lift by comparing against the standing baseline.
    '''

    vec_knee = knee_point - ankle_point
    toe_mid = (big_toe_point + small_toe_point)/2
    vec_foot = toe_mid - heel_point

    ankle_raw = angle_between(vec_foot, vec_knee)
    foot_pitch = np.degrees(np.arctan2(-vec_foot[1], abs(vec_foot[0])))   # image y points down; abs(dx) makes it facing-independent

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
    conv maps each raw joint angle to the exercise's clinical convention (see OVERHEAD_SQUAT).
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
    


    




while True:
    ok, img = video_loader.read()
    if not ok: break

    n_frame += 1
    dets = model.det_model(img) #for the bbox of the people in the n_frame
    if len(dets) == 0:
        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    box = max(dets, key=lambda b: (b[2] - b[0])*(b[3]-b[1]))

    keypoints, scores = model.pose_model(img, bboxes=[box])
    k, s = keypoints[0], scores[0]

    facing_pos = detect_facing_position(k)
    
    cv2.putText(img, f"{facing_pos} FACING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("frame", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    

cv2.destroyAllWindows()   
