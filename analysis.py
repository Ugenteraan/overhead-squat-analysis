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
