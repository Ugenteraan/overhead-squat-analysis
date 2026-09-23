# Overhead squat scoring

`analysis.py` runs RTMW-m (rtmlib, ONNX) on a squat video, measures shoulder / hip / knee / ankle angles on the side facing the camera, finds the held squat positions and maps them to the score bands from the brief.

## Run

```
pip install rtmlib onnxruntime opencv-python numpy pandas
python analysis.py
```

Change `video_path` at the top of the file. Outputs land next to the script:

- `overhead_squat_frames.csv` – angles per frame
- `overhead_squat_results.json` – per hold: median, band, IQR, valid fraction, heel lift, timing
- `overhead_squat_scored.mp4` – video with skeleton, angles and the band panel

Set `SHOW = True` to watch frames while it runs (slower, `q` quits).

GPU: install `onnxruntime-gpu` instead of `onnxruntime` and set `device="cuda"`. If `onnxruntime.get_available_providers()` doesn't list CUDA it silently runs on CPU, timing won't change.

## Parameters

| | default | |
|---|---|---|
| `thr` | 0.3 | keypoint confidence. Angles using anything below this are NaN. |
| view ratio | 0.45 | shoulder width / torso length. Above = front/back, below = side. Measured 0.69 frontal, 0.44 oblique, 0.07 profile. |
| `POSE` | knee ≥ 90, hip ≥ 70, shoulder ≥ 120, knee speed < 3°/frame | what counts as a held overhead squat frame. Absolute, so it doesn't matter how much of the video is something else. |
| `MIN_HOLD_S` | 1.0 | shortest hold |
| `MERGE_GAP_S` | 0.5 | joins holds split by a dropped frame |
| `HEEL_LIFT_DEG` | 12 | above this the ankle band is withheld |
| standing ref | knee < 20 in the 5 s before a hold | baseline for heel lift |

## Angles

2D, camera-facing side only. Shoulder: angle between shoulder→hip and shoulder→elbow. Hip and knee: 180 minus the angle between the two segments. Ankle: 90 minus the angle between shin and sole (heel→toe midpoint). Conversions live in the `OVERHEAD_SQUAT` table so another exercise is just another table.

Heel lift = sole pitch in the hold minus sole pitch standing. The ankle bands assume a flat foot, so with heel lift the band is `n/a` and the lift is reported instead.

## Timing

Full run on `Example1.mov` (1648 frames, 720×1280, 54.9 s), CPU only, RTMW-m lightweight mode:

| stage | total | per frame |
|---|---|---|
| detector (YOLOX-tiny) | 47.4 s | 28.7 ms |
| pose (RTMW-m) | 57.6 s | 34.9 ms |
| view + angles | 0.09 s | 0.06 ms |
| first pass total | 107 s | 1.95× video length |
| hold detection + bands | 8 ms | |
| output video | 8.4 s | |

So ~64 ms per frame for the models, 15.7 fps. The geometry is free, it's all detector and pose. Detector is 45% of the budget and runs every frame; re-detecting every 30 frames and cropping from the last keypoints in between is the obvious next step and should bring it under real time on this CPU.

Model comparison from before settling on RTMW-m (CPU, per frame): RTMPose-s 6.6 ms, RTMPose-m 18 ms, RTMW-m 36 ms, RTMW-l 101 ms, RTMW3D-x 150 ms, BlazePose lite/full/heavy 36/49/151 ms. SAM3D ~45 ms and Sapiens-1B 82 ms on GPU. RTMW-m was the smallest one with feet and hands that kept the knee within a few degrees of SAM3D. See `pose_model_comparison.pdf`.

## Checking it

Open the scored video and look at where the dots are during the holds. `Test keypoints.ipynb` and `Untitled.ipynb` draw single keypoints by index if you need to confirm the map. IQR above ~5° or valid fraction under 0.8 in the JSON means don't trust that hold.
