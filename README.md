# Overhead Squat Joint-Angle Scoring

`analysis.py` takes a phone video of an overhead squat, measures shoulder, hip, knee and ankle angles on the side facing the camera, finds the held squat positions, and maps the medians to the research score bands.

Pose model: RTMW-m (COCO-WholeBody, 133 keypoints) via [rtmlib](https://github.com/Tau-J/rtmlib), ONNX Runtime, CPU by default.

## Setup

```
pip install rtmlib onnxruntime opencv-python numpy pandas
```

For GPU, install `onnxruntime-gpu` instead of `onnxruntime` and set `device="cuda"` in the `Wholebody(...)` line. Check that `onnxruntime.get_available_providers()` lists `CUDAExecutionProvider`, otherwise it silently runs on CPU.

## Run

1. Set `video_path` at the top of `analysis.py`.
2. `python analysis.py`

Outputs, written next to the script:

| file | contents |
|---|---|
| `overhead_squat_frames.csv` | one row per frame: view, four angles, foot pitch |
| `overhead_squat_results.json` | per side, one entry per hold: median, band, IQR, valid fraction per joint, heel lift, frame range, plus timing |
| `overhead_squat_scored.mp4` | input video with skeleton, angles at each joint, and the band panel during holds |

Recording guidance: phone on the floor or a low stand, full body in frame, hold the bottom of the squat for at least a second, once facing left and once facing right. Front view is used only for symmetry, not scoring.

## Parameters

All at the top of the file or in the scoring section.

| name | default | what it does |
|---|---|---|
| `video_path` | `Example1.mov` | input video |
| `SHOW` | `False` | show each frame in a window during the first pass. Slower. `q` quits. |
| `PROGRESS_EVERY` | `100` | print a progress line every N frames |
| `mode` in `Wholebody(...)` | `lightweight` | RTMW-m. `balanced` / `performance` are larger and slower. |
| `thr` in `score_overhead_squat` | `0.3` | keypoint confidence gate. Any angle using a keypoint below this is NaN for that frame. |
| `ratio > 0.45` in `detect_facing_position` | `0.45` | shoulder-width / torso-length above this is FRONT or BACK, below is a side view. Tested: frontal 0.69, oblique 0.44, profile 0.07. |
| `POSE.knee_min` | `90` | minimum knee flexion for a frame to count as a squat |
| `POSE.hip_min` | `70` | minimum hip flexion |
| `POSE.shoulder_min` | `120` | minimum shoulder flexion, i.e. arms actually overhead. Rules out kneeling and sitting. |
| `POSE.max_speed` | `3.0` | max knee change in deg/frame. Separates a hold from the descent and rise. |
| `MIN_HOLD_S` | `1.0` | shortest run of squat frames that counts as a hold |
| `MERGE_GAP_S` | `0.5` | holds closer than this are joined, so one dropped frame does not split a rep |
| `HEEL_LIFT_DEG` | `12.0` | heel lift above this withholds the ankle band |
| `0.85 * quantile(0.98)` in the hold loop | | trims each hold to frames near its own knee peak, dropping settle-in and rise-out |
| `knee < 20` in `score_hold` | `20` | frames in the 5 s before a hold with knee under this are the standing reference for heel lift |

## Angle definitions

All 2D, image plane, camera-facing side only. Raw angle between two segments, then converted by the `OVERHEAD_SQUAT` table.

| joint | vectors from the vertex | conversion |
|---|---|---|
| shoulder | shoulder→hip, shoulder→elbow | raw |
| hip | hip→shoulder, hip→knee | 180 − raw |
| knee | knee→hip, knee→ankle | 180 − raw |
| ankle | ankle→knee, heel→toe-midpoint | 90 − raw |

Foot pitch is the heel→toe line against the image horizontal. Heel lift is the hold's pitch minus the standing pitch in the same view.

Other exercises: add a new conversion table and pass it as `conv` to `score_overhead_squat`; the angle functions stay unchanged.

## Bands

From the brief, applied to the hold median only, never to a single frame.

| joint | High Risk | Below Avg | Average | Above Avg | Elite |
|---|---|---|---|---|---|
| shoulder | <145 | 145–150 | 150–161.5 | 161.5–170 | ≥170 |
| hip | <102.2 | 102.2–112.1 | 112.1–122.0 | 122.0–131.9 | ≥131.9 |
| knee | <106.7 | 106.7–114.4 | 114.4–123.1 | 123.1–134.3 | ≥134.3 |
| ankle | <30.1 | 30.1–33.3 | 33.3–36.0 | 36.0–39.1 | ≥39.1 |

The ankle bands assume a flat foot. If the heel lifts, the band is reported as `n/a (heel lift)` and the lift angle is given instead.

## Verifying accuracy

- Open `overhead_squat_scored.mp4` and check the dots sit on the joints during the holds. A misplaced heel or toe shows up directly in the ankle number.
- `Test keypoints.ipynb` and `Untitled.ipynb` draw individual keypoints by index on still frames to confirm the index map.
- IQR and valid fraction in the JSON say how stable each hold was. IQR above about 5 degrees or valid fraction below 0.8 means the hold is not trustworthy.
- For a reference measurement, run a 3D model such as SAM3D on the same frames and compare the hold medians. On `Example1.mov` the RTMW-m knee and hip medians were within 2–10 degrees of SAM3D's 2D projection.

## Timing

Printed at the end of the run and stored under `timing` in the JSON. On a desktop CPU with RTMW-m: detector about 29 ms per frame, pose about 35 ms, geometry under 0.1 ms. The detector runs every frame; re-detecting every 30 frames and cropping from the previous keypoints in between is the first optimisation to make.
