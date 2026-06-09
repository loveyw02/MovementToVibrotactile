# Process

## 2026-06-05

- Reviewed the existing `main.py` pose tracking flow.
- Kept the implementation compact and close to the original MediaPipe Pose loop.
- Added selected skeleton keypoints for shoulders, elbows, wrists, hips, knees, and ankles.
- Added per-frame pixel movement values: `dx_px`, `dy_px`, and `distance_px`.
- Added speed calculation as `distance_px / dt`, using the video FPS when available.
- Added joint angle calculation for left/right elbows and knees.
- Updated the CSV shape to store both `keypoint` rows and `angle` rows.
- Found an environment issue: installed NumPy 2.x is incompatible with the current MediaPipe/matplotlib binary stack.
- Pinned `numpy<2` in `requirements.txt` so the project installs a compatible NumPy version.
- Installed compatible packages and verified imports: NumPy 1.26.4, OpenCV 4.11.0, MediaPipe 0.10.35.
- Confirmed that the remaining runtime requirement is setting `VIDEO_PATH` to a real video file.
- Found that `pose_coords.csv` was not created because `VIDEO_PATH` still pointed to `C:\path\to\video.mp4`.
- Changed the default video path to the existing local MP4 and added a `--video` argument.
- Found that MediaPipe 0.10.35 no longer exposes `mp.solutions`, which this compact script uses.
- Pinned `mediapipe==0.10.14` to keep compatibility with `mp.solutions.pose`.
- Added `nose` to the tracked keypoints so head movement can be calculated.
- Added `movement_to_actuator_wav.py` to convert keypoint movement into a 5-channel actuator amplitude WAV.
- Defined WAV channels as left arm, right arm, left leg, right leg, and head.
