# Pose Keypoint Tracker

Requirements:

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py --video KakaoTalk_20260605_161243459.mp4 --output output/pose_coords.csv
```

You can also edit `VIDEO_PATH` in `main.py` or pass another `.mp4` file with `--video`.

Press `q` or `Esc` to quit. The script draws pose landmarks and writes per-frame landmark coordinates to the CSV.

Generate actuator WAV versions:

```bash
python movement_to_actuator_wav.py --input output/pose_coords.csv --output-dir output/wav_versions
```

This writes 15 mono WAV files: 5 body parts (`head`, `left_arm`, `right_arm`, `left_leg`, `right_leg`) across 3 amplitude mappings (`angle_delta`, `movement`, `movement_delta`).

Run the GUI workflow:

```bash
python gui_vibrotactile_extractor.py
```

The GUI lets you choose a video, reviews detected person candidates, merge candidates that belong to the same person, then generate the selected person's pose CSV and 15 vibrotactile WAV files.
You can also play, pause, and scrub through the selected video inside the GUI while checking the detected person boxes.

## Computational Method

This project extracts human body motion from video using YOLO person detection and MediaPipe Pose Landmarker. The GUI workflow first detects candidate person regions, selects the target person, estimates body landmarks, interpolates missing frames, and writes a frame-wise pose CSV. The CSV is then converted into vibrotactile actuator WAV signals.

### Keypoint extraction summary

For each video frame, the current GUI workflow detects candidate person regions with a YOLO person detector, selects the candidate closest to the center of the frame, and passes that region to MediaPipe Pose Landmarker to extract pose keypoints. If pose estimation fails on the YOLO candidate crop, the workflow falls back to running Pose Landmarker on the full frame.

Only the 13 keypoints needed for vibrotactile conversion are written to the CSV:

| Body region | Keypoints |
| --- | --- |
| Head | `nose` |
| Arms | `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist` |
| Legs | `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle` |

Each keypoint row includes `frame`, `timestamp`, `type=keypoint`, `name`, normalized coordinates (`x_norm`, `y_norm`, `z_norm`), `visibility`, pixel coordinates (`x_px`, `y_px`), frame-to-frame displacement (`dx_px`, `dy_px`), movement distance (`distance_px`), and pixel speed (`speed_px_per_sec`).

Frames with missing detections are filled by linear interpolation from the nearest valid frames before and after the gap. Frame-to-frame movement and joint angles are then computed from this interpolated keypoint sequence.

### Pose landmarks

For each video frame \(t\), MediaPipe returns normalized landmark coordinates:

$$p_i(t) = (x_i(t), y_i(t), z_i(t))$$

where \(i\) is the landmark index. The normalized image coordinates are converted to pixel coordinates by:

$$X_i(t) = x_i(t) W$$
$$\quad Y_i(t) = y_i(t) H$$

where \(W\) and \(H\) are the video frame width and height. The main target landmarks are the nose, shoulders, elbows, wrists, hips, knees, and ankles.

### Frame-to-frame movement

For a landmark \(i\), the displacement between consecutive frames is computed as:

$$\Delta X_i(t) = X_i(t) - X_i(t-1)$$


$$\Delta Y_i(t) = Y_i(t) - Y_i(t-1)$$


The Euclidean movement distance in pixels is:


$$d_i(t) = \sqrt{\Delta X_i(t)^2 + \Delta Y_i(t)^2}$$


The instantaneous pixel speed is:


$$v_i(t) = \frac{d_i(t)}{\Delta t}$$


where ($\Delta t = 1 / fps$). In the exported CSV, these values correspond to `dx_px`, `dy_px`, `distance_px`, and `speed_px_per_sec`.

### Joint angle calculation

Joint angles are calculated using three landmarks. Given three 2D points \(A\), \(B\), and \(C\), the angle at the center point \(B\) is computed from the vectors:


$$\vec{BA} = A - B$$
$$\quad \vec{BC} = C - B$$


The joint angle is:


$$\theta = \cos^{-1}\left(\frac{\vec{BA} \cdot \vec{BC}}{||\vec{BA}||\,||\vec{BC}||}\right)$$


The result is converted from radians to degrees:


$$\theta_{deg} = \theta \times \frac{180}{\pi}$$


The implemented angle function clamps the cosine value to \([-1, 1]\) before applying arccosine to prevent numerical errors from floating-point precision.

The current GUI workflow exports the following anatomical angles:

| CSV angle name | Landmark triplet | Center joint |
| --- | --- | --- |
| `left_elbow` | left shoulder - left elbow - left wrist | left elbow |
| `right_elbow` | right shoulder - right elbow - right wrist | right elbow |
| `left_knee` | left hip - left knee - left ankle | left knee |
| `right_knee` | right hip - right knee - right ankle | right knee |

### Head orientation

For vibrotactile conversion, head orientation is estimated from the nose and the midpoint of the two shoulders. Let:


```python
S(t) = \left(
            \frac{X_{left\_shoulder}(t) + X_{right\_shoulder}(t)}{2},
            \frac{Y_{left\_shoulder}(t) + Y_{right\_shoulder}(t)}{2}
            \right)
```


The orientation angle is:


$$\phi(t) = atan2(Y_{nose}(t) - S_y(t), X_{nose}(t) - S_x(t))$$


and is converted to degrees. Because this angle is circular, frame-to-frame change is computed with circular wrapping:


$$\Delta \phi(t) = \left|((\phi(t)-\phi(t-1)+180) \bmod 360) - 180 \right|$$


### Body-part motion features

The WAV conversion script computes three motion-to-amplitude mappings for each body part:

| Method | Raw feature |
| --- | --- |
| `movement` | Mean landmark movement distance for the body part |
| `movement_delta` | Absolute frame-to-frame change of the movement feature |
| `angle_delta` | Absolute frame-to-frame change of the body-part angle |

All three methods first compute a raw feature \(r_B(t)\) for each body part and then normalize it into a 0 to 1 amplitude envelope. The methods differ in how they define the raw feature.

The body-part definitions are:

| Body part | Movement landmarks | Angle source |
| --- | --- | --- |
| `head` | nose | head orientation |
| `left_arm` | left shoulder, left elbow, left wrist | left elbow |
| `right_arm` | right shoulder, right elbow, right wrist | right elbow |
| `left_leg` | left hip, left knee, left ankle | left knee |
| `right_leg` | right hip, right knee, right ankle | right knee |

For a body part \(B\) with landmarks \(K_B\), the movement feature is:

\[
m_B(t) = \frac{1}{|K_B|}\sum_{i \in K_B} d_i(t)
\]

`movement` method uses this value directly:

\[
r_B(t) = m_B(t)
\]

This method represents how far the body part moved in a frame as vibration intensity. Larger movement distances produce larger actuator amplitudes.

The movement-change feature is:


$$\Delta m_B(t) = |m_B(t) - m_B(t-1)|$$


`movement_delta` method uses the movement-change value:


$$r_B(t) = \Delta m_B(t)$$


This method responds to how abruptly the amount of movement changes, rather than to the movement magnitude itself. It produces stronger vibration around motion changes such as starts, stops, and direction changes than during steady movement at a similar speed.

The joint-angle-change feature is:


$$\Delta \theta_B(t) = |\theta_B(t) - \theta_B(t-1)|$$


except for head orientation, which uses the circular angle difference above.

`angle_delta` method uses the body-part angle-change value:


$$r_B(t) = \Delta \theta_B(t)$$


For arms and legs, this method uses the change in the corresponding elbow or knee angle. For the head, it uses the change in head orientation computed from the nose and the midpoint between the two shoulders. This converts posture changes, such as bending, extending, or turning the head, into vibrotactile cues rather than relying on absolute keypoint displacement.

### Amplitude normalization and WAV synthesis

Each raw feature series is normalized independently to the range \([0, 1]\):


$$a_B(t) = \begin{cases} \frac{r_B(t)}{\max_t r_B(t)}, & \max_t r_B(t) > 0 \\0, & \max_t r_B(t) = 0 \end{cases}$$


where ($r_B(t)$) is the selected raw feature and ($a_B(t)$) is the actuator amplitude. The WAV signal is generated as a mono sine carrier:

\[
s(n) = \sin(2\pi f_c n / f_s)\,a_B(t)\,g
\]

where \(f_c\) is the carrier frequency, \(f_s\) is the sample rate, and \(g\) is the output gain. By default, the script uses a 175 Hz carrier, 44.1 kHz sampling rate, and gain 0.8.
