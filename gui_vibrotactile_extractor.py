import csv
import math
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from tkinter import filedialog, messagebox, ttk

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision

from movement_to_actuator_wav import generate_all_wavs, read_pose_csv, write_summary_csv


TARGET_PADDING = 0.18
MIN_TRACK_HITS = 2
SAMPLE_SECONDS = 0.5
CENTER_TRACK_ID = 0
MAX_POSES = 6
MIN_LANDMARK_VISIBILITY = 0.2
YOLO_MODEL_NAME = "yolo11n.pt"
YOLO_PERSON_CLASS_ID = 0
YOLO_CONFIDENCE = 0.25
YOLO_MAX_CANDIDATES = 3
POSE_LANDMARKER_MODEL_URL = (
	"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
	"pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
POSE_LANDMARKER_MODEL_PATH = Path.home() / ".mediapipe_models" / "pose_landmarker_full.task"
TARGET_LANDMARKS = {
	"nose": 0,
	"left_shoulder": 11,
	"right_shoulder": 12,
	"left_elbow": 13,
	"right_elbow": 14,
	"left_wrist": 15,
	"right_wrist": 16,
	"left_hip": 23,
	"right_hip": 24,
	"left_knee": 25,
	"right_knee": 26,
	"left_ankle": 27,
	"right_ankle": 28,
}
POSE_CONNECTIONS = [
	(11, 12),
	(11, 13),
	(13, 15),
	(12, 14),
	(14, 16),
	(11, 23),
	(12, 24),
	(23, 24),
	(23, 25),
	(25, 27),
	(24, 26),
	(26, 28),
]


def angle_deg(a, b, c):
	ab = (a["x_px"] - b["x_px"], a["y_px"] - b["y_px"])
	cb = (c["x_px"] - b["x_px"], c["y_px"] - b["y_px"])
	dot = ab[0] * cb[0] + ab[1] * cb[1]
	len_ab = math.hypot(*ab)
	len_cb = math.hypot(*cb)
	if len_ab == 0 or len_cb == 0:
		return None
	cosine = max(-1.0, min(1.0, dot / (len_ab * len_cb)))
	return math.degrees(math.acos(cosine))


def iou(a, b):
	ax, ay, aw, ah = a
	bx, by, bw, bh = b
	x1 = max(ax, bx)
	y1 = max(ay, by)
	x2 = min(ax + aw, bx + bw)
	y2 = min(ay + ah, by + bh)
	inter = max(0, x2 - x1) * max(0, y2 - y1)
	union = aw * ah + bw * bh - inter
	return inter / union if union else 0


def center_distance(a, b):
	ax, ay, aw, ah = a
	bx, by, bw, bh = b
	return math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))


def clamp_box(box, width, height, padding=0.0):
	x, y, w, h = box
	pad_x = int(w * padding)
	pad_y = int(h * padding)
	x1 = max(0, int(x - pad_x))
	y1 = max(0, int(y - pad_y))
	x2 = min(width, int(x + w + pad_x))
	y2 = min(height, int(y + h + pad_y))
	return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def union_boxes(boxes):
	x1 = min(x for x, y, w, h in boxes)
	y1 = min(y for x, y, w, h in boxes)
	x2 = max(x + w for x, y, w, h in boxes)
	y2 = max(y + h for x, y, w, h in boxes)
	return x1, y1, x2 - x1, y2 - y1


def ensure_pose_landmarker_model(status=None):
	if POSE_LANDMARKER_MODEL_PATH.exists():
		return POSE_LANDMARKER_MODEL_PATH
	POSE_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	if status:
		status("Pose Landmarker 모델을 다운로드하고 있습니다...")
	urllib.request.urlretrieve(POSE_LANDMARKER_MODEL_URL, POSE_LANDMARKER_MODEL_PATH)
	return POSE_LANDMARKER_MODEL_PATH


def create_pose_landmarker(running_mode, num_poses=MAX_POSES, status=None):
	model_path = ensure_pose_landmarker_model(status)
	options = mp_tasks_vision.PoseLandmarkerOptions(
		base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
		running_mode=running_mode,
		num_poses=num_poses,
		min_pose_detection_confidence=0.35,
		min_pose_presence_confidence=0.35,
		min_tracking_confidence=0.35,
		output_segmentation_masks=False,
	)
	return mp_tasks_vision.PoseLandmarker.create_from_options(options)


def create_yolo_model(status=None):
	try:
		from ultralytics import YOLO
	except ImportError as exc:
		raise RuntimeError("YOLO detector requires ultralytics. Run: pip install ultralytics") from exc
	if status:
		status("YOLO person detector를 로드하고 있습니다...")
	return YOLO(YOLO_MODEL_NAME)


def frame_to_mp_image(frame):
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect_yolo_person_boxes(yolo_model, frame):
	results = yolo_model.predict(
		frame,
		classes=[YOLO_PERSON_CLASS_ID],
		conf=YOLO_CONFIDENCE,
		verbose=False,
	)
	if not results:
		return []

	boxes = []
	image_h, image_w = frame.shape[:2]
	for result in results:
		if result.boxes is None:
			continue
		for box in result.boxes:
			x1, y1, x2, y2 = box.xyxy[0].tolist()
			confidence = float(box.conf[0]) if box.conf is not None else 0.0
			x1 = max(0, min(image_w - 1, int(x1)))
			y1 = max(0, min(image_h - 1, int(y1)))
			x2 = max(0, min(image_w, int(x2)))
			y2 = max(0, min(image_h, int(y2)))
			if x2 <= x1 or y2 <= y1:
				continue
			boxes.append((x1, y1, x2 - x1, y2 - y1, confidence))
	return boxes


def center_distance_from_frame(box, frame_width, frame_height):
	x, y, w, h = box
	return math.hypot((x + w / 2) - (frame_width / 2), (y + h / 2) - (frame_height / 2))


def landmark_visibility(landmark):
	return getattr(landmark, "visibility", getattr(landmark, "presence", 1.0))


def visible_landmarks(landmarks, indexes=None):
	if indexes is None:
		indexes = range(len(landmarks))
	return [
		landmarks[idx]
		for idx in indexes
		if idx < len(landmarks) and landmark_visibility(landmarks[idx]) >= MIN_LANDMARK_VISIBILITY
	]


def pose_center(landmarks, frame_width, frame_height):
	torso = visible_landmarks(landmarks, [11, 12, 23, 24])
	points = torso or visible_landmarks(landmarks)
	if not points:
		return None
	x = sum(point.x for point in points) / len(points) * frame_width
	y = sum(point.y for point in points) / len(points) * frame_height
	return x, y


def pose_box(landmarks, frame_width, frame_height):
	points = visible_landmarks(landmarks)
	if not points:
		return None
	xs = [point.x * frame_width for point in points]
	ys = [point.y * frame_height for point in points]
	x1 = max(0, min(xs))
	y1 = max(0, min(ys))
	x2 = min(frame_width, max(xs))
	y2 = min(frame_height, max(ys))
	if x2 <= x1 or y2 <= y1:
		return None
	return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def select_center_pose(pose_landmarks, frame_width, frame_height, target_box=None):
	candidates = []
	for landmarks in pose_landmarks:
		center = pose_center(landmarks, frame_width, frame_height)
		box = pose_box(landmarks, frame_width, frame_height)
		if center is None or box is None:
			continue
		confidence = sum(landmark_visibility(point) for point in visible_landmarks(landmarks)) / max(1, len(visible_landmarks(landmarks)))
		distance = math.hypot(center[0] - frame_width / 2, center[1] - frame_height / 2)
		candidates.append((distance, -confidence, landmarks, box, center))
	if not candidates:
		return None, None, None
	if target_box is not None:
		_, _, landmarks, box, center = min(
			candidates,
			key=lambda item: (
				0 if iou(item[3], target_box) >= 0.05 else 1,
				-iou(item[3], target_box),
				center_distance(item[3], target_box),
				item[0],
				item[1],
			),
		)
	else:
		_, _, landmarks, box, center = min(candidates, key=lambda item: (item[0], item[1]))
	return landmarks, box, center


def average_pose_confidence(landmarks):
	points = visible_landmarks(landmarks)
	if not points:
		return 0.0
	return sum(landmark_visibility(point) for point in points) / len(points)


def landmarks_from_crop_to_frame(landmarks, crop_box, frame_width, frame_height):
	x, y, w, h = crop_box
	full_landmarks = []
	for lm in landmarks:
		full_landmarks.append(
			SimpleNamespace(
				x=(x + lm.x * w) / frame_width,
				y=(y + lm.y * h) / frame_height,
				z=lm.z,
				visibility=landmark_visibility(lm),
				presence=getattr(lm, "presence", landmark_visibility(lm)),
			)
		)
	return full_landmarks


def select_center_pose_with_yolo(frame, yolo_model, pose_landmarker, fallback_landmarker=None, target_box=None, previous_box=None):
	image_h, image_w = frame.shape[:2]
	person_boxes = detect_yolo_person_boxes(yolo_model, frame)
	anchor_box = previous_box or target_box
	if anchor_box is not None:
		person_boxes = sorted(
			person_boxes,
			key=lambda box: (
				0 if iou(box[:4], anchor_box) >= 0.05 else 1,
				-iou(box[:4], anchor_box),
				center_distance(box[:4], anchor_box),
				center_distance_from_frame(box[:4], image_w, image_h),
				-box[4],
			),
		)[:YOLO_MAX_CANDIDATES]
	else:
		person_boxes = sorted(
			person_boxes,
			key=lambda box: (center_distance_from_frame(box[:4], image_w, image_h), -box[4]),
		)[:YOLO_MAX_CANDIDATES]

	candidates = []
	for detected_box in person_boxes:
		crop_box = clamp_box(detected_box[:4], image_w, image_h, TARGET_PADDING)
		x, y, w, h = crop_box
		crop = frame[y:y + h, x:x + w]
		if crop.size == 0:
			continue
		result = pose_landmarker.detect(frame_to_mp_image(crop))
		crop_landmarks, _, _ = select_center_pose(result.pose_landmarks, w, h)
		if crop_landmarks is None:
			continue
		full_landmarks = landmarks_from_crop_to_frame(crop_landmarks, crop_box, image_w, image_h)
		center = pose_center(full_landmarks, image_w, image_h)
		box = pose_box(full_landmarks, image_w, image_h)
		if center is None or box is None:
			continue
		confidence = average_pose_confidence(full_landmarks)
		distance = math.hypot(center[0] - image_w / 2, center[1] - image_h / 2)
		candidates.append((distance, -confidence, -detected_box[4], full_landmarks, box, center))

	if candidates:
		if anchor_box is not None:
			_, _, _, landmarks, box, center = min(
				candidates,
				key=lambda item: (
					0 if iou(item[4], anchor_box) >= 0.05 else 1,
					-iou(item[4], anchor_box),
					center_distance(item[4], anchor_box),
					item[0],
					item[1],
					item[2],
				),
			)
		else:
			_, _, _, landmarks, box, center = min(candidates, key=lambda item: (item[0], item[1], item[2]))
		return landmarks, box, center

	if fallback_landmarker is None:
		return None, None, None
	result = fallback_landmarker.detect(frame_to_mp_image(frame))
	return select_center_pose(result.pose_landmarks, image_w, image_h, target_box=anchor_box)


def interpolate_value(before_idx, before_value, after_idx, after_value, frame_idx):
	if before_idx == after_idx:
		return before_value
	ratio = (frame_idx - before_idx) / (after_idx - before_idx)
	return before_value + (after_value - before_value) * ratio


def interpolate_pose_frames(raw_frames):
	total_frames = len(raw_frames)
	fields = ("x_norm", "y_norm", "z_norm", "visibility", "x_px", "y_px")
	interpolated = [{name: {} for name in TARGET_LANDMARKS} for _ in range(total_frames)]

	for name in TARGET_LANDMARKS:
		valid_indices = [idx for idx, frame in enumerate(raw_frames) if name in frame]
		if not valid_indices:
			for frame in interpolated:
				frame[name] = {field: 0.0 for field in fields}
			continue

		for frame_idx in range(total_frames):
			before = max((idx for idx in valid_indices if idx <= frame_idx), default=valid_indices[0])
			after = min((idx for idx in valid_indices if idx >= frame_idx), default=valid_indices[-1])
			values = {}
			for field in fields:
				values[field] = interpolate_value(
					before,
					raw_frames[before][name][field],
					after,
					raw_frames[after][name][field],
					frame_idx,
				)
			interpolated[frame_idx][name] = values

	return interpolated


class PersonTrack:
	def __init__(self, track_id, frame_idx, box):
		self.id = track_id
		self.boxes = {frame_idx: box}
		self.last_frame = frame_idx
		self.last_box = box

	def add(self, frame_idx, box):
		self.boxes[frame_idx] = box
		self.last_frame = frame_idx
		self.last_box = box

	@property
	def hits(self):
		return len(self.boxes)

	def frame_range(self):
		keys = sorted(self.boxes)
		return keys[0], keys[-1]


def detect_person_tracks(video_path, status=None):
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video: {video_path}")

	fps = cap.get(cv2.CAP_PROP_FPS) or 30
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	step = max(1, int(fps * SAMPLE_SECONDS))
	center_track = PersonTrack(CENTER_TRACK_ID, 0, (0, 0, 1, 1))
	center_track.boxes.clear()
	preview_frame = None
	frame_idx = 0

	yolo_model = create_yolo_model(status)
	with create_pose_landmarker(mp_tasks_vision.RunningMode.IMAGE, num_poses=1, status=status) as pose_landmarker, create_pose_landmarker(
		mp_tasks_vision.RunningMode.IMAGE,
		num_poses=MAX_POSES,
		status=status,
	) as fallback_landmarker:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			if preview_frame is None:
				preview_frame = frame.copy()
			if frame_idx % step != 0:
				frame_idx += 1
				continue

			image_h, image_w = frame.shape[:2]
			_, box, _ = select_center_pose_with_yolo(frame, yolo_model, pose_landmarker, fallback_landmarker)
			if box is not None:
				center_track.add(frame_idx, box)

			if status and frame_count:
				status(f"YOLO + MediaPipe 중앙 인물 분석 중... {min(100, int(frame_idx / frame_count * 100))}%")
			frame_idx += 1

	cap.release()
	filtered = [center_track] if center_track.hits >= MIN_TRACK_HITS else []
	return filtered, preview_frame


def select_box_for_frame(tracks, frame_idx, frame_width, frame_height, max_frame_gap=None):
	candidates = []
	for track in tracks:
		if not track.boxes:
			continue
		nearest = min(track.boxes, key=lambda idx: abs(idx - frame_idx))
		frame_gap = abs(nearest - frame_idx)
		if max_frame_gap is not None and frame_gap > max_frame_gap:
			continue
		candidates.append((track.boxes[nearest], frame_gap))
	if not candidates:
		return 0, 0, frame_width, frame_height

	center_box, _ = min(
		candidates,
		key=lambda item: (
			center_distance_from_frame(item[0], frame_width, frame_height),
			item[1],
			-(item[0][2] * item[0][3]),
		),
	)
	return clamp_box(center_box, frame_width, frame_height, TARGET_PADDING)


def nearest_track_box_for_frame(tracks, frame_idx, max_frame_gap=None):
	candidates = []
	for track in tracks:
		if not track.boxes:
			continue
		nearest = min(track.boxes, key=lambda idx: abs(idx - frame_idx))
		frame_gap = abs(nearest - frame_idx)
		if max_frame_gap is not None and frame_gap > max_frame_gap:
			continue
		candidates.append((track.boxes[nearest], frame_gap))
	if not candidates:
		return None
	box, _ = min(
		candidates,
		key=lambda item: (
			item[1],
			-(item[0][2] * item[0][3]),
		),
	)
	return box


def write_selected_person_pose_csv(video_path, selected_tracks, output_csv, status=None):
	angles = {
		"left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
		"right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
		"left_knee": ("left_hip", "left_knee", "left_ankle"),
		"right_knee": ("right_hip", "right_knee", "right_ankle"),
	}

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video: {video_path}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 30
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	dt = 1 / fps

	out_path = Path(output_csv)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	raw_frames = []

	yolo_model = create_yolo_model(status)
	with create_pose_landmarker(mp_tasks_vision.RunningMode.IMAGE, num_poses=1, status=status) as pose, create_pose_landmarker(
		mp_tasks_vision.RunningMode.IMAGE,
		num_poses=MAX_POSES,
		status=status,
	) as fallback_pose:
		frame_idx = 0
		previous_box = None
		max_track_gap = max(1, int(fps * SAMPLE_SECONDS))
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			image_h, image_w = frame.shape[:2]
			target_box = nearest_track_box_for_frame(selected_tracks, frame_idx, max_frame_gap=max_track_gap)
			landmarks, selected_box, _ = select_center_pose_with_yolo(
				frame,
				yolo_model,
				pose,
				fallback_pose,
				target_box=target_box,
				previous_box=previous_box,
			)
			if selected_box is not None:
				previous_box = selected_box
			frame_points = {}

			if landmarks:
				for name, landmark_id in TARGET_LANDMARKS.items():
					lm = landmarks[landmark_id]
					frame_points[name] = {
						"x_norm": float(lm.x),
						"y_norm": float(lm.y),
						"z_norm": float(lm.z),
						"visibility": float(landmark_visibility(lm)),
						"x_px": float(lm.x * image_w),
						"y_px": float(lm.y * image_h),
					}

			raw_frames.append(frame_points)
			if status and frame_count and frame_idx % 10 == 0:
				status(f"YOLO + MediaPipe 포즈 수집 중... {min(100, int(frame_idx / frame_count * 100))}%")
			frame_idx += 1

	cap.release()
	pose_frames = interpolate_pose_frames(raw_frames)

	with open(out_path, "w", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow([
			"frame",
			"timestamp",
			"type",
			"name",
			"x_norm",
			"y_norm",
			"z_norm",
			"visibility",
			"x_px",
			"y_px",
			"dx_px",
			"dy_px",
			"distance_px",
			"speed_px_per_sec",
			"angle_deg",
		])

		prev_points = {}
		for frame_idx, frame_points in enumerate(pose_frames):
			timestamp = frame_idx / fps
			current_points = {}
			for name in TARGET_LANDMARKS:
				point = frame_points[name]
				x_px = int(round(point["x_px"]))
				y_px = int(round(point["y_px"]))
				current_points[name] = {
					"x_px": x_px,
					"y_px": y_px,
				}

				prev = prev_points.get(name)
				dx_px = x_px - prev["x_px"] if prev else 0
				dy_px = y_px - prev["y_px"] if prev else 0
				distance_px = math.hypot(dx_px, dy_px) if prev else 0
				speed = distance_px / dt if prev else 0
				writer.writerow([
					frame_idx,
					f"{timestamp:.4f}",
					"keypoint",
					name,
					f"{point['x_norm']:.6f}",
					f"{point['y_norm']:.6f}",
					f"{point['z_norm']:.6f}",
					f"{point['visibility']:.6f}",
					x_px,
					y_px,
					dx_px,
					dy_px,
					f"{distance_px:.4f}",
					f"{speed:.4f}",
					"0.0000",
				])

			for name, point_names in angles.items():
				a, b, c = (current_points[p] for p in point_names)
				angle = angle_deg(a, b, c)
				writer.writerow([
					frame_idx,
					f"{timestamp:.4f}",
					"angle",
					name,
					"0.000000",
					"0.000000",
					"0.000000",
					"0.000000",
					0,
					0,
					0,
					0,
					"0.0000",
					"0.0000",
					"0.0000" if angle is None else f"{angle:.4f}",
				])

			prev_points = current_points
			if status and frame_count and frame_idx % 10 == 0:
				status(f"보간된 포즈 CSV 작성 중... {min(100, int(frame_idx / frame_count * 100))}%")


class VibrotactileExtractorApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Center Person Vibrotactile Extractor")
		self.root.geometry("1040x720")

		self.video_path = None
		self.tracks = []
		self.groups = {}
		self.preview_frame = None
		self.preview_image = None
		self.preview_landmarker = None
		self.preview_fallback_landmarker = None
		self.preview_previous_box = None
		self.yolo_model = None
		self.selected_group_id = None
		self.video_cap = None
		self.video_fps = 30
		self.video_frame_count = 0
		self.current_frame_idx = 0
		self.is_playing = False
		self.deleted_groups_undo = None

		self.build_ui()
		self.root.protocol("WM_DELETE_WINDOW", self.on_close)
		self.root.bind_all("<Control-z>", self.undo_delete)
		self.root.bind_all("<Control-Z>", self.undo_delete)

	def build_ui(self):
		main = ttk.Frame(self.root, padding=16)
		main.pack(fill="both", expand=True)

		top = ttk.Frame(main)
		top.pack(fill="x")
		ttk.Button(top, text="영상 선택", command=self.choose_video).pack(side="left")
		self.video_label = ttk.Label(top, text="선택된 영상 없음")
		self.video_label.pack(side="left", padx=12)

		content = ttk.PanedWindow(main, orient="horizontal")
		content.pack(fill="both", expand=True, pady=14)

		left = ttk.Frame(content, padding=(0, 0, 12, 0))
		right = ttk.Frame(content)
		content.add(left, weight=1)
		content.add(right, weight=2)

		ttk.Label(left, text="감지된 인물 후보").pack(anchor="w")
		self.track_list = tk.Listbox(left, selectmode="browse", height=12, exportselection=False)
		self.track_list.pack(fill="both", expand=False, pady=(6, 10))
		self.track_list.bind("<<ListboxSelect>>", lambda event: self.refresh_video_frame())

		buttons = ttk.Frame(left)
		buttons.pack(fill="x")
		ttk.Button(buttons, text="가운데 인물 추출", command=self.extract_selected).pack(fill="x", pady=6)

		ttk.Label(
			left,
			text="프레임마다 화면 중앙에 가장 가까운 사람만 자동으로 선택합니다. 중앙에 있는 사람이 바뀌어도 같은 데이터 파일에 이어서 추출합니다.",
			wraplength=280,
		).pack(anchor="w", pady=(8, 0))

		self.status_label = ttk.Label(left, text="먼저 영상을 선택하세요.", wraplength=280)
		self.status_label.pack(anchor="w", pady=(16, 0))
		self.progress = ttk.Progressbar(left, mode="indeterminate")
		self.progress.pack(fill="x", pady=(8, 0))

		ttk.Label(right, text="영상 미리보기").pack(anchor="w")
		self.canvas = tk.Canvas(right, bg="#202020", highlightthickness=0)
		self.canvas.pack(fill="both", expand=True, pady=(6, 8))
		self.canvas.bind("<Configure>", lambda event: self.refresh_video_frame())

		playback = ttk.Frame(right)
		playback.pack(fill="x")
		self.play_button = ttk.Button(playback, text="Play", command=self.toggle_playback)
		self.play_button.pack(side="left")
		ttk.Button(playback, text="처음", command=self.seek_start).pack(side="left", padx=(6, 0))
		self.time_label = ttk.Label(playback, text="00:00 / 00:00")
		self.time_label.pack(side="right")

		self.frame_slider = ttk.Scale(right, from_=0, to=0, orient="horizontal", command=self.on_slider_changed)
		self.frame_slider.pack(fill="x", pady=(8, 0))

	def set_status(self, text):
		self.root.after(0, lambda: self.status_label.configure(text=text))

	def choose_video(self):
		path = filedialog.askopenfilename(
			title="추출할 영상 선택",
			filetypes=[
				("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
				("All files", "*.*"),
			],
		)
		if not path:
			return
		self.stop_playback()
		if self.preview_landmarker is not None:
			self.preview_landmarker.close()
			self.preview_landmarker = None
		if self.preview_fallback_landmarker is not None:
			self.preview_fallback_landmarker.close()
			self.preview_fallback_landmarker = None
		self.preview_previous_box = None
		self.tracks = []
		self.groups = {}
		self.video_path = Path(path)
		self.video_label.configure(text=str(self.video_path))
		self.open_playback_video()
		self.progress.start(10)
		self.set_status("인물 후보를 분석하고 있습니다...")
		threading.Thread(target=self.analyze_video, daemon=True).start()

	def analyze_video(self):
		try:
			tracks, preview_frame = detect_person_tracks(self.video_path, self.set_status)
			self.tracks = tracks
			self.groups = {track.id: [track] for track in tracks}
			self.deleted_groups_undo = None
			self.preview_frame = preview_frame
			self.root.after(0, self.populate_tracks)
			self.root.after(0, self.refresh_video_frame)
			self.set_status("가운데 인물 skeleton 분석이 끝났습니다. 바로 추출할 수 있습니다.")
		except Exception as exc:
			error = str(exc)
			self.root.after(0, lambda: messagebox.showerror("분석 실패", error))
			self.set_status("분석에 실패했습니다.")
		finally:
			self.root.after(0, self.progress.stop)

	def populate_tracks(self):
		self.track_list.delete(0, tk.END)
		for group_id, tracks in self.groups.items():
			hits = sum(track.hits for track in tracks)
			ranges = [track.frame_range() for track in tracks]
			start = min(item[0] for item in ranges)
			end = max(item[1] for item in ranges)
			label = f"center_target | skeleton 감지 {hits}회 | frame {start}-{end}"
			self.track_list.insert(tk.END, label)
		if self.groups:
			self.track_list.selection_set(0)
		self.refresh_video_frame()

	def selected_group_ids(self):
		keys = list(self.groups.keys())
		return [keys[index] for index in self.track_list.curselection()]

	def merge_selected(self):
		selected = self.selected_group_ids()
		if len(selected) < 2:
			messagebox.showinfo("merge", "merge할 인물 후보를 2개 이상 선택하세요.")
			return
		self.deleted_groups_undo = None
		target = selected[0]
		for group_id in selected[1:]:
			self.groups[target].extend(self.groups.pop(group_id))
		self.populate_tracks()
		keys = list(self.groups.keys())
		self.track_list.selection_clear(0, tk.END)
		self.track_list.selection_set(keys.index(target))
		self.refresh_video_frame()
		self.set_status(f"person_{target}에 후보 {len(selected)}개를 merge했습니다.")

	def delete_selected(self):
		selected = self.selected_group_ids()
		if not selected:
			messagebox.showinfo("delete", "삭제할 인물 후보를 선택하세요.")
			return
		self.deleted_groups_undo = {
			"group_ids": selected,
			"groups": {group_id: self.groups[group_id] for group_id in selected if group_id in self.groups},
			"selected_group_id": self.selected_group_id,
		}
		for group_id in selected:
			self.groups.pop(group_id, None)
		if self.selected_group_id in selected:
			self.selected_group_id = None
		self.populate_tracks()
		self.refresh_video_frame()
		self.set_status(f"인물 후보 {len(selected)}개를 삭제했습니다. Ctrl+Z로 되돌릴 수 있습니다.")

	def undo_delete(self, event=None):
		if not self.deleted_groups_undo:
			return "break"

		restored = self.deleted_groups_undo["groups"]
		for group_id, tracks in restored.items():
			self.groups[group_id] = tracks
		self.selected_group_id = self.deleted_groups_undo["selected_group_id"]
		self.deleted_groups_undo = None

		self.populate_tracks()
		keys = list(self.groups.keys())
		self.track_list.selection_clear(0, tk.END)
		for group_id in restored:
			if group_id in keys:
				self.track_list.selection_set(keys.index(group_id))
		self.refresh_video_frame()
		self.set_status(f"삭제한 인물 후보 {len(restored)}개를 복구했습니다.")
		return "break"

	def group_box_for_frame(self, tracks, frame_idx):
		boxes = []
		for track in tracks:
			if not track.boxes:
				continue
			nearest = min(track.boxes, key=lambda idx: abs(idx - frame_idx))
			boxes.append(track.boxes[nearest])
		return union_boxes(boxes) if boxes else None

	def overlay_groups(self, frame, frame_idx):
		if not self.tracks:
			return frame
		image_h, image_w = frame.shape[:2]
		if self.yolo_model is None:
			self.yolo_model = create_yolo_model(status=self.set_status)
		if self.preview_landmarker is None:
			self.preview_landmarker = create_pose_landmarker(mp_tasks_vision.RunningMode.IMAGE, num_poses=1, status=self.set_status)
		if self.preview_fallback_landmarker is None:
			self.preview_fallback_landmarker = create_pose_landmarker(mp_tasks_vision.RunningMode.IMAGE, status=self.set_status)
		landmarks, box, _ = select_center_pose_with_yolo(
			frame,
			self.yolo_model,
			self.preview_landmarker,
			self.preview_fallback_landmarker,
			previous_box=self.preview_previous_box,
		)
		if landmarks is None or box is None:
			return frame
		self.preview_previous_box = box
		x, y, w, h = clamp_box(box, image_w, image_h, TARGET_PADDING)
		color = (80, 220, 255)
		center_x = image_w // 2
		cv2.line(frame, (center_x, 0), (center_x, image_h), (160, 160, 160), 1, cv2.LINE_AA)
		for start, end in POSE_CONNECTIONS:
			if start >= len(landmarks) or end >= len(landmarks):
				continue
			a = landmarks[start]
			b = landmarks[end]
			if landmark_visibility(a) < MIN_LANDMARK_VISIBILITY or landmark_visibility(b) < MIN_LANDMARK_VISIBILITY:
				continue
			ax, ay = int(a.x * image_w), int(a.y * image_h)
			bx, by = int(b.x * image_w), int(b.y * image_h)
			cv2.line(frame, (ax, ay), (bx, by), color, 3, cv2.LINE_AA)
		for landmark in landmarks:
			if landmark_visibility(landmark) < MIN_LANDMARK_VISIBILITY:
				continue
			lx, ly = int(landmark.x * image_w), int(landmark.y * image_h)
			cv2.circle(frame, (lx, ly), 3, color, -1, cv2.LINE_AA)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
		cv2.putText(
			frame,
			"center target",
			(x, max(24, y - 10)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.75,
			color,
			2,
			cv2.LINE_AA,
		)
		return frame

	def refresh_video_frame(self):
		frame = self.read_frame(self.current_frame_idx)
		if frame is None and self.preview_frame is not None:
			frame = self.preview_frame.copy()
		if frame is None:
			return
		self.draw_frame(self.overlay_groups(frame, self.current_frame_idx))
		self.update_time_label()

	def open_playback_video(self):
		if self.video_cap is not None:
			self.video_cap.release()
		self.video_cap = cv2.VideoCapture(str(self.video_path))
		if not self.video_cap.isOpened():
			self.video_cap = None
			messagebox.showerror("영상 열기 실패", f"영상을 열 수 없습니다:\n{self.video_path}")
			return
		self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
		self.video_frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		self.current_frame_idx = 0
		self.frame_slider.configure(to=max(0, self.video_frame_count - 1))
		self.frame_slider.set(0)
		self.refresh_video_frame()

	def read_frame(self, frame_idx):
		if self.video_cap is None:
			return None
		frame_idx = max(0, min(frame_idx, max(0, self.video_frame_count - 1)))
		self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
		ret, frame = self.video_cap.read()
		return frame if ret else None

	def toggle_playback(self):
		if self.video_cap is None:
			return
		self.is_playing = not self.is_playing
		self.play_button.configure(text="Pause" if self.is_playing else "Play")
		if self.is_playing:
			self.playback_tick()

	def stop_playback(self):
		self.is_playing = False
		if hasattr(self, "play_button"):
			self.play_button.configure(text="Play")

	def playback_tick(self):
		if not self.is_playing:
			return
		if self.video_frame_count and self.current_frame_idx >= self.video_frame_count - 1:
			self.stop_playback()
			return
		self.current_frame_idx += 1
		self.frame_slider.set(self.current_frame_idx)
		self.refresh_video_frame()
		delay_ms = max(1, int(1000 / self.video_fps))
		self.root.after(delay_ms, self.playback_tick)

	def seek_start(self):
		self.current_frame_idx = 0
		self.frame_slider.set(0)
		self.refresh_video_frame()

	def on_slider_changed(self, value):
		if self.video_cap is None:
			return
		self.current_frame_idx = int(float(value))
		if not self.is_playing:
			self.refresh_video_frame()

	def update_time_label(self):
		current_seconds = self.current_frame_idx / self.video_fps if self.video_fps else 0
		total_seconds = self.video_frame_count / self.video_fps if self.video_fps else 0
		self.time_label.configure(text=f"{self.format_time(current_seconds)} / {self.format_time(total_seconds)}")

	def format_time(self, seconds):
		minutes = int(seconds // 60)
		seconds = int(seconds % 60)
		return f"{minutes:02d}:{seconds:02d}"

	def draw_frame(self, frame):
		canvas_w = max(1, self.canvas.winfo_width())
		canvas_h = max(1, self.canvas.winfo_height())
		h, w = frame.shape[:2]
		scale = min(canvas_w / w, canvas_h / h)
		new_w = max(1, int(w * scale))
		new_h = max(1, int(h * scale))
		resized = cv2.resize(frame, (new_w, new_h))
		rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
		ppm_header = f"P6 {new_w} {new_h} 255 ".encode("ascii")
		self.preview_image = tk.PhotoImage(data=ppm_header + rgb.tobytes(), format="PPM")
		self.canvas.delete("all")
		self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_image)

	def extract_selected(self):
		if self.video_path is None:
			messagebox.showinfo("가운데 인물 추출", "먼저 영상을 선택해주세요.")
			return
		self.selected_group_id = CENTER_TRACK_ID
		self.progress.start(10)
		self.set_status("가운데 인물 기준으로 vibrotactile 추출을 시작합니다...")
		threading.Thread(target=self.extract_worker, daemon=True).start()

	def extract_worker(self):
		try:
			selected_tracks = self.tracks
			stem = self.video_path.stem
			out_dir = Path("output") / "gui_extractions" / stem / "center_person"
			csv_path = out_dir / "pose_coords.csv"
			wav_dir = out_dir / "wav_versions"

			write_selected_person_pose_csv(self.video_path, selected_tracks, csv_path, self.set_status)
			self.set_status("WAV 파일 생성 중...")
			frames = read_pose_csv(csv_path)
			rows = generate_all_wavs(frames, wav_dir, sample_rate=44100, carrier_hz=175, gain=0.8)
			write_summary_csv(wav_dir / "wav_versions.summary.csv", rows)

			self.set_status(f"완료: {wav_dir}")
			self.root.after(
				0,
				lambda: messagebox.showinfo("완료", f"CSV와 18개 WAV를 생성했습니다.\n\n{out_dir}"),
			)
		except Exception as exc:
			error = str(exc)
			self.root.after(0, lambda: messagebox.showerror("추출 실패", error))
			self.set_status("추출에 실패했습니다.")
		finally:
			self.root.after(0, self.progress.stop)

	def on_close(self):
		self.stop_playback()
		if self.preview_landmarker is not None:
			self.preview_landmarker.close()
		if self.preview_fallback_landmarker is not None:
			self.preview_fallback_landmarker.close()
		if self.video_cap is not None:
			self.video_cap.release()
		self.root.destroy()


def main():
	root = tk.Tk()
	VibrotactileExtractorApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
