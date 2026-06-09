import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp

from gui_vibrotactile_extractor import main as run_gui


def run_pose_csv(video_path: str, output_csv: str, show_window: bool):
	mp_drawing = mp.solutions.drawing_utils
	mp_pose = mp.solutions.pose

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video file: {video_path}")

	out_path = Path(output_csv)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	with open(out_path, "w", newline="", encoding="utf-8") as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow([
			"frame",
			"timestamp",
			"landmark_index",
			"x_norm",
			"y_norm",
			"z_norm",
			"visibility",
			"x_px",
			"y_px",
		])

		with mp_pose.Pose(
			static_image_mode=False,
			model_complexity=1,
			enable_segmentation=False,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
		) as pose:
			frame_idx = 0
			start_time = time.time()
			while True:
				ret, frame = cap.read()
				if not ret:
					break

				image_h, image_w = frame.shape[:2]
				image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				results = pose.process(image_rgb)

				if results.pose_landmarks:
					for i, lm in enumerate(results.pose_landmarks.landmark):
						x_px = int(lm.x * image_w)
						y_px = int(lm.y * image_h)
						timestamp = time.time() - start_time
						csv_writer.writerow([
							frame_idx,
							f"{timestamp:.4f}",
							i,
							f"{lm.x:.6f}",
							f"{lm.y:.6f}",
							f"{lm.z:.6f}",
							f"{getattr(lm, 'visibility', '')}",
							x_px,
							y_px,
						])

					mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

				if show_window:
					cv2.imshow("Pose Tracking", frame)
					key = cv2.waitKey(1) & 0xFF
					if key == 27 or key == ord("q"):
						break

				frame_idx += 1

	cap.release()
	cv2.destroyAllWindows()


def parse_args():
	parser = argparse.ArgumentParser(description="Vibrotactile pose extractor")
	parser.add_argument("--cli", action="store_true", help="Run the legacy single-video CSV extractor instead of the GUI")
	parser.add_argument("--video", "-v", help="Video path for --cli mode")
	parser.add_argument("--output", "-o", default="output/pose_coords.csv", help="Output CSV path for --cli mode")
	parser.add_argument("--no-window", dest="show_window", action="store_false", help="Do not show preview window in --cli mode")
	parser.set_defaults(show_window=True)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.cli:
		if not args.video:
			raise SystemExit("--cli mode requires --video")
		run_pose_csv(args.video, args.output, args.show_window)
	else:
		run_gui()
