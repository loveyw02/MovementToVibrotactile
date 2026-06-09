import argparse
import csv
import math
import wave
from pathlib import Path


BODY_PARTS = {
	"head": {
		"movement_keypoints": ("nose",),
		"angle": "head_orientation",
	},
	"left_arm": {
		"movement_keypoints": ("left_shoulder", "left_elbow", "left_wrist"),
		"angle": "left_elbow",
	},
	"right_arm": {
		"movement_keypoints": ("right_shoulder", "right_elbow", "right_wrist"),
		"angle": "right_elbow",
	},
	"left_leg": {
		"movement_keypoints": ("left_hip", "left_knee", "left_ankle"),
		"angle": "left_knee",
	},
	"right_leg": {
		"movement_keypoints": ("right_hip", "right_knee", "right_ankle"),
		"angle": "right_knee",
	},
}

METHODS = {
	"angle_delta": "Angle difference -> amplitude",
	"movement": "Movement amount -> amplitude",
	"movement_delta": "Movement change amount -> amplitude",
}

METHOD_DIRS = {
	"angle_delta": "angle",
	"movement": "movement",
	"movement_delta": "movement_delta",
}


def read_pose_csv(csv_path):
	frames = {}
	with open(csv_path, newline="", encoding="utf-8") as f:
		for row in csv.DictReader(f):
			frame = int(row["frame"])
			entry = frames.setdefault(
				frame,
				{
					"timestamp": float(row["timestamp"]),
					"keypoints": {},
					"angles": {},
				},
			)

			if row["type"] == "keypoint":
				entry["keypoints"][row["name"]] = {
					"x_px": float(row["x_px"] or 0),
					"y_px": float(row["y_px"] or 0),
					"distance_px": float(row["distance_px"] or 0),
				}
			elif row["type"] == "angle" and row["angle_deg"]:
				entry["angles"][row["name"]] = float(row["angle_deg"])

	ordered = [frames[i] for i in sorted(frames)]
	add_head_orientation_angles(ordered)
	return ordered


def add_head_orientation_angles(frames):
	for frame in frames:
		points = frame["keypoints"]
		if not all(name in points for name in ("nose", "left_shoulder", "right_shoulder")):
			continue

		shoulder_x = (points["left_shoulder"]["x_px"] + points["right_shoulder"]["x_px"]) / 2
		shoulder_y = (points["left_shoulder"]["y_px"] + points["right_shoulder"]["y_px"]) / 2
		nose = points["nose"]
		dx = nose["x_px"] - shoulder_x
		dy = nose["y_px"] - shoulder_y
		frame["angles"]["head_orientation"] = math.degrees(math.atan2(dy, dx))


def frame_dt(frames):
	if len(frames) < 2:
		return 1 / 30
	return max(1 / 240, frames[1]["timestamp"] - frames[0]["timestamp"])


def circular_angle_delta(current, previous):
	delta = (current - previous + 180) % 360 - 180
	return abs(delta)


def average_available(values, names, label):
	missing = [name for name in names if name not in values]
	if missing:
		raise ValueError(f"Missing {label} in CSV: {', '.join(missing)}")
	return sum(values[name] for name in names) / len(names)


def raw_series_for_part(frames, part_name, method):
	part = BODY_PARTS[part_name]
	if method == "movement":
		return [
			average_available(
				{
					name: point["distance_px"]
					for name, point in frame["keypoints"].items()
				},
				part["movement_keypoints"],
				"keypoints",
			)
			for frame in frames
		]

	if method == "movement_delta":
		movement_values = raw_series_for_part(frames, part_name, "movement")
		return absolute_deltas(movement_values)

	if method == "angle_delta":
		angle_name = part["angle"]
		angles = []
		for frame in frames:
			if angle_name not in frame["angles"]:
				raise ValueError(f"Missing angle in CSV: {angle_name}")
			angles.append(frame["angles"][angle_name])
		return angle_deltas(angles, circular=angle_name == "head_orientation")

	raise ValueError(f"Unknown method: {method}")


def absolute_deltas(values):
	if not values:
		return []
	return [0.0] + [abs(values[i] - values[i - 1]) for i in range(1, len(values))]


def angle_deltas(values, circular=False):
	if not values:
		return []
	deltas = [0.0]
	for i in range(1, len(values)):
		if circular:
			deltas.append(circular_angle_delta(values[i], values[i - 1]))
		else:
			deltas.append(abs(values[i] - values[i - 1]))
	return deltas


def normalize(values):
	max_value = max(values, default=0)
	if max_value <= 0:
		return [0.0 for _ in values]
	return [min(1.0, value / max_value) for value in values]


def average_series(series_list):
	if not series_list:
		return []
	length = min(len(series) for series in series_list)
	if length <= 0:
		return []
	return [
		sum(series[i] for series in series_list) / len(series_list)
		for i in range(length)
	]


def write_mono_wav(amplitudes, out_path, sample_rate, frame_seconds, carrier_hz, gain):
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	samples_per_frame = max(1, round(sample_rate * frame_seconds))
	phase = 0.0
	phase_step = 2 * math.pi * carrier_hz / sample_rate

	with wave.open(str(out_path), "wb") as wav:
		wav.setnchannels(1)
		wav.setsampwidth(2)
		wav.setframerate(sample_rate)

		for amp in amplitudes:
			sample_bytes = bytearray()
			for _ in range(samples_per_frame):
				value = int(math.sin(phase) * amp * gain * 32767)
				sample_bytes += value.to_bytes(2, "little", signed=True)
				phase += phase_step
			wav.writeframesraw(sample_bytes)


def write_summary_csv(summary_path, rows):
	with open(summary_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([
			"file",
			"method",
			"method_description",
			"body_part",
			"source",
			"max_raw_value",
			"max_amplitude_in_wav",
		])
		writer.writerows(rows)


def source_description(part_name, method):
	part = BODY_PARTS[part_name]
	if method == "angle_delta":
		return part["angle"]
	return "+".join(part["movement_keypoints"])


def generate_all_wavs(frames, output_dir, sample_rate, carrier_hz, gain):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	rows = []
	dt = frame_dt(frames)

	for method, method_description in METHODS.items():
		method_dir_name = METHOD_DIRS[method]
		method_dir = output_dir / method_dir_name
		method_amplitudes = []
		method_raw_values = []
		for part_name in BODY_PARTS:
			raw_values = raw_series_for_part(frames, part_name, method)
			amplitudes = normalize(raw_values)
			method_raw_values.append(raw_values)
			method_amplitudes.append(amplitudes)
			file_name = f"{part_name}.wav"
			out_path = method_dir / file_name
			write_mono_wav(amplitudes, out_path, sample_rate, dt, carrier_hz, gain)
			rows.append([
				str(Path(method_dir_name) / file_name),
				method,
				method_description,
				part_name,
				source_description(part_name, method),
				f"{max(raw_values, default=0):.6f}",
				f"{max(amplitudes, default=0):.6f}",
			])

		average_amplitudes = average_series(method_amplitudes)
		average_raw_values = average_series(method_raw_values)
		average_file_name = "average.wav"
		write_mono_wav(average_amplitudes, method_dir / average_file_name, sample_rate, dt, carrier_hz, gain)
		rows.append([
			str(Path(method_dir_name) / average_file_name),
			method,
			method_description,
			"average",
			"average of all body-part wav amplitudes",
			f"{max(average_raw_values, default=0):.6f}",
			f"{max(average_amplitudes, default=0):.6f}",
		])

	return rows


def parse_args():
	parser = argparse.ArgumentParser(description="Convert pose CSV into body-part actuator WAV files plus method average WAVs")
	parser.add_argument("--input", "-i", default="output/pose_coords.csv", help="Input pose CSV path")
	parser.add_argument("--output-dir", "-o", default="output/wav_versions", help="Output directory")
	parser.add_argument("--sample-rate", type=int, default=44100, help="WAV sample rate")
	parser.add_argument("--carrier-hz", type=float, default=175, help="Actuator carrier frequency")
	parser.add_argument("--gain", type=float, default=0.8, help="Output gain from 0.0 to 1.0")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	frames = read_pose_csv(args.input)
	if not frames:
		raise RuntimeError(f"No rows found in {args.input}")

	summary_rows = generate_all_wavs(
		frames,
		args.output_dir,
		args.sample_rate,
		args.carrier_hz,
		args.gain,
	)
	summary_path = Path(args.output_dir) / "wav_versions.summary.csv"
	write_summary_csv(summary_path, summary_rows)

	print(f"WAV files written: {len(summary_rows)}")
	print(f"Output directory: {args.output_dir}")
	print(f"Summary written: {summary_path}")
