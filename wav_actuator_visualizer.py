import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np


ACTUATORS = [
	("head", "Head"),
	("left_arm", "Left Arm"),
	("left_leg", "Left Leg"),
	("right_arm", "Right Arm"),
	("right_leg", "Right Leg"),
]


@dataclass
class ActuatorWav:
	path: Path
	sample_rate: int
	samples: np.ndarray
	peak_rms: float

	@property
	def duration(self):
		if self.sample_rate <= 0:
			return 0.0
		return len(self.samples) / self.sample_rate

	def amplitude_at(self, time_sec, window_ms=35):
		if self.samples.size == 0 or self.sample_rate <= 0:
			return 0.0
		center = int(time_sec * self.sample_rate)
		if center < 0 or center >= len(self.samples):
			return 0.0

		half_window = max(1, int(self.sample_rate * window_ms / 1000 / 2))
		start = max(0, center - half_window)
		end = min(len(self.samples), center + half_window)
		window = self.samples[start:end]
		if window.size == 0:
			return 0.0

		rms = float(np.sqrt(np.mean(window * window)))
		if self.peak_rms <= 0:
			return 0.0
		return max(0.0, min(1.0, rms / self.peak_rms))


def read_wav(path):
	with wave.open(str(path), "rb") as wav_file:
		channels = wav_file.getnchannels()
		sample_width = wav_file.getsampwidth()
		sample_rate = wav_file.getframerate()
		frame_count = wav_file.getnframes()
		raw = wav_file.readframes(frame_count)

	if sample_width == 1:
		data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
		data = (data - 128.0) / 128.0
	elif sample_width == 2:
		data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
	elif sample_width == 4:
		data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
	else:
		raise ValueError(f"Unsupported WAV sample width: {sample_width * 8} bit")

	if channels > 1:
		data = data.reshape(-1, channels).mean(axis=1)

	peak_rms = compute_peak_rms(data, sample_rate)
	return ActuatorWav(path=Path(path), sample_rate=sample_rate, samples=data, peak_rms=peak_rms)


def compute_peak_rms(samples, sample_rate, window_ms=35):
	if samples.size == 0 or sample_rate <= 0:
		return 0.0
	window_size = max(1, int(sample_rate * window_ms / 1000))
	step = max(1, window_size // 2)
	peak = 0.0
	for start in range(0, len(samples), step):
		window = samples[start:start + window_size]
		if window.size == 0:
			continue
		peak = max(peak, float(np.sqrt(np.mean(window * window))))
	return peak


class ActuatorVisualizerApp:
	def __init__(self, root):
		self.root = root
		self.root.title("WAV Actuator Visualizer")
		self.root.geometry("1040x640")
		self.root.minsize(860, 520)

		self.wavs = {actuator_id: None for actuator_id, _ in ACTUATORS}
		self.file_labels = {}
		self.status_var = tk.StringVar(value="Select WAV files for each actuator.")
		self.time_var = tk.StringVar(value="00:00.00 / 00:00.00")
		self.progress_var = tk.DoubleVar(value=0.0)
		self.is_playing = False
		self.paused_at = 0.0
		self.started_at = 0.0

		self.build_layout()
		self.draw_actuators({})

	def build_layout(self):
		self.root.columnconfigure(0, weight=0)
		self.root.columnconfigure(1, weight=1)
		self.root.rowconfigure(0, weight=1)

		left = ttk.Frame(self.root, padding=(18, 18, 12, 18))
		left.grid(row=0, column=0, sticky="nsw")
		left.columnconfigure(0, weight=1)

		right = ttk.Frame(self.root, padding=(12, 18, 18, 18))
		right.grid(row=0, column=1, sticky="nsew")
		right.columnconfigure(0, weight=1)
		right.rowconfigure(0, weight=1)

		self.build_file_panel(left)
		self.build_canvas_panel(right)

	def build_file_panel(self, parent):
		title = ttk.Label(parent, text="Actuator WAV Files", font=("Segoe UI", 14, "bold"))
		title.grid(row=0, column=0, sticky="w", pady=(0, 14))

		for row, (actuator_id, label) in enumerate(ACTUATORS, start=1):
			frame = ttk.Frame(parent)
			frame.grid(row=row, column=0, sticky="ew", pady=7)
			frame.columnconfigure(1, weight=1)

			ttk.Label(frame, text=label, width=10).grid(row=0, column=0, sticky="w")
			file_label = ttk.Label(frame, text="No file selected", width=28)
			file_label.grid(row=0, column=1, sticky="ew", padx=(8, 8))
			self.file_labels[actuator_id] = file_label

			ttk.Button(frame, text="Browse", command=lambda key=actuator_id: self.select_wav(key)).grid(row=0, column=2, padx=(0, 5))
			ttk.Button(frame, text="Clear", command=lambda key=actuator_id: self.clear_wav(key)).grid(row=0, column=3)

		controls = ttk.Frame(parent)
		controls.grid(row=len(ACTUATORS) + 1, column=0, sticky="ew", pady=(24, 8))
		for col in range(3):
			controls.columnconfigure(col, weight=1)

		ttk.Button(controls, text="Play", command=self.play).grid(row=0, column=0, sticky="ew", padx=(0, 5))
		ttk.Button(controls, text="Pause", command=self.pause).grid(row=0, column=1, sticky="ew", padx=5)
		ttk.Button(controls, text="Stop", command=self.stop).grid(row=0, column=2, sticky="ew", padx=(5, 0))

		ttk.Label(parent, textvariable=self.time_var).grid(row=len(ACTUATORS) + 2, column=0, sticky="w", pady=(12, 2))
		ttk.Progressbar(parent, variable=self.progress_var, maximum=1.0).grid(row=len(ACTUATORS) + 3, column=0, sticky="ew")
		ttk.Label(parent, textvariable=self.status_var, wraplength=340).grid(row=len(ACTUATORS) + 4, column=0, sticky="ew", pady=(16, 0))

	def build_canvas_panel(self, parent):
		self.canvas = tk.Canvas(parent, bg="#10131a", highlightthickness=0)
		self.canvas.grid(row=0, column=0, sticky="nsew")
		self.canvas.bind("<Configure>", lambda _event: self.draw_actuators(self.current_amplitudes()))

	def select_wav(self, actuator_id):
		path = filedialog.askopenfilename(
			title=f"Select WAV for {self.label_for(actuator_id)}",
			filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
		)
		if not path:
			return
		try:
			self.wavs[actuator_id] = read_wav(path)
		except Exception as exc:
			messagebox.showerror("Cannot load WAV", str(exc))
			return

		self.file_labels[actuator_id].configure(text=Path(path).name)
		self.status_var.set(f"Loaded {Path(path).name} for {self.label_for(actuator_id)}.")
		self.stop(redraw=False)
		self.draw_actuators(self.current_amplitudes())

	def clear_wav(self, actuator_id):
		self.wavs[actuator_id] = None
		self.file_labels[actuator_id].configure(text="No file selected")
		self.status_var.set(f"Cleared {self.label_for(actuator_id)}.")
		self.stop(redraw=False)
		self.draw_actuators(self.current_amplitudes())

	def play(self):
		if self.total_duration() <= 0:
			messagebox.showinfo("No WAV files", "Select at least one WAV file before playing.")
			return
		if self.is_playing:
			return
		self.is_playing = True
		self.started_at = time.perf_counter() - self.paused_at
		self.status_var.set("Playing actuator visualization.")
		self.update_animation()

	def pause(self):
		if not self.is_playing:
			return
		self.paused_at = self.elapsed_time()
		self.is_playing = False
		self.status_var.set("Paused.")

	def stop(self, redraw=True):
		self.is_playing = False
		self.paused_at = 0.0
		self.progress_var.set(0.0)
		self.update_time_label(0.0)
		if redraw:
			self.draw_actuators({})
			self.status_var.set("Stopped.")

	def update_animation(self):
		if not self.is_playing:
			return
		elapsed = self.elapsed_time()
		duration = self.total_duration()
		if elapsed >= duration:
			self.stop()
			return

		self.paused_at = elapsed
		self.progress_var.set(elapsed / duration if duration else 0.0)
		self.update_time_label(elapsed)
		self.draw_actuators(self.current_amplitudes(elapsed))
		self.root.after(16, self.update_animation)

	def elapsed_time(self):
		return time.perf_counter() - self.started_at

	def current_amplitudes(self, elapsed=None):
		if elapsed is None:
			elapsed = self.paused_at
		return {
			actuator_id: wav.amplitude_at(elapsed) if wav else 0.0
			for actuator_id, wav in self.wavs.items()
		}

	def draw_actuators(self, amplitudes):
		width = max(1, self.canvas.winfo_width())
		height = max(1, self.canvas.winfo_height())
		self.canvas.delete("all")

		cx = width / 2
		cy = height / 2
		orbit = min(width, height) * 0.29
		positions = {
			"head": (cx, cy - orbit * 0.78),
			"left_arm": (cx - orbit * 0.88, cy - orbit * 0.12),
			"left_leg": (cx - orbit * 0.55, cy + orbit * 0.72),
			"right_arm": (cx + orbit * 0.88, cy - orbit * 0.12),
			"right_leg": (cx + orbit * 0.55, cy + orbit * 0.72),
		}

		self.canvas.create_oval(
			cx - orbit,
			cy - orbit,
			cx + orbit,
			cy + orbit,
			outline="#2a3445",
			width=2,
		)

		for actuator_id, label in ACTUATORS:
			x, y = positions[actuator_id]
			amp = amplitudes.get(actuator_id, 0.0)
			self.draw_actuator(x, y, label, amp, self.wavs[actuator_id] is not None)

	def draw_actuator(self, x, y, label, amplitude, loaded):
		base_radius = 42
		pulse = base_radius + amplitude * 13
		glow = base_radius + amplitude * 38
		intensity = int(70 + amplitude * 185)
		fill = f"#{intensity:02x}{min(255, intensity + 28):02x}{min(255, intensity + 55):02x}" if loaded else "#303846"
		outline = "#dff7ff" if loaded else "#5b6574"

		if amplitude > 0.01:
			self.canvas.create_oval(
				x - glow,
				y - glow,
				x + glow,
				y + glow,
				fill="#244d67",
				outline="",
				stipple="gray25",
			)

		self.canvas.create_oval(
			x - pulse,
			y - pulse,
			x + pulse,
			y + pulse,
			fill=fill,
			outline=outline,
			width=3,
		)
		self.canvas.create_text(x, y, text=label, fill="#f7fbff", font=("Segoe UI", 11, "bold"))

	def total_duration(self):
		return max((wav.duration for wav in self.wavs.values() if wav), default=0.0)

	def update_time_label(self, elapsed):
		self.time_var.set(f"{format_time(elapsed)} / {format_time(self.total_duration())}")

	def label_for(self, actuator_id):
		return next(label for key, label in ACTUATORS if key == actuator_id)


def format_time(seconds):
	minutes = int(seconds // 60)
	remaining = seconds - minutes * 60
	return f"{minutes:02d}:{remaining:05.2f}"


def main():
	root = tk.Tk()
	ActuatorVisualizerApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
