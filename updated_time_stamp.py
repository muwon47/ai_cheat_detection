import cv2
import math
import cvzone
import torch
from ultralytics import YOLO
import os
import json
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import time
from datetime import timedelta
import numpy as np


class ExamReviewSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Exam Footage Review System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.model_path = r"C:\Users\ajswa\PycharmProjects\PythonProject2\cheat_detection\ajinkya_yolol_bsz6_70epochs.pt"
        self.classNames = ['chit', 'phone', 'hand', 'peeking', 'supplement-passing']
        self.class_colors = {
            'chit': (255, 0, 0),
            'phone': (0, 0, 255),
            'hand': (0, 255, 0),
            'peeking': (255, 255, 0),
            'supplement-passing': (255, 0, 255)
        }

        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame = 0
        self.playing = False
        self.after_id = None
        self.detection_timestamps = {}

        self.display_width = 600
        self.display_height = 400

        self.setup_ui()
        self.load_model()

    def load_model(self):

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.status_var.set(f"Model loaded successfully on {self.device}")
        except Exception as e:
            self.status_var.set(f"Error loading model: {e}")

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_frame = tk.Frame(main_frame, bg="#f0f0f0")
        top_frame.pack(fill=tk.X, pady=5)

        open_btn = tk.Button(top_frame, text="Open Video", command=self.open_video,
                             bg="#4CAF50", fg="white", padx=10)
        open_btn.pack(side=tk.LEFT, padx=5)

        process_btn = tk.Button(top_frame, text="Process Video", command=self.process_video,
                                bg="#2196F3", fg="white", padx=10)
        process_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = tk.Label(top_frame, textvariable=self.status_var, bg="#f0f0f0", anchor="w")
        status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.video_frame = tk.Label(main_frame, bg="black",
                                    width=self.display_width,
                                    height=self.display_height)
        self.video_frame.pack_propagate(False)
        self.video_frame.pack(pady=10)

        progress_frame = tk.Frame(main_frame, bg="#f0f0f0")
        progress_frame.pack(fill=tk.X, pady=5)

        self.time_var = tk.StringVar()
        self.time_var.set("00:00 / 00:00")
        time_label = tk.Label(progress_frame, textvariable=self.time_var, bg="#f0f0f0")
        time_label.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Scale(progress_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)


        controls_frame = tk.Frame(main_frame, bg="#f0f0f0")
        controls_frame.pack(fill=tk.X, pady=5)

        play_btn = tk.Button(controls_frame, text="Play/Pause", command=self.toggle_play,
                             bg="#555555", fg="white", padx=10)
        play_btn.pack(side=tk.LEFT, padx=5)

        timestamps_frame = tk.LabelFrame(main_frame, text="Detection Timestamps",
                                         bg="#f0f0f0", padx=10, pady=10)
        timestamps_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        timestamps_container = tk.Frame(timestamps_frame, bg="#f0f0f0")
        timestamps_container.pack(fill=tk.BOTH, expand=True)

        scrollbar_y = tk.Scrollbar(timestamps_container, orient="vertical")
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        scrollbar_x = tk.Scrollbar(timestamps_container, orient="horizontal")
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        style = ttk.Style()
        style.configure("Treeview", foreground="black", background="white", fieldbackground="white")

        self.timestamps_tree = ttk.Treeview(timestamps_container,
                                            columns=("timestamp", "class", "confidence"),
                                            show="headings",
                                            yscrollcommand=scrollbar_y.set,
                                            xscrollcommand=scrollbar_x.set)
        self.timestamps_tree.heading("timestamp", text="Timestamp")
        self.timestamps_tree.heading("class", text="Detected Class")
        self.timestamps_tree.heading("confidence", text="Confidence")
        self.timestamps_tree.column("timestamp", width=200, stretch=True)
        self.timestamps_tree.column("class", width=200, stretch=True)
        self.timestamps_tree.column("confidence", width=100, stretch=True)
        self.timestamps_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar_y.config(command=self.timestamps_tree.yview)
        scrollbar_x.config(command=self.timestamps_tree.xview)

        self.timestamps_tree.bind("<Double-1>", self.jump_to_timestamp)

        filter_frame = tk.Frame(timestamps_frame, bg="#f0f0f0", pady=5)
        filter_frame.pack(fill=tk.X)

        tk.Label(filter_frame, text="Filter by class:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)

        self.filter_var = tk.StringVar()
        self.filter_var.set("All")
        filter_options = ["All"] + self.classNames
        filter_menu = ttk.Combobox(filter_frame, textvariable=self.filter_var,
                                   values=filter_options, state="readonly", width=15)
        filter_menu.pack(side=tk.LEFT, padx=5)
        filter_menu.bind("<<ComboboxSelected>>", self.apply_filter)





    def open_video(self):

        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not video_path:
            return

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame = 0
        self.playing = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        video_name = os.path.basename(video_path)
        duration = self.total_frames / self.fps
        self.status_var.set(f"Loaded: {video_name} ({duration:.2f} seconds)")

        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        self.update_time_display()

        json_path = os.path.splitext(video_path)[0] + "_timestamps.json"
        if os.path.exists(json_path):
            self.load_timestamps(json_path)
        else:
            self.detection_timestamps = {}
            self.timestamps_tree.delete(*self.timestamps_tree.get_children())

    def process_video(self):

        if self.cap is None:
            self.status_var.set("Please open a video first.")
            return
        threading.Thread(target=self.process_video_thread, daemon=True).start()

    def process_video_thread(self):

        self.status_var.set("Processing video...")
        self.root.update()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        self.detection_timestamps = {}

        frame_count = 0
        detection_count = 0
        total_frames = self.total_frames

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            progress_percent = (frame_count / total_frames) * 100
            self.status_var.set(f"Processing: {progress_percent:.1f}% complete")
            self.root.update()


            results = self.model(frame, stream=True, conf=0.34)

            timestamp_sec = frame_count / self.fps
            timestamp_str = "{:.2f}".format(timestamp_sec)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence < 0.34:
                        continue
                    class_id = int(box.cls[0])
                    if class_id < len(self.classNames):
                        class_name = self.classNames[class_id]
                        xyxy = list(map(int, box.xyxy[0]))
                        detection = {
                            "class": class_name,
                            "confidence": confidence,
                            "frame": frame_count,
                            "bbox": xyxy
                        }

                        if timestamp_str not in self.detection_timestamps:
                            self.detection_timestamps[timestamp_str] = []
                        self.detection_timestamps[timestamp_str].append(detection)
                        detection_count += 1

            frame_count += 1

        json_path = os.path.splitext(self.video_path)[0] + "_timestamps.json"
        with open(json_path, 'w') as f:
            json.dump(self.detection_timestamps, f)

        self.update_timestamps_tree()
        if not self.detection_timestamps:
            self.timestamps_tree.insert("", "end", values=("No detections", "", ""))
        self.status_var.set(f"Processing complete. Found {detection_count} detections.")
        print("Detection timestamps keys:", list(self.detection_timestamps.keys()))

    def update_timestamps_tree(self):

        self.timestamps_tree.delete(*self.timestamps_tree.get_children())
        if not self.detection_timestamps:
            self.timestamps_tree.insert("", "end", values=("No detections", "", ""))
            return

        for timestamp, detections in self.detection_timestamps.items():
            for detection in detections:
                class_name = detection["class"]
                confidence = detection["confidence"]
                self.timestamps_tree.insert("", "end", values=(timestamp, class_name, f"{confidence:.2f}"))

    def apply_filter(self, event=None):

        selected_class = self.filter_var.get()
        self.timestamps_tree.delete(*self.timestamps_tree.get_children())

        for timestamp, detections in self.detection_timestamps.items():
            for detection in detections:
                class_name = detection["class"]
                confidence = detection["confidence"]
                if selected_class == "All" or selected_class == class_name:
                    self.timestamps_tree.insert("", "end", values=(timestamp, class_name, f"{confidence:.2f}"))

    def jump_to_timestamp(self, event):

        selected_item = self.timestamps_tree.focus()
        if not selected_item:
            return

        timestamp_str = self.timestamps_tree.item(selected_item)["values"][0]
        try:
            seconds = float(timestamp_str)
        except ValueError:
            try:
                h, m, s = map(int, timestamp_str.split(':'))
                seconds = h * 3600 + m * 60 + s
            except Exception as e:
                self.status_var.set(f"Invalid timestamp format: {timestamp_str}")
                return

        frame_number = int(seconds * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number

        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            progress_value = (self.current_frame / self.total_frames) * 100
            self.progress.set(progress_value)
            self.update_time_display()

    def load_timestamps(self, json_path):

        try:
            with open(json_path, 'r') as f:
                self.detection_timestamps = json.load(f)
            self.update_timestamps_tree()
            self.status_var.set(f"Loaded existing timestamps from {os.path.basename(json_path)}")
        except Exception as e:
            self.status_var.set(f"Error loading timestamps: {e}")

    def display_frame(self, frame):

        frame_with_boxes = frame.copy()
        threshold = 2
        for ts, detections in self.detection_timestamps.items():
            for detection in detections:
                det_frame = detection.get("frame", -1)
                if abs(det_frame - self.current_frame) <= threshold:
                    bbox = detection.get("bbox", None)
                    if bbox is not None:
                        class_name = detection.get("class", "")
                        color = self.class_colors.get(class_name, (255, 255, 255))
                        cv2.rectangle(frame_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        label = f"{class_name} {detection.get('confidence', 0):.2f}"
                        cv2.putText(frame_with_boxes, label, (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        rgb_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (self.display_width, self.display_height))
        img = Image.fromarray(resized)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_frame.configure(image=img_tk)
        self.video_frame.image = img_tk

    def toggle_play(self):

        if self.playing:
            self.playing = False
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None
        else:
            self.playing = True
            self.play_video_frame()

    def play_video_frame(self):

        if not self.playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            self.current_frame = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        self.display_frame(frame)
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0

        progress_value = (self.current_frame / self.total_frames) * 100
        self.progress.set(progress_value)
        self.update_time_display()

        delay_ms = int(1000 / self.fps)
        self.after_id = self.root.after(delay_ms, self.play_video_frame)

    def seek(self, value):

        if self.cap is None:
            return

        frame_number = int((float(value) / 100) * self.total_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number

        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.update_time_display()

    def update_time_display(self):
        if self.cap is None:
            return
        current_time = self.current_frame / self.fps
        total_time = self.total_frames / self.fps

        current_str = str(timedelta(seconds=int(current_time)))
        total_str = str(timedelta(seconds=int(total_time)))
        self.time_var.set(f"{current_str} / {total_str}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ExamReviewSystem(root)
    root.mainloop()
