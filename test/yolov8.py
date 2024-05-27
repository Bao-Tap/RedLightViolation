import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from util import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Boundary Drawing")

        self.canvas_width = 1200
        self.canvas_height = 700
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.btn_open = tk.Button(root, text="Open Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT)

        self.btn_draw_left = tk.Button(root, text="Draw Boundary", command=self.draw_left_boundary)
        self.btn_draw_left.pack(side=tk.LEFT)

        self.btn_draw_right = tk.Button(root, text="Draw Right Boundary", command=self.draw_right_boundary)
        self.btn_draw_right.pack(side=tk.LEFT)

        self.btn_run = tk.Button(root, text="Run Detection", command=self.run_detection)
        self.btn_run.pack(side=tk.LEFT)

        self.btn_detect_red_light = tk.Button(root, text="Detect Red Light Violation", command=self.toggle_red_light_detection)
        self.btn_detect_red_light.pack(side=tk.LEFT)

        # Labels to display boundary coordinates
        self.left_coords_label = tk.Label(root, text="Left line coordinates: None")
        self.left_coords_label.pack(side=tk.LEFT)
        self.right_coords_label = tk.Label(root, text="Right line coordinates: None")
        self.right_coords_label.pack(side=tk.LEFT)

        self.cap = None
        self.frame = None
        self.left_line = None
        self.right_line = None
        self.left_coords = None
        self.right_coords = None
        self.drawing = False
        self.point1 = ()
        self.point2 = ()
        self.current_line = "left"
        self.scale_x = 1
        self.scale_y = 1

        # Load models
        self.coco_model = YOLO('yolov8n.pt')
        self.license_plate_detector = YOLO('last1.pt')
        self.tracker = DeepSort(max_age=30)
        self.vehicles_info = {}

        # Detection flag
        self.detect_red_light = False

        # Results storage
        self.results = {}
        self.frame_nmr = -1

    def open_video(self):
        video_path = filedialog.askopenfilename()
        if not video_path:
            return
        
        self.cap = cv2.VideoCapture(video_path)
        self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.resize_frame(frame, self.canvas_width, self.canvas_height)
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(self.frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

            if self.detect_red_light:
                self.frame_nmr += 1
                self.detect_and_track(frame)

            self.root.after(10, self.show_frame)

    def resize_frame(self, frame, width, height):
        h, w, _ = frame.shape
        scale = min(width / w, height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.scale_x = w / new_w
        self.scale_y = h / new_h
        resized_frame = cv2.resize(frame, (new_w, new_h))
        return resized_frame

    def draw_left_boundary(self):
        self.current_line = "left"
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def draw_right_boundary(self):
        self.current_line = "right"
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_click(self, event):
        self.point1 = (event.x, event.y)
        self.drawing = True

    def on_drag(self, event):
        if self.drawing:
            self.canvas.delete("preview_line")
            self.canvas.create_line(self.point1[0], self.point1[1], event.x, event.y, fill="green", width=2, tags="preview_line")

    def on_release(self, event):
        self.point2 = (event.x, event.y)
        self.drawing = False
        scaled_point1 = (int(self.point1[0] * self.scale_x), int(self.point1[1] * self.scale_y))
        scaled_point2 = (int(self.point2[0] * self.scale_x), int(self.point2[1] * self.scale_y))
        
        if self.current_line == "left":
            if self.left_line:
                self.canvas.delete(self.left_line)
            self.left_line = self.canvas.create_line(self.point1[0], self.point1[1], self.point2[0], self.point2[1], fill="green", width=2)
            self.left_coords = (scaled_point1, scaled_point2)
        elif self.current_line == "right":
            if self.right_line:
                self.canvas.delete(self.right_line)
            self.right_line = self.canvas.create_line(self.point1[0], self.point1[1], self.point2[0], self.point2[1], fill="blue", width=2)
            self.right_coords = (scaled_point1, scaled_point2)
        self.canvas.delete("preview_line")

    def run_detection(self):
        if not self.left_coords and not self.right_coords:
            messagebox.showwarning("Warning", "Please draw at least one boundary line.")
            return
        
        # Update labels with boundary coordinates
        self.left_coords_label.config(text=f"Left line coordinates: {self.left_coords}")
        self.right_coords_label.config(text=f"Right line coordinates: {self.right_coords}")

    def toggle_red_light_detection(self):
        self.detect_red_light = not self.detect_red_light
        if self.detect_red_light:
            self.btn_detect_red_light.config(text="Stop Red Light Detection")
        else:
            self.btn_detect_red_light.config(text="Detect Red Light Violation")

    def detect_and_track(self, frame):
        vehicles = [2, 3, 5, 7]
        self.results[self.frame_nmr] = {}
        
        # Detect vehicles
        detections = self.coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if int(class_id) in vehicles:
                detections_.append([[x1, y1, x2-x1, y2-y1], score, int(class_id)])

        # Track vehicles
        track_ids = self.tracker.update_tracks(detections_, frame=frame)
        track_id = []

        for track in track_ids:
            x1, y1, x2, y2 = track.to_tlbr()
            car_id = track.track_id
            track_id.append((x1, y1, x2, y2, car_id))

            if car_id in self.vehicles_info:
                self.vehicles_info[car_id][self.frame_nmr] = {
                    'bbox': [x1, y1, x2, y2],
                }

        # Detect license plates
        license_plates = self.license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            if int(class_id) == 0:
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_id)
                if car_id != -1:
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    height, width = license_plate_crop_gray.shape[:2]
                    new_dimensions = (width * 6, height * 6)
                    resized_image = cv2.resize(license_plate_crop_gray, new_dimensions, interpolation=cv2.INTER_LINEAR)
