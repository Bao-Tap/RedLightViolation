import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from util import *


class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Boundary Drawing")

        self.canvas_width = 1200
        self.canvas_height = 600
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.btn_open = tk.Button(root, text="Open Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT)

        self.btn_draw_left = tk.Button(root, text="Draw Boundary", command=self.draw_left_boundary)
        self.btn_draw_left.pack(side=tk.LEFT)

        self.btn_draw_right = tk.Button(root, text="Draw Right Boundary", command=self.draw_right_boundary)
        self.btn_draw_right.pack(side=tk.LEFT)

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
        self.license_plate_detector = YOLO('last.pt')
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
            self.frame = frame.copy()  # Lưu khung hình gốc
            resized_frame = self.resize_frame(frame, self.canvas_width, self.canvas_height)
            display_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

            if self.detect_red_light:
                self.frame_nmr += 1
                self.detect_and_track(self.frame)

            self.root.after(200, self.show_frame)

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
            self.left_coords_label.config(text=f"Left line coordinates: {self.left_coords}")
        elif self.current_line == "right":
            if self.right_line:
                self.canvas.delete(self.right_line)
            self.right_line = self.canvas.create_line(self.point1[0], self.point1[1], self.point2[0], self.point2[1], fill="blue", width=2)
            self.right_coords = (scaled_point1, scaled_point2)
            self.right_coords_label.config(text=f"Right line coordinates: {self.right_coords}")
        self.canvas.delete("preview_line")

    def toggle_red_light_detection(self):
        self.detect_red_light = not self.detect_red_light
        if self.detect_red_light:
            self.btn_detect_red_light.config(text="Stop Red Light Detection")
        else:
            self.btn_detect_red_light.config(text="Detect Red Light Violation")

    def detect_and_track(self, frame):
        vehicles = [2, 3, 5, 7]
        self.results[self.frame_nmr] = {}
        display_frame = frame.copy()
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
                    license_plate_crop = cv2.resize(license_plate_crop, (int((x2 - x1) * 3), int((y2 - y1) * 3)))
                    height, width = license_plate_crop_gray.shape[:2]
                    new_dimensions = (width * 5, height * 5)
                    resized_image = cv2.resize(license_plate_crop_gray, new_dimensions, interpolation=cv2.INTER_LINEAR)
                    license_plate_text, license_plate_text_score = read_license_plate(resized_image)

                    if license_plate_text is not None:
                        if car_id not in self.vehicles_info:
                            self.vehicles_info[car_id] = {'license_plate': (license_plate_text, license_plate_text_score),
                                                          'license_crop': license_plate_crop}
                        else:
                            _, current_score = self.vehicles_info[car_id]['license_plate']
                            if license_plate_text_score > current_score:
                                self.vehicles_info[car_id]['license_plate'] = (license_plate_text, license_plate_text_score)
                                self.vehicles_info[car_id]['license_crop'] = license_plate_crop

        # Check for red light violation and display results
        for track in track_ids:
            x1, y1, x2, y2 = track.to_tlbr()
            car_id = track.track_id


            if car_id in self.vehicles_info and 'license_plate' in self.vehicles_info[car_id]:
                license_plate_text, _ = self.vehicles_info[car_id]['license_plate']
                license_plate_crop = self.vehicles_info[car_id]['license_crop']
                if license_plate_text is not None:
                    # Calculate the position to place the license plate crop
                    H, W ,_= license_plate_crop.shape
                    license_crop = cv2.resize(license_plate_crop, (W, H))
                    # Ensure the region in frame matches the size of license_crop
                    region_height = min(H, display_frame.shape[0] - int(y1))
                    region_width = min(W, display_frame.shape[1] - int((x2 + x1 - W) / 2))
                    
                    if region_height > 0 and region_width > 0:
                        # Place the license plate crop on the frame
                        display_frame[int(y1):int(y1 + region_height), int((x2 + x1 - W) / 2):int((x2 + x1 + region_width) / 2), :] = license_crop[:region_height, :region_width]
                        

                    # Draw a white rectangle above the license plate for the text
                    display_frame[int(y1) - int(H / 2):int(y1), int((x2 + x1 - W) / 2):int((x2 + x1 + W) / 2), :] = (255, 255, 255)

                    # Calculate the text size and position
                    (text_width, text_height), _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                    text_x = int((x2 + x1 - text_width) / 2)
                    text_y = int(y1 - H / 8 - (text_height / 2))

                    # Put the text on the frame
                    cv2.putText(frame, license_plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

        # Check if the car crosses the left boundary line (red light violation)
        if self.left_coords and self.right_coords:
            left_line_x1, left_line_y1 = self.left_coords[0]
            left_line_x2, left_line_y2 = self.left_coords[1]
            right_line_x1, right_line_y1 = self.right_coords[0]
            right_line_x2, right_line_y2 = self.right_coords[1]

            if y2 > left_line_y1 and y1 < left_line_y2:
                if (x1 < left_line_x1 and x2 > left_line_x2) or (x1 < right_line_x1 and x2 > right_line_x2):
                    cv2.putText(frame, "Red Light Violation", (int(x1), int(y2) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Display the frame with annotations
        # Display the frame with annotations
        display_frame_resized = self.resize_frame(display_frame, self.canvas_width, self.canvas_height)
        display_frame_rgb = cv2.cvtColor(display_frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
