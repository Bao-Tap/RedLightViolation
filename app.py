import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import os
import csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from util import *
from datetime import datetime

class TrafficViolationApp:
    def __init__(self, root):
        self.root = root

        self.violations_frame = ttk.LabelFrame(root, text="Detected Violations")
        self.violations_frame.grid(row=0, column=1, pady=10, sticky="nsew")

        # Cấu hình lưới bên trong violations_frame
        self.violations_frame.grid_rowconfigure(0, weight=1)
        self.violations_frame.grid_columnconfigure(0, weight=1)

        # Tạo treeview để hiển thị danh sách vi phạm
        self.tree = ttk.Treeview(self.violations_frame, columns=("ID", "License Plate", "Time"), show="headings")
        self.tree.heading("ID", text="Car ID")
        self.tree.heading("License Plate", text="License Plate")
        self.tree.heading("Time", text="Time")
        self.tree.grid(row=0, column=0, sticky="nsew")

        # Thêm scrollbar cho treeview
        self.scrollbar = ttk.Scrollbar(self.violations_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=self.scrollbar.set)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # Tạo frame cho chi tiết vi phạm
        self.details_frame = ttk.LabelFrame(root, text="Violation Details")
        self.details_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.details_frame.grid_rowconfigure(0, weight=1)
        # Tạo canvas để hiển thị hình ảnh biển số và đèn tín hiệu giao thông
        self.license_plate_canvas = tk.Canvas(self.details_frame, width=200, height=100)
        self.license_plate_canvas.grid(row=0, column=0, padx=10, pady=10)
        self.red_light_canvas = tk.Canvas(self.details_frame, width=100, height=200)
        self.red_light_canvas.grid(row=0, column=1, padx=10, pady=10)
        self.car_canvas = tk.Canvas(self.details_frame, width=200, height=200)
        self.car_canvas.grid(row=0, column=2, padx=10, pady=10)

        # Tạo label để hiển thị biển số
        self.license_plate_label = ttk.Label(self.details_frame, text="License Plate:", font=("Helvetica", 15, "bold"))
        self.license_plate_label.grid(row=1, column=0, padx=10, pady=10)

        self.license_plate_text = ttk.Label(self.details_frame, text="", font=("Helvetica", 15, "bold"))
        self.license_plate_text.grid(row=1, column=1, padx=10, pady=10)

        # Ràng buộc sự kiện chọn hàng trên treeview
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

    def add_violation(self, car_id, license_plate_text, license_plate_image_path, red_light_image_path,car_image_path, time):
        self.tree.insert("", "end", values=(car_id, license_plate_text, time))


    def on_tree_select(self, event):
        try:
            selected_item = self.tree.selection()[0]
            values = self.tree.item(selected_item, "values")

            car_id = values[0]
            license_plate_text = values[1]
            license_plate_image_path = f"violations/{car_id}_license_plate.png"
            red_light_image_path = f"violations/{car_id}_red_light.png"
            car_image_path= f"violations/{car_id}_car.png"

            # Kiểm tra sự tồn tại của các tệp hình ảnh
            if not os.path.exists(license_plate_image_path):
                print(f"License plate image not found: {license_plate_image_path}")
                return
            if not os.path.exists(red_light_image_path):
                print(f"Red light image not found: {red_light_image_path}")
                return
            if not os.path.exists(car_image_path):
                print(f"car image not found: {car_image_path}")
                return

            # Hiển thị hình ảnh biển số
            license_plate_image = Image.open(license_plate_image_path)
            license_plate_image = license_plate_image.resize((200, 100))
            self.license_plate_photo = ImageTk.PhotoImage(license_plate_image)
            self.license_plate_canvas.create_image(0, 0, anchor="nw", image=self.license_plate_photo)

            # Hiển thị hình ảnh đèn tín hiệu giao thông
            red_light_image = Image.open(red_light_image_path)
            red_light_image = red_light_image.resize((100, 200))
            self.red_light_photo = ImageTk.PhotoImage(red_light_image)
            self.red_light_canvas.create_image(0, 0, anchor="nw", image=self.red_light_photo)

            car_image = Image.open(car_image_path)
            car_image = car_image.resize((200, 200))
            self.car_photo = ImageTk.PhotoImage(car_image)
            self.car_canvas.create_image(0, 0, anchor="nw", image=self.car_photo)

            # Hiển thị biển số
            self.license_plate_text.config(text=license_plate_text)
        except IndexError:
            print("No item selected")
        except Exception as e:
            print(f"Error: {e}")



class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Light Violation Alert")

        # Tạo frame cho video
        self.video_frame = ttk.Frame(root)
        self.video_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.canvas_width = 900
        self.canvas_height = 700
        self.canvas = tk.Canvas(self.video_frame, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(expand=True, fill="both")

        self.btn_open = tk.Button(self.video_frame, text="Open Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT)

        self.btn_draw_left = tk.Button(self.video_frame, text="Draw Boundary", command=self.draw_left_boundary)
        self.btn_draw_left.pack(side=tk.LEFT)

        self.btn_draw_right = tk.Button(self.video_frame, text="Draw Right Boundary", command=self.draw_right_boundary)
        self.btn_draw_right.pack(side=tk.LEFT)
        self.btn_draw_right.config(state=tk.DISABLED)  # Ban đầu nút này bị vô hiệu hóa

        self.btn_detect_red_light = tk.Button(self.video_frame, text="Detect Red Light Violation", command=self.toggle_red_light_detection)
        self.btn_detect_red_light.pack(side=tk.LEFT)

        # Labels to display boundary coordinates
        self.left_coords_label = tk.Label(self.video_frame, text="Line coordinates: None")
        self.left_coords_label.pack(side=tk.LEFT)
        
        # Right line button
        self.right_coords_btn = tk.Button(self.video_frame, text="Allow Right Turn?", command=self.enable_right_draw)
        self.right_coords_btn.pack(side=tk.LEFT)
        self.right_draw_enabled = False

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

        self.red_light_detected = False
        self.red_light_counter = 0
        self.red_light_threshold = 5  # Số khung hình liên tiếp để xác nhận đèn đã chuyển xanh
        self.red_light_bbox = (0, 0, 0, 0)

        # Load models
        self.license_plate_detector = YOLO('lastict.pt')
        self.tracker = DeepSort(max_age=15)
        self.vehicles_info = {}

        # Detection flag
        self.detect_red_light = False
        self.list_violation = set()
        # Results storage
        self.results = {}
        self.frame_nmr = -1
        self.app = TrafficViolationApp(root)

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

            self.root.after(100, self.show_frame)

    def save_violation_info(self, car_id, license_plate_crop, red_light_bbox, license_plate_text, car_bbox):
        # Tạo thư mục violations nếu chưa tồn tại
        if not os.path.exists("violations"):
            os.makedirs("violations")

        # Lưu hình ảnh biển số
        license_plate_image_path = f"violations/{car_id}_license_plate.png"
        cv2.imwrite(license_plate_image_path, license_plate_crop)

        # Lưu hình ảnh đèn tín hiệu giao thông
        red_light_image = self.frame[int(red_light_bbox[1]):int(red_light_bbox[3]), int(red_light_bbox[0]):int(red_light_bbox[2])]
        red_light_image_path = f"violations/{car_id}_red_light.png"
        cv2.imwrite(red_light_image_path, red_light_image)

        car_image = self.frame[int(car_bbox[1]-100 if car_bbox[1]>100 else 0):int(car_bbox[3]+100 if car_bbox[3]+100 < self.frame.shape[0] else self.frame.shape[0]), 
                               int(car_bbox[0]-100 if car_bbox[0]>100 else 0):int(car_bbox[2]+100 if car_bbox[2]+100 < self.frame.shape[1] else self.frame.shape[1])]
        car_image_path = f"violations/{car_id}_car.png"
        cv2.imwrite(car_image_path, car_image)

        # Lưu thông tin vào file CSV
        with open("violations/violations.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            violation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([car_id, license_plate_text, license_plate_image_path, red_light_image_path, car_image_path, violation_time])

        # Thêm thông tin vào giao diện Tkinter
        self.app.add_violation(car_id, license_plate_text, license_plate_image_path, red_light_image_path, car_image_path, violation_time)

    def remove_violation(self, car_id):
        
        if car_id in self.list_violation:
            self.list_violation.remove(car_id)
        
        # Xóa thông tin khỏi Treeview
        for item in self.app.tree.get_children():
            if self.app.tree.item(item, "values")[0] == str(car_id):
                self.app.tree.delete(item)
                break
        
        # Xóa thông tin khỏi file CSV
        temp_file = "violations/temp_violations.csv"
        with open("violations/violations.csv", mode="r", newline="") as infile, open(temp_file, mode="w", newline="") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            for row in reader:
                if row[0] != str(car_id):
                    writer.writerow(row)
        os.replace(temp_file, "violations/violations.csv")
    
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

    def enable_right_draw(self):
        if self.right_draw_enabled:
            self.btn_draw_right.config(state=tk.DISABLED)
            self.right_coords_btn.config(text="Allow Right Turn?")
        else:
            self.btn_draw_right.config(state=tk.NORMAL)
            self.right_coords_btn.config(text="Disable Right Turn")
        self.right_draw_enabled = not self.right_draw_enabled

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
            self.left_line = self.canvas.create_line(self.point1[0], self.point1[1], self.point2[0], self.point2[1], fill="red", width=2)
            self.left_coords = (scaled_point1, scaled_point2)
            self.left_coords_label.config(text=f"Line coordinates: {self.left_coords}")
        elif self.current_line == "right":
            if self.right_line:
                self.canvas.delete(self.right_line)
            self.right_line = self.canvas.create_line(self.point1[0], self.point1[1], self.point2[0], self.point2[1], fill="green", width=2)
            self.right_coords = (scaled_point1, scaled_point2)
            self.right_coords_btn.config(text=f"Right line coordinates: {self.right_coords}")
        self.canvas.delete("preview_line")

    def toggle_red_light_detection(self):
        self.detect_red_light = not self.detect_red_light
        if self.detect_red_light:
            self.btn_detect_red_light.config(text="Stop Red Light Detection")

        else:
            self.btn_detect_red_light.config(text="Detect Red Light Violation")

    def detect_and_track(self, frame):
        self.results[self.frame_nmr] = {}
        # Detect vehicles
        red_light_detected_in_frame=False
        detections = self.license_plate_detector(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if int(class_id) == 2:
                detections_.append([[x1, y1, x2-x1, y2-y1], score, int(class_id)])

        # Track vehicles
        track_ids = self.tracker.update_tracks(detections_, frame=frame)
        track_id = []

        for track in track_ids:
            x1, y1, x2, y2 = track.to_tlbr()
            car_id = track.track_id
            track_id.append((x1, y1, x2, y2, car_id))

            if car_id in self.vehicles_info:
                self.vehicles_info[car_id]["detect_license"] = False

        # Detect license plates
        # license_plates = self.license_plate_detector(frame)[0]

        for license_plate in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            if int(class_id) == 0:
                _, _, _, _, car_id = get_car(license_plate, track_id)
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
                                                            'license_crop': license_plate_crop,
                                                            "current_license_bbox": [x1,y1,x2,y2],
                                                            "detect_license": True,
                                                            "violation": False,
                                                            "crossed_stop_line": False,  # Xe đã vượt qua vạch dừng
                                                            "direction": None,  # Hướng di chuyển của xe
                                                        }
                        else:
                            self.vehicles_info[car_id]['current_license_bbox'] = [x1,y1,x2,y2]
                            self.vehicles_info[car_id]['detect_license'] = True
                            _, current_score = self.vehicles_info[car_id]['license_plate']
                            if license_plate_text_score > current_score:
                                self.vehicles_info[car_id]['license_plate'] = (license_plate_text, license_plate_text_score)
                                self.vehicles_info[car_id]['license_crop'] = license_plate_crop
                    else:
                        if car_id not in self.vehicles_info:
                            continue
                        else: 
                            self.vehicles_info[car_id]['detect_license'] = True
                            self.vehicles_info[car_id]['current_license_bbox'] = [x1,y1,x2,y2]
            elif int(class_id) == 1:  # Red light
                red_light_detected_in_frame = True
                self.red_light_bbox = (x1, y1, x2, y2)
        # Cập nhật trạng thái đèn đỏ 
        if red_light_detected_in_frame:
            self.red_light_counter = 0
            self.red_light_detected = True
        else:
            self.red_light_counter += 1
            if self.red_light_counter >= self.red_light_threshold:
                self.red_light_detected = False
        if self.left_coords :
            left_line_x1, left_line_y1 = self.left_coords[0]
            left_line_x2, left_line_y2 = self.left_coords[1]
        if  self.right_coords and self.right_draw_enabled:
            right_line_x1, right_line_y1 = self.right_coords[0]
            right_line_x2, right_line_y2 = self.right_coords[1]
        # Check for red light violation and display results
        for track in track_ids:
            x1, y1, x2, y2 = track.to_tlbr()
            car_id = track.track_id

            if car_id in self.vehicles_info and 'license_plate' in self.vehicles_info[car_id]:
                license_plate_text, _ = self.vehicles_info[car_id]['license_plate']
                license_plate_crop = self.vehicles_info[car_id]['license_crop']
                license_bbox = self.vehicles_info[car_id].get('current_license_bbox', None)
                license_detect = self.vehicles_info[car_id]['detect_license']
                # if license_detect:
                if license_plate_text is not None:
                    color=(0, 255, 0) 
                        # Kiểm tra hướng di chuyển của xe
                    if self.vehicles_info[car_id]['direction'] is None:
                            if y2 < left_line_y1 and y2 < left_line_y2:
                                self.vehicles_info[car_id]['direction'] = 'backward'
                            elif y2 > left_line_y1 and y2 > left_line_y2:
                                self.vehicles_info[car_id]['direction'] = 'forward'

                        # Kiểm tra vi phạm đèn đỏ
                    if self.vehicles_info[car_id]['direction'] == 'forward':
                            if y2 < left_line_y1 and y2 < left_line_y2 and self.vehicles_info[car_id]['crossed_stop_line'] == False:
                                self.vehicles_info[car_id]['crossed_stop_line'] = True
                                if self.red_light_detected:
                                    self.vehicles_info[car_id]['violation'] = True

                    if self.right_coords and self.right_draw_enabled:
                            if x2 > right_line_x1 and x2 > right_line_x2 and self.vehicles_info[car_id]['violation']==True:
                                color=(0, 255, 0)       
                                self.vehicles_info[car_id]['violation']=False
                                if car_id in self.list_violation:
                                    self.remove_violation(car_id)

                    if self.vehicles_info[car_id]['violation']:
                            if car_id not in self.list_violation:
                                self.save_violation_info(car_id, license_plate_crop
                                                         , self.red_light_bbox, license_plate_text, (x1,y1,x2,y2))
                                self.list_violation.add(car_id)
                            color = (0, 0, 255)
                                                   
                    draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4,
                                            line_length_x=50, line_length_y=50)                        
                    if license_detect:
                            # Use license plate bounding box
                            lx1, ly1, lx2, ly2 = license_bbox
                            H, W, _ = license_plate_crop.shape

                            if (
                                0 < int(ly1) - H < frame.shape[0] and
                                0 < int(ly1) < frame.shape[0] and
                                0 < int((lx2 + lx1 - W) / 2) < frame.shape[1] and
                                0 < int((lx2 + lx1 + W) / 2) < frame.shape[1]
                                
                            ):
                                
                                frame[int(ly1) - H:int(ly1), int((lx2 + lx1 - W) / 2):int((lx2 + lx1 + W) / 2), :] = license_plate_crop
                                frame[int(ly1) - int(1.5 * H):int(ly1) - H, int((lx2 + lx1 - W) / 2):int((lx2 + lx1 + W) / 2), :] = (255, 255, 255)

                            (text_width, text_height), _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                            text_x = int((lx2 + lx1 - text_width) / 2)
                            text_y = int(ly1 - H - H / 8 - (text_height / 2))
                    else:
                            # Use vehicle bounding box
                            H, W, _ = license_plate_crop.shape

                            if (
                                0 < int(y1)+int(0.5 * H) < frame.shape[0] and
                                0 < int(y1)+int(1.5 * H) < frame.shape[0] and
                                0 < int((x2 + x1 - W) / 2) < frame.shape[1] and
                                0 < int((x2 + x1 + W) / 2) < frame.shape[1]
                            ):
                                frame[int(y1)+int(0.5 * H) :int(y1)+int(1.5 * H), int((x2 + x1 - W) / 2):int((x2 + x1 + W) / 2), :] = license_plate_crop
                                frame[int(y1) :int(y1)+int(0.5 * H) , int((x2 + x1 - W) / 2):int((x2 + x1 + W) / 2), :] = (255, 255, 255)

                            (text_width, text_height), _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                            text_x = int((x2 + x1 - text_width) / 2)
                            text_y = int(y1  + H / 8 + (text_height / 2))

                        # Put the text on the frame
                    cv2.putText(frame, license_plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)


        # Display the frame with annotations
        display_frame_resized = self.resize_frame(frame, self.canvas_width, self.canvas_height)
        display_frame_rgb = cv2.cvtColor(display_frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

if __name__ == "__main__":
    root = tk.Tk()
    root.grid_columnconfigure(0, weight=5)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=5)
    root.grid_rowconfigure(1, weight=1)

    app = VideoApp(root)
    root.mainloop()
