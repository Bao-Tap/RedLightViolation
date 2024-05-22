import cv2

from ultralytics import YOLO
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


# Config value
video_path = "Video.mp4439.mp4"
conf_threshold = 0.3
tracking_class = [2,3,5,7] # None: track all

# Khởi tạo DeepSort
tracker = DeepSort(max_age=20)

model = YOLO('yolov8n.pt')

class_names = ["","","CAR","moto","","bus","","truck"]

colors = np.random.randint(0,255, size=(10,3 ))
print(colors)
tracks = []

# Khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(video_path)

# Tiến hành đọc từng frame từ video
while True:
    # Đọc
    ret, frame = cap.read()
    if not ret:
        continue
    detections = model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        x1,y1, x2, y2 = map(int, (x1, y1, x2, y2))
        if int(class_id) in tracking_class:
            if score < conf_threshold:
                continue
        if score < conf_threshold:
                continue

        if int(class_id) in tracking_class:
            detections_.append([ [x1, y1, x2-x1, y2 - y1], score, class_id ])


    # Cập nhật,gán ID băằng DeepSort
    tracks = tracker.update_tracks(detections_, frame = frame)

    # Vẽ lên màn hình các khung chữ nhật kèm ID
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Lấy toạ độ, class_id để vẽ lên hình ảnh
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[int(class_id)]
            B, G, R = map(int,color)

            label = "{}-{}".format(class_names[int(class_id)], track_id)

            # Độ dày của đường viền
            line_thickness = 20  # Tăng từ 10 lên 20 cho rõ ràng hơn trên ảnh 4K

            # Kích thước phông chữ và độ dày của văn bản
            font_scale = 1.0  # Tăng kích thước phông chữ
            text_thickness = 3  # Tăng độ dày của văn bản

            # Vẽ hộp chứa
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), line_thickness)
            cv2.rectangle(frame, (x1 - 1, y1 - 40), (x1 + len(label) * 24, y1), (B, G, R), -1)

            # Vẽ văn bản
            cv2.putText(frame, label, (x1 + 10, y1 - 16), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)


    # Show hình ảnh lên màn hình
    frame = cv2.resize(frame, (1200,700))
    cv2.imshow("OT", frame)
    # Bấm Q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()