from ultralytics import YOLO
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from deep_sort_realtime.deepsort_tracker import DeepSort
from util import *


results = {}

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('last.pt')


vehicles_info = {}
cap = cv2.VideoCapture('Video.mp4440.mp4')

vehicles = [2, 3, 5, 7]
tracker = DeepSort(max_age=30)
# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # results[frame_nmr] = {}
        # # detect vehicles
        # detections = coco_model(frame)[0]
        # detections_ = []
        # for detection in detections.boxes.data.tolist():
        #     x1, y1, x2, y2, score, class_id = detection
        #     x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        #     if int(class_id) in vehicles:
        #         detections_.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

        # # track vehicles
        # track_ids = tracker.update_tracks(detections_, frame=frame)
        # # Khởi tạo danh sách track_id rỗng
        # track_id = []

        # # Duyệt qua mỗi đối tượng track trong track_ids
        # for track in track_ids:
        #     # Lấy thông tin từ track
        #     x1, y1, x2, y2 = track.to_tlbr()
        #     car_id = track.track_id

        #     # Cập nhật danh sách track_id
        #     track_id.append((x1, y1, x2, y2, car_id))

        #     # Cập nhật vehicles_info với thông tin mới
        #     if car_id in vehicles_info:
        #         vehicles_info[car_id][frame_nmr] = {
        #             'bbox': [x1, y1, x2, y2],
        #         }

        # detect license plates
        license_plates = coco_model(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            if int(class_id) != 3:
                # Vẽ hình chữ nhật xung quanh đèn đỏ
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, 'Red Light', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, 'plate', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Hiển thị khung hình
        frame=cv2.resize(frame, (1200,700))
        cv2.imshow('Frame', frame)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

# write results
# write_csv(results, './test.csv', vehicles_info)
