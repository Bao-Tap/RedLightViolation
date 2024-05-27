from ultralytics import YOLO
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from deep_sort_realtime.deepsort_tracker import DeepSort
from util import *


results = {}

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('last.pt')


vehicles_info = {}
cap = cv2.VideoCapture('_storage_emulated_0_Download_VID_20240520_105404.mp4')

vehicles = [2, 3, 5, 7]
tracker = DeepSort(max_age=30)
# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if int(class_id) in vehicles:
                detections_.append([[x1, y1, x2-x1, y2-y1], score,int(class_id)] )

        # track vehicles
        track_ids = tracker.update_tracks(detections_,frame = frame )
        # Khởi tạo danh sách track_id rỗng
        track_id = []

        # Duyệt qua mỗi đối tượng track trong track_ids
        for track in track_ids:
            # Lấy thông tin từ track
            x1, y1, x2, y2 = track.to_tlbr()
            car_id = track.track_id
            
            # Cập nhật danh sách track_id
            track_id.append((x1, y1, x2, y2, car_id))
            
            # Cập nhật vehicles_info với thông tin mới
            if car_id in vehicles_info:
                
                vehicles_info[car_id][frame_nmr] = {
                    'bbox': [x1, y1, x2, y2],
                }

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            if (int(class_id)== 1): 
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            if int(class_id) == 0:
                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_id)

                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)


                    height, width = license_plate_crop_gray.shape[:2]
                    new_dimensions = (width * 6, height * 6)
                    # Thay đổi kích thước hình ảnh
                    resized_image = cv2.resize(license_plate_crop_gray, new_dimensions, interpolation=cv2.INTER_LINEAR)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(resized_image)

                    if license_plate_text is not None:
                        if car_id not in vehicles_info:
                            vehicles_info[car_id] = {}
                            vehicles_info[car_id][frame_nmr] = {
                                'bbox': [xcar1, ycar1, xcar2, ycar2],
                            }
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                       'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                        

# write results
write_csv(results, './test.csv',vehicles_info)
