import csv
import numpy as np
from scipy.interpolate import interp1d

def interpolate_bounding_boxes(data):
    interpolated_data = []

    # Chuyển đổi dữ liệu thành từ điển
    cars_dict = {}
    for row in data:
        car_id = row['car_id']
        frame_nmr = row['frame_nmr']
        
        # Thông tin cần lưu trữ cho mỗi frame
        frame_info = {
            'car_bbox': row['car_bbox'],
            'license_plate_bbox': row['license_plate_bbox'],
            'license_plate_bbox_score': row['license_plate_bbox_score'],
            'license_number': row['license_number'],
            'license_number_score': row['license_number_score']
        }  
        
        if car_id not in cars_dict:
            cars_dict[car_id] = {}
        cars_dict[car_id][frame_nmr] = frame_info

    # Tạo từ điển để theo dõi các frame đã được thêm vào interpolated_data
    added_frames = set()
    for car_id, frames_dict in cars_dict.items():
        frame_numbers = [int(frame_nmr) for frame_nmr in frames_dict.keys()]
        license_plate_bboxes = {int(frame_nmr): list(map(float, frame_info['license_plate_bbox'][1:-1].split())) 
                                for frame_nmr, frame_info in frames_dict.items()}

        # Lọc ra các frame có và không có biển số xe được phát hiện
        detected_frames = [frame for frame, bbox in license_plate_bboxes.items() if bbox != [0.0, 0.0, 0.0, 0.0]]
        # print(f"{car_id}{detected_frames}")
        missing_frames = [frame for frame in frame_numbers if frame not in detected_frames]
        # print(f"{car_id}{missing_frames}")
        # Duyệt qua các frame có biển số xe và nội suy nếu cần
        for i, frame in enumerate(detected_frames):
            if i == 0 or (i > 0 and detected_frames[i-1] == frame - 1):  # Kiểm tra xem có phải frame liền kề không
                continue  # Không cần nội suy cho frame đầu tiên hoặc các frame liền kề


            start_frame = detected_frames[i-1]
            end_frame = frame
            # Nội suy cho các frame bị mất giữa start_frame và end_frame
            if end_frame - start_frame > 1:
                start_bbox = np.array(license_plate_bboxes[start_frame])
                end_bbox = np.array(license_plate_bboxes[end_frame])
                x = np.array([start_frame, end_frame])
                print(x)
                for missing_frame in range(start_frame + 1, end_frame):
                    t = np.array([missing_frame])
                    interpolated_bbox = interp1d(x, np.vstack([start_bbox, end_bbox]), axis=0, kind='linear')(t)
                    interpolated_bbox_str = ' '.join(map(str, interpolated_bbox.flatten().tolist()))
                    interpolated_data.append({
                        'frame_nmr': str(missing_frame),
                        'car_id': str(car_id),
                        'car_bbox': frames_dict[str(missing_frame)]['car_bbox'],  # Sử dụng car_bbox gốc nếu có
                        'license_plate_bbox': '[' + interpolated_bbox_str + ']',
                        'license_plate_bbox_score': '0',  # Đặt giá trị mặc định vì nội suy
                        'license_number': '0',  # Đặt giá trị mặc định vì nội suy
                        'license_number_score': '0'  # Đặt giá trị mặc định vì nội suy
                    })
                    added_frames.add((int(car_id), missing_frame))


    # Thêm tất cả các frame vào danh sách interpolated_data và tránh trùng lặp
    for car_id, frames_dict in cars_dict.items():
        for frame_nmr, frame_info in frames_dict.items():
            key = (int(car_id), int(frame_nmr))
            if key not in added_frames:
                interpolated_data.append({
                    'frame_nmr': frame_nmr,
                    'car_id': car_id,
                    'car_bbox': frame_info['car_bbox'],
                    'license_plate_bbox': frame_info['license_plate_bbox'],
                    'license_plate_bbox_score': frame_info['license_plate_bbox_score'],
                    'license_number': frame_info['license_number'],
                    'license_number_score': frame_info['license_number_score']
                })
                added_frames.add(key)

    # Sắp xếp lại danh sách theo car_id và frame_nmr tăng dần
    sorted_list = sorted(interpolated_data, key=lambda x: (int(x['car_id']), int(x['frame_nmr'])))

    return sorted_list




# Load the CSV file
with open('test_modified.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)


