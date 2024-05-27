import csv

# Đọc file dữ liệu vào một danh sách các dòng
with open('test.csv', 'r') as file:
    reader = csv.reader(file)
    # Bỏ qua dòng đầu tiên (header)
    next(reader)
    data = list(reader)

car_data = {}

# Duyệt qua từng dòng trong dữ liệu
for row in data:
    frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_number, license_number_score = row
    license_plate_bbox = license_plate_bbox if license_plate_bbox != '' else '[0 0 0 0]'
    license_plate_bbox_score = license_plate_bbox_score if license_plate_bbox_score != '' else 0
    license_number = license_number if license_number != '' else 0
    license_number_score = license_number_score if license_number_score != '' else 0

    # Nếu chưa có thông tin về xe này trong dictionary, thêm mới
    if (frame_nmr, car_id) not in car_data:
        car_data[(frame_nmr, car_id)] = {
            'frame_nmr': frame_nmr,
            'car_id': car_id,
            'car_bbox': car_bbox,
            'license_plate_bbox': license_plate_bbox,
            'license_plate_bbox_score': license_plate_bbox_score,
            'license_number': license_number,
            'license_number_score': license_number_score
        }


    
# print(car_data.values())

# Sắp xếp lại dữ liệu theo frame_nmr và car_id
sorted_data = sorted(car_data.values(), key=lambda x: (int(x['car_id']), int(x['frame_nmr'])))
print("Kiểm tra số lượng dòng dữ liệu:", type(sorted_data))
if len(sorted_data) > 0:
    print("Dữ liệu dòng đầu tiên:", sorted_data[0])
else:
    print("Không có dữ liệu để ghi!")

# Ghi dữ liệu đã sửa đổi vào file mới

header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_modified.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(sorted_data)