import re
import string
from paddleocr import PaddleOCR
import cv2

# Initialize the OCR reader
ocr = PaddleOCR(use_angle_cls=True,lang="en",use_gpu=False,show_log=False) 

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'Z': '2',
                    'B': '8'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '2': 'Z',
                    '8': 'B'}


def write_csv(results, output_path,vehicles_info):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        # Ghi thông tin từ vehicles_info
        for car_id, frames_info in vehicles_info.items():
            for frame_nmr, info in frames_info.items():
                f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(*info['bbox']),
                                                            '',
                                                            '',
                                                            '',
                                                            ''))

        f.close()

    # import csv
    # # Mở file CSV với chế độ append
    # with open(output_path, 'a', newline='') as csvfile:
    #     # Tạo một DictWriter với tên các cột như trong file CSV của bạn
    #     fieldnames = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
    #     # Duyệt qua vehicles_info để ghi thông tin vào CSV
    #     for car_id, info in vehicles_info.items():
    #         # Chuẩn bị dòng dữ liệu để ghi
    #         row = {
    #             'frame_nmr': info['last_seen_frame'],
    #             'car_id': car_id,
    #             'car_bbox': str(info['bbox']),
    #             'license_plate_bbox': '',  # Trường này để trống
    #             'license_plate_bbox_score': '',  # Trường này để trống
    #             'license_number': '',  # Trường này để trống
    #             'license_number_score': ''  # Trường này để trống
    #         }
    #         # Ghi dòng dữ liệu vào CSV
    #         writer.writerow(row)



def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) <7:
        return False

    # if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
    #    (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
    #    (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
    #    (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
    #    (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
    #    (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()):
    #     return True
    # else:
    #     return False
    return True


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """

    license_plate_ = ''


    if (len(text) == 10 and text[0] == "1") or (len(text) == 10 and text[0] == "I"):
        text = text[1:]     
    mapping = {0: dict_char_to_int, 1: dict_char_to_int, 2:dict_int_to_char, 4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int}
    for j in range(len(text)):
        if j not in mapping.keys():
            license_plate_ += text[j]
        elif text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = ocr.ocr(license_plate_crop)
    merged_text = ''
    total_score = 0
    count=0   
    if detections == [None]: 
        return None, None 
    for detection in detections:
        for word in detection:
            count+=1
            merged_text += word[1][0]
            total_score += word[1][1]           
    text = merged_text.upper()
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    average_score = total_score / count if detections else 0
    if license_complies_format(cleaned_text):
        return format_license(cleaned_text), average_score

    return None, None
def compute_iou(box1, box2):

    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    intersection = max(0, x2 - x1 ) * max(0, y2 - y1)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0] ) * (box1[3] - box1[1] )
    box2_area = (box2[2] - box2[0] ) * (box2[3] - box2[1] )

    # Compute the Intersection over Union
    iou = intersection / float(box1_area + box2_area - intersection)
    return iou


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    max_iou = 0
    selected_vehicle = None
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # Kiểm tra xem biển số xe có nằm trong bounding box của xe không
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            # Tính toán IoU giữa vùng chứa biển số xe và bounding box của xe
            iou = compute_iou([x1, y1, x2, y2], [xcar1, ycar1, xcar2, ycar2])
            
            if iou > max_iou:
                max_iou = iou
                selected_vehicle = vehicle_track_ids[j]

    if selected_vehicle is not None:
        return selected_vehicle

    return -1, -1, -1, -1, -1
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=1, line_length_x=20, line_length_y=20):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

