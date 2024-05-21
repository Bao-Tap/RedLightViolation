import ast

import cv2
import numpy as np
import pandas as pd

TyleanhCat=(3,3) #Gấp 3 lần
#Thông số box xe
thickness=10 # Độ dày đường viền kẻ xe
line_length_x=50 
line_length_y=50

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


results = pd.read_csv('./test_interpolated.csv')

# load video
cap = cv2.VideoCapture('VID_20240520_105151.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out2.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}
    frame_nmr = results[(results['car_id'] == car_id) & 
                        (results['license_number_score'] == max_)]['frame_nmr'].iloc[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
    ret, frame = cap.read()

    if not ret:
        print(f"Cannot read frame {frame_nmr} for car_id {car_id}")
        continue

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) & 
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    # Ensure coordinates are within the frame size
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        print(f"Invalid bounding box for car_id {car_id}: ({x1}, {y1}, {x2}, {y2})")
        continue

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

    if license_crop.size == 0:
        print(f"Empty crop for car_id {car_id}")
        continue

    try:
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * TyleanhCat[0]), int((y2 - y1) * TyleanhCat[1])))
    except cv2.error as e:
        print(f"Resize error for car_id {car_id}: {e}")
        continue

    license_plate[car_id]['license_crop'] = license_crop



frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]

        for row_indx in range(len(df_)):
            # draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), thickness,
                        line_length_x=line_length_x, line_length_y=line_length_y)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            if int(x1)!=0: 
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                H, W, _ = license_crop.shape

                try:
                    y1_clip = max(int(y1) - H, 0)
                    x1_clip = max(int((car_x2 + car_x1 - W) / 2), 0)
                    x2_clip = min(int((car_x2 + car_x1 + W) / 2), frame.shape[1])

                    if y1_clip + H <= frame.shape[0] and x2_clip - x1_clip == W:
                        frame[y1_clip:int(y1),
                            x1_clip:x2_clip, :] = license_crop
                    # frame[int(y1) - H : int(y1),
                    #     int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                    frame[int(y1) - H- int(H/2):int(y1) - H,
                        int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        4)

                    cv2.putText(frame,
                                license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                (int((car_x2 + car_x1 - text_width) / 2), int(y1 - H-(H/8)  - (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 0, 0),
                                4)

                except:
                    pass
            else:
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
                H, W, _ = license_crop.shape
                try:
                    frame[int(car_y1)  : int(car_y1)+ H,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                    frame[int(car_y1)-int(H/2) :int(car_y1) ,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        4)

                    cv2.putText(frame,
                                license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H/8  - (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 0, 0),
                                4)
                except:
                    pass                          

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        #cv2.imshow('frame', frame)
        #cv2.waitKey(0)

out.release()
cap.release()
