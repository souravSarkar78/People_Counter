import cv2
import numpy as np
from tracker import *

# Initialize Tracker
tracker = Tracker()


# Define the background substraction algo
algo = cv2.createBackgroundSubtractorMOG2()

obj_list = []
count = 0

up_pos = 430
down_pos = 440

def reset():
    global count, obj_list
    count = 0
    obj_list = []


def find_center(x, y, w, h):
    """Takes rectangle coordinate and returns center point og that rectangle"""
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy


def count_object(box):
    """Takes detection boxes and counts object within area"""
    global count
    x, y, w, h, id = box

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    if (iy < down_pos) and (iy > up_pos):

        if id not in obj_list:
            obj_list.append(id)
            count += 1


def Detector(frame):

    frame_height, frame_width, c = frame.shape

    # Show the vehicle counting on the frame
    poly = np.array([[[0, 350], [frame_width, 350], [frame_width, 500], [0, 500]]], np.int32)
    cv2.polylines(frame, [poly], True, (0,255,0), thickness=3)


    # Perform background substraction technique to detect vahicles

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    img_sub = algo.apply(blur)


    # Filter substracted output
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # kernel to apply to the morphology

    closing = cv2.morphologyEx(img_sub, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel)
    ret,binary = cv2.threshold(dilation ,200,255,cv2.THRESH_BINARY)

    # Find the contour of the final mask
    contour, h = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in (contour):

        # Finf rectangle boundaries of the detected contours

        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 500:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            cx, cy = find_center(x, y, w, h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            # Checkes if a 2D point inside of a polygon
            intersection_distance = cv2.pointPolygonTest(poly, (cx, cy), True)

            # print(intersection_distance)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            if intersection_distance >= 0:

                detections.append([x, y, w, h])

    
    return detections


def objectTracker(frame):

    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
    detections = Detector(frame)
    # 2. Object Tracking
    boxes_ids = tracker.update(detections)

    for index, box_id in enumerate(boxes_ids):
        x, y, w, h, id = box_id
        
        # Count
        count_object(box_id)

        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    
    cv2.line(frame, (0, up_pos+5), (1152, up_pos+5), (255, 0, 255), 2)

    cv2.putText(frame, "Count- "+str(count), (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    
    return frame