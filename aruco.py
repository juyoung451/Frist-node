import cv2 as cv
from cv2 import aruco
import numpy as np

# dictionary to specify type of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
detector = aruco.ArucoDetector(marker_dict)

# detect the markerq
param_markers = aruco.DetectorParameters()

# utilizes default camera/webcam driver
cap = cv.VideoCapture(0)

# iterate through multiple frames, in a live video feed
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # turning the frame to grayscale-only (for efficiency)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    marker_corners, marker_IDs, reject = detector.detectMarkers(gray_frame)
 
    #print("corner : ", marker_corners)
    
    # getting conrners of markers
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            print("bottom_right : ", top_left)
            
            center = (int((top_right[0] + bottom_right[0]) / 2), int((top_right[1] + bottom_right[1]) / 2))
            cv.circle(frame, center, 5, (0, 0, 255), -1)
            ##cv.rectangle(frame, top_right, bottom_right, 5)

            cv.putText(
                frame,
                f"id: {ids[0]}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (200, 100, 0),
                2,
                cv.LINE_AA,
            )
            #print(ids, "  ", corners)
    cv.imshow("frame", frame)
    print("frame : ", frame)
    #cv.imshow("gray_frame", gray_frame)
    key = cv.waitKey(1)
    #w = frame.get(cv.CAP_PROP_FRAME_WIDTH)
    #h = frame.get(cv.CAP_PROP_FRAME_HEIGHT)
    #print("원본 동영상 너비(가로) : {}, 높이(세로) : {}".format(w, h))
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()