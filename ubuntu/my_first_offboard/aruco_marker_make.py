import cv2
import numpy as np

# 아루코 딕셔너리 가져오기 (4x4, ID 수: 50)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# 마커 ID 설정 (0~49)
marker_id = 0

# 마커 이미지 크기 (픽셀 단위)
marker_size = 200

# 마커 이미지 생성
marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# 이미지 저장
filename = f"aruco_marker_{marker_id}.png"
cv2.imwrite(filename, marker_img)

print(f"{filename} 저장 완료.")

