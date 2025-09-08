import cv2
import numpy as np
from cv2 import aruco

# 1. 카메라 내부 파라미터
camera_matrix = np.array([[640, 0, 320],
                          [0, 640, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # 왜곡 없음 가정

# 2. 마커 딕셔너리 및 감지기 초기화
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
aruco_detector = aruco.ArucoDetector(aruco_dict)

# 3. 큐브 정의
cube_vertices = np.array([
    [0.03, -0.03, 0.06],
    [0.03, 0.03, 0.06],
    [-0.03, 0.03, 0.06],
    [-0.03, -0.03, 0.06],
    [0.03, -0.03, 0],
    [0.03, 0.03, 0],
    [-0.03, 0.03, 0],
    [-0.03, -0.03, 0]])

cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]

# 마커 크기
marker_length = 0.05  # 5cm

# 마커 3D 좌표 (왼쪽 아래부터 시계방향)
object_points = np.array([
    [-marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

# 4. 메인 루프
def cube_on_aruco():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        corners, ids, _ = aruco_detector.detectMarkers(frame)

        if ids is not None:
            for i in range(len(ids)):
                image_points = corners[i][0].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

                if success:
                    cube_points, _ = cv2.projectPoints(cube_vertices, rvec, tvec, camera_matrix, dist_coeffs)

                    for edge in cube_edges:
                        start = tuple(map(int, cube_points[edge[0]].ravel()))
                        end = tuple(map(int, cube_points[edge[1]].ravel()))
                        cv2.line(frame, start, end, (255, 255, 0), 2)

                    # 축도 그려보기 (선택)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

        cv2.imshow("AR Cube", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Esc
            break

    cap.release()
    cv2.destroyAllWindows()


print("Drawing 3D cube on ArUco marker...")
cube_on_aruco()

