import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2
from cv2 import aruco
import numpy as np

class ArucoCameraSubscriber(Node):
    def __init__(self):
        super().__init__('aruco_camera_subscriber')

        self.subscription = self.create_subscription(
            Image,
            '/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image',
            self.listener_callback,
            10)
        
        self.bridge = CvBridge()

        # ArUco 설정
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.marker_dict, self.parameters)

    def listener_callback(self, msg):
        # ROS 이미지 → OpenCV 이미지
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 마커 탐지
        marker_corners, marker_IDs, rejected = self.detector.detectMarkers(gray_frame)

        if marker_corners:
            for ids, corners in zip(marker_IDs, marker_corners):
                cv2.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
                )
                corners = corners.reshape(4, 2).astype(int)
                top_right = corners[0]
                bottom_right = corners[2]
                center = ((top_right[0] + bottom_right[0]) // 2, (top_right[1] + bottom_right[1]) // 2)

                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"id: {ids[0]}",
                    top_right,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.3,
                    (200, 100, 0),
                    2,
                    cv2.LINE_AA,
                )

        # 화면 표시
        cv2.imshow("Aruco Detection", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            rclpy.shutdown()  # 'q'를 누르면 종료

def main(args=None):
    rclpy.init(args=args)
    node = ArucoCameraSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

