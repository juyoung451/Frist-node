#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 퍼블리셔 생성
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # 서브스크라이버 생성
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback, qos_profile)

        # 변수 초기화
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()

        self.takeoff_height = -5.0  # NED 기준: z = -5m 위로
        self.takeoff_completed = False
        self.landing_started = False
        self.landed = False

        # 타이머 시작 (10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

    # 콜백: 위치 및 상태 수신
    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    # 오프보드 모드 제어 신호
    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.get_timestamp()
        self.offboard_control_mode_publisher.publish(msg)

    # TrajectorySetpoint 발행
    def publish_position_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # 90도 (rad)
        msg.timestamp = self.get_timestamp()
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f'Setpoint: [{x}, {y}, {z}]')

    # VehicleCommand 발행
    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.param3 = params.get('param3', 0.0)
        msg.param4 = params.get('param4', 0.0)
        msg.param5 = params.get('param5', 0.0)
        msg.param6 = params.get('param6', 0.0)
        msg.param7 = params.get('param7', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_timestamp()
        self.vehicle_command_publisher.publish(msg)

    # 타임스탬프 생성
    def get_timestamp(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    # Arm
    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Sent ARM command")

    # Disarm
    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info("Sent DISARM command")

    # 오프보드 모드 전환
    def engage_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to OFFBOARD mode")

    # 착륙
    def land(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Landing initiated")

    # 주 타이머 루프
    def timer_callback(self):
        self.publish_offboard_control_heartbeat_signal()

        # 초기 10회 오프보드 준비용 setpoint 송신
        if self.offboard_setpoint_counter < 10:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            self.offboard_setpoint_counter += 1
            return

        # 10회 이후: 오프보드 진입 & ARM
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            time.sleep(0.1)
            self.arm()
            self.get_logger().info("OFFBOARD mode engaged and vehicle armed")
            self.offboard_setpoint_counter += 1
            return

        # 이륙 중: 목표 고도까지 계속 setpoint 전송
        if not self.takeoff_completed:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            if self.vehicle_local_position.z <= self.takeoff_height + 0.2:
                self.get_logger().info("Takeoff complete")
                self.takeoff_completed = True
                self.takeoff_hold_counter = 0
            return

        # 고도 유지: 5초 대기
        if self.takeoff_completed and not self.landing_started:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            self.takeoff_hold_counter += 1
            if self.takeoff_hold_counter >= 50:  # 50 x 0.1s = 5초
                self.land()
                self.landing_started = True
            return

        # 착륙 중
        if self.landing_started and not self.landed:
            if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.landed = True
                self.get_logger().info("Landed and disarmed. Exiting...")
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = OffboardControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        
