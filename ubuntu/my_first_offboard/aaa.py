import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
import time

class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')
        
        # Publishers
        self.pub_ctrl = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.pub_setpoint = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # Waypoints (NED frame: z 음수 = 상승)
        self.waypoints = [
            (0.0, 0.0, -2.0),
            (5.0, 0.0, -2.0),
            (5.0, 5.0, -2.0),
            (0.0, 5.0, -2.0)
        ]
        self.idx = 0
        self.counter = 0

        # 20Hz 주기 타이머
        self.timer = self.create_timer(0.05, self.timer_cb)

        # 사전 준비
        self.armed = False
        self.offboard_started = False

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        cmd = VehicleCommand()
        cmd.command = command
        cmd.param1 = float(param1)
        cmd.param2 = float(param2)
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.pub_cmd.publish(cmd)

    def timer_cb(self):
        # 1) 항상 OffboardControlMode 전송
        ctrl = OffboardControlMode()
        ctrl.position = True
        ctrl.velocity = False
        ctrl.acceleration = False
        ctrl.attitude = False
        ctrl.body_rate = False
        self.pub_ctrl.publish(ctrl)

        # 2) 현재 waypoint setpoint 전송
        x, y, z = self.waypoints[self.idx]
        sp = TrajectorySetpoint()
        sp.position = [float(x), float(y), float(z)]
        self.pub_setpoint.publish(sp)

        # 3) 초기 40번(약 2초) 동안 setpoint 보내면서 Arm & Offboard 준비
        if self.counter == 5 and not self.armed:
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.get_logger().info('Arming...')
            self.armed = True

        if self.counter == 40 and not self.offboard_started:
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.get_logger().info('Switching to Offboard mode...')
            self.offboard_started = True

        # 4) waypoint 변경
        if self.offboard_started and self.counter % 200 == 0 and self.counter > 0:
            self.idx = (self.idx + 1) % len(self.waypoints)
            self.get_logger().info(f"Moving to waypoint {self.idx}: {self.waypoints[self.idx]}")

        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
