#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class SanitizeJointStates(Node):
    def __init__(self):
        super().__init__("sanitize_joint_states")
        self.sub = self.create_subscription(JointState, "/joint_states", self.cb, 10)
        self.pub = self.create_publisher(JointState, "/joint_states_clean", 10)

    def cb(self, msg: JointState):
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = list(msg.position)

        # Drop velocity/effort entirely (MoveIt doesn't need them)
        out.velocity = []
        out.effort = []

        # Also guard against NaNs in position
        for i, p in enumerate(out.position):
            if p is None or (isinstance(p, float) and math.isnan(p)):
                self.get_logger().warn(f"NaN position for joint {out.name[i] if i < len(out.name) else i}, setting 0.0")
                out.position[i] = 0.0

        self.pub.publish(out)

def main():
    rclpy.init()
    node = SanitizeJointStates()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
