#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


class DebugGripperState(Node):
    def __init__(self):
        super().__init__("debug_gripper_state")

        self.create_subscription(
            String,
            "/gripper_impact_state",
            self.state_cb,
            10
        )

        self.create_subscription(
            Bool,
            "/gripper_impact_detected",
            self.bool_cb,
            10
        )

        self.get_logger().info("Debug subscriber started")

    def state_cb(self, msg):
        self.get_logger().info(f"STATE: {msg.data}")

    def bool_cb(self, msg):
        self.get_logger().info(f"IMPACT: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = DebugGripperState()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()