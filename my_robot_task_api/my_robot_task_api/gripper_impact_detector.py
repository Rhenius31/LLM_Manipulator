#!/usr/bin/env python3

import math
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class GripperImpactDetector(Node):
    def __init__(self):
        super().__init__("gripper_impact_detector")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.impact_pub = self.create_publisher(Bool, "/gripper_impact_detected", qos)
        self.state_pub = self.create_publisher(String, "/gripper_impact_state", qos)

        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("gripper_joint_name", "right_finger_bottom_joint")

        # minimum finger speed 
        self.declare_parameter("moving_threshold", 0.002)

        # sudden drop in joint velocity while closing
        self.declare_parameter("impact_acc_threshold", 0.5)

        # cooldown to avoid repeated triggers
        self.declare_parameter("cooldown", 0.2)

        # detect only when finger is closing
        self.declare_parameter("closing_direction", 1.0)

        self.joint_topic = self.get_parameter("joint_state_topic").value
        self.gripper_joint_name = self.get_parameter("gripper_joint_name").value
        self.moving_threshold = float(self.get_parameter("moving_threshold").value)
        self.impact_acc_threshold = float(self.get_parameter("impact_acc_threshold").value)
        self.cooldown = float(self.get_parameter("cooldown").value)
        self.closing_direction = float(self.get_parameter("closing_direction").value)

        self.prev_vel = None
        self.prev_time = None
        self.last_trigger_time = 0.0

        self.sub = self.create_subscription(
            JointState,
            self.joint_topic,
            self.joint_callback,
            10
        )


        self.get_logger().info(
            f"Gripper impact detector listening on {self.joint_topic}, "
            f"watching joint '{self.gripper_joint_name}'"
        )

    def joint_callback(self, msg: JointState):
        if not msg.name or not msg.velocity:
            return

        if self.gripper_joint_name not in msg.name:
            return

        idx = msg.name.index(self.gripper_joint_name)
        if idx >= len(msg.velocity):
            return

        vel = msg.velocity[idx]
        now = self.get_clock().now().nanoseconds * 1e-9

        acc = 0.0
        impact = False

        if self.prev_vel is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 1e-4:
                acc = (vel - self.prev_vel) / dt

                closing_now = (self.prev_vel * self.closing_direction) > self.moving_threshold

                # sudden slowdown while closing
                sudden_stop = (acc * self.closing_direction) < -self.impact_acc_threshold

                if closing_now and sudden_stop:
                    if (now - self.last_trigger_time) > self.cooldown:
                        impact = True
                        self.last_trigger_time = now

        b = Bool()
        b.data = impact
        self.impact_pub.publish(b)

        state = {
            "joint": self.gripper_joint_name,
            "velocity": round(vel, 6),
            "acceleration": round(acc, 6),
            "impact_detected": impact
        }

        s = String()
        s.data = json.dumps(state)
        self.state_pub.publish(s)

        if impact:
            self.get_logger().warn(
                f"Gripper contact/impact suspected | joint={self.gripper_joint_name} "
                f"| vel={vel:.6f} | acc={acc:.6f}"
            )

        self.prev_vel = vel
        self.prev_time = now


def main(args=None):
    rclpy.init(args=args)
    node = GripperImpactDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()