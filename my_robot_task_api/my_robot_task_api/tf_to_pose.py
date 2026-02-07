#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class TfToPose(Node):
    def __init__(self):
        super().__init__("tf_to_pose")

        # ---- Params (change defaults if your frame names differ) ----
        self.declare_parameter("base_frame", "gen3_lite/base_link")
        self.declare_parameter("target_frame", "tag36h11:0")  # apriltag frame
        self.declare_parameter("output_topic", "/object_pose")
        self.declare_parameter("rate_hz", 10.0)

        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value

        self.pub = self.create_publisher(PoseStamped, self.output_topic, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        period = 1.0 / max(self.rate_hz, 1.0)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"Publishing {self.output_topic} from TF {self.base_frame} -> {self.target_frame} @ {self.rate_hz:.1f} Hz"
        )

    def tick(self):
        try:
            # Latest available transform
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.target_frame, rclpy.time.Time())

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame

            msg.pose.position.x = tf.transform.translation.x
            msg.pose.position.y = tf.transform.translation.y
            msg.pose.position.z = tf.transform.translation.z

            msg.pose.orientation = tf.transform.rotation

            self.pub.publish(msg)

        except (LookupException, ConnectivityException, ExtrapolationException):
            # TF not ready yet (tag not detected / buffer not filled)
            pass

def main():
    rclpy.init()
    node = TfToPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
