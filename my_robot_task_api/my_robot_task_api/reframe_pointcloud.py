#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class ReframePC(Node):
    def __init__(self):
        super().__init__('reframe_pointcloud')

        in_topic = '/rgbd_camera/points'
        out_topic = '/rgbd_camera/points_reframed'
        self.new_frame = 'camera_link'

        # Subscribe to sensor stream (usually BEST_EFFORT)
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publish in RELIABLE so RViz/CLI (often RELIABLE) can receive
        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.pub = self.create_publisher(PointCloud2, out_topic, pub_qos)
        self.sub = self.create_subscription(PointCloud2, in_topic, self.cb, sub_qos)

        self.get_logger().info(
            f"Reframing {in_topic} -> {out_topic}, frame_id='{self.new_frame}' "
            f"(sub BEST_EFFORT, pub RELIABLE)"
        )

    def cb(self, msg: PointCloud2):
        msg.header.frame_id = self.new_frame
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = ReframePC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
