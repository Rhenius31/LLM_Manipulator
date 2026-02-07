#!/usr/bin/env python3
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import tf2_ros
import tf2_geometry_msgs

from cv_bridge import CvBridge

class BBoxToPose(Node):
    def __init__(self):
        super().__init__('bbox_to_pose')

        self.bridge = CvBridge()

        self.declare_parameter('depth_topic', '/rgbd_camera/depth_image')
        self.declare_parameter('info_topic',  '/rgbd_camera/camera_info')
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('window', 11)  # median window size

        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic  = self.get_parameter('info_topic').value
        self.det_topic   = self.get_parameter('detections_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.window = int(self.get_parameter('window').value)

        self.last_objects = []
        self.timer = self.create_timer(0.2, self.publish_last)  # 5 Hz steady output

        self.depth_img = None
        self.cam_info = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pose = self.create_publisher(PoseStamped, '/scene/object_pose', 10)
        self.pub_json = self.create_publisher(String, '/scene/objects_json', 10)

        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.cb_depth, qos_profile_sensor_data)
        self.sub_info  = self.create_subscription(CameraInfo, self.info_topic, self.cb_info, 10)
        self.sub_det   = self.create_subscription(Detection2DArray, self.det_topic, self.cb_det, 10)

        self.get_logger().info("BBox->Pose online (depth + camera_info + detections)")

    def publish_last(self):
        self.pub_json.publish(String(data=json.dumps({"objects": self.last_objects})))

    def cb_depth(self, msg: Image):
        # depth encoding may be 32FC1 or 16UC1 depending on bridge/plugin
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_img = (depth, msg.header)

    def cb_info(self, msg: CameraInfo):
        self.cam_info = msg

    def cb_det(self, msg: Detection2DArray):
        if self.depth_img is None or self.cam_info is None:
            return

        depth, depth_header = self.depth_img

        fx = self.cam_info.k[0]
        fy = self.cam_info.k[4]
        cx = self.cam_info.k[2]
        cy = self.cam_info.k[5]

        objects = []
        for det in msg.detections:
            if not det.results:
                continue
            cls = det.results[0].hypothesis.class_id
            score = float(det.results[0].hypothesis.score)

            u = int(det.bbox.center.position.x)
            v = int(det.bbox.center.position.y)

            half = self.window // 2
            u0, u1 = max(0, u-half), min(depth.shape[1], u+half+1)
            v0, v1 = max(0, v-half), min(depth.shape[0], v+half+1)

            self.pub_json.publish(String(data=json.dumps({"objects": objects})))
            self.get_logger().info(f"det cls={cls} u={u} v={v}", throttle_duration_sec=1.0)
            self.get_logger().info(f"depth_frame={depth_header.frame_id}", throttle_duration_sec=1.0)

            patch = depth[v0:v1, u0:u1].astype(np.float32)

            # Convert if 16UC1 millimeters
            if patch.dtype != np.float32:
                patch = patch.astype(np.float32)

            # Heuristic: if values look like mm, convert to meters
            if np.nanmax(patch) > 100.0:
                patch = patch / 1000.0

            # Remove invalids
            patch = patch[np.isfinite(patch)]
            patch = patch[patch > 0.05]
            self.get_logger().info(
    f"patch stats: size={patch.size} finite={np.isfinite(patch).sum()} min={np.nanmin(patch):.3f} max={np.nanmax(patch):.3f}",
    throttle_duration_sec=1.0
)

            if patch.size == 0:
                self.get_logger().warn("Depth patch empty/invalid", throttle_duration_sec=1.0)
                continue

            Z = float(np.median(patch))
            self.get_logger().info(f"Z(m)={Z:.3f}", throttle_duration_sec=1.0)

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            pose_cam = PoseStamped()
            pose_cam.header = depth_header
            pose_cam.header.frame_id = depth_header.frame_id
            pose_cam.pose.position.x = X
            pose_cam.pose.position.y = Y
            pose_cam.pose.position.z = Z
            pose_cam.pose.orientation.w = 1.0  # position-only for now

            try:
                tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                pose_cam.header.frame_id,
                pose_cam.header.stamp,   # use sensor stamp
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

                pose_base = tf2_geometry_msgs.do_transform_pose_stamped(pose_cam, tf)
                self.get_logger().info(
    f"TF OK -> base: x={pose_base.pose.position.x:.3f} y={pose_base.pose.position.y:.3f} z={pose_base.pose.position.z:.3f}",
    throttle_duration_sec=1.0
)

            except Exception as e:
                self.get_logger().warn(f"TF failed: {e}", throttle_duration_sec=1.0)
                continue

            # publish single pose
            out = PoseStamped()
            out.header = pose_base.header
            out.pose = pose_base.pose
            # hack: carry class in frame_id suffix for quick debugging
            out.header.frame_id = f"{self.target_frame}:{cls}"
            self.pub_pose.publish(out)
            

            objects.append({
                "class": cls,
                "score": score,
                "pose": {
                    "frame": self.target_frame,
                    "x": pose_base.pose.position.x,
                    "y": pose_base.pose.position.y,
                    "z": pose_base.pose.position.z
                }
                
            })
            if len(objects) > 0:
                self.last_objects = objects

           # self.get_logger().info(f"Publishing {len(objects)} objects", throttle_duration_sec=1.0)




def main():
    rclpy.init()
    node = BBoxToPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
