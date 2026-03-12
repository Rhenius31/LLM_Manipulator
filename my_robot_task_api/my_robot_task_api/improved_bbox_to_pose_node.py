#!/usr/bin/env python3
import json
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

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
        self.declare_parameter('info_topic', '/rgbd_camera/camera_info')
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('window', 11)

        self.declare_parameter('publish_period', 0.2)
        self.declare_parameter('track_match_distance', 0.08)
        self.declare_parameter('track_timeout_sec', 1.0)
        self.declare_parameter('min_score', 0.25)
        self.declare_parameter('ema_alpha', 0.65)

        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic = self.get_parameter('info_topic').value
        self.det_topic = self.get_parameter('detections_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.window = int(self.get_parameter('window').value)

        self.publish_period = float(self.get_parameter('publish_period').value)
        self.track_match_distance = float(self.get_parameter('track_match_distance').value)
        self.track_timeout_sec = float(self.get_parameter('track_timeout_sec').value)
        self.min_score = float(self.get_parameter('min_score').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)

        self.depth_img = None
        self.cam_info = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pose = self.create_publisher(PoseStamped, '/scene/object_pose', 10)
        self.pub_json = self.create_publisher(String, '/scene/objects_json', 10)

        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.cb_depth, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.info_topic, self.cb_info, 10
        )
        self.sub_det = self.create_subscription(
            Detection2DArray, self.det_topic, self.cb_det, 10
        )

        self.tracks = {}
        self.next_track_idx = {}

        self.timer = self.create_timer(self.publish_period, self.publish_tracks)

        self.get_logger().info("BBox->Pose with tracking online")

    # --------------------------------------------------
    # callbacks
    # --------------------------------------------------

    def cb_depth(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_img = (depth, msg.header)
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}", throttle_duration_sec=1.0)

    def cb_info(self, msg: CameraInfo):
        self.cam_info = msg

    def cb_det(self, msg: Detection2DArray):
        if self.depth_img is None or self.cam_info is None:
            return

        depth, depth_header = self.depth_img

        fx = float(self.cam_info.k[0])
        fy = float(self.cam_info.k[4])
        cx = float(self.cam_info.k[2])
        cy = float(self.cam_info.k[5])

        measurements = []

        for det in msg.detections:
            if not det.results:
                continue

            cls = str(det.results[0].hypothesis.class_id).strip().lower()
            score = float(det.results[0].hypothesis.score)
            if score < self.min_score:
                continue

            u = int(det.bbox.center.position.x)
            v = int(det.bbox.center.position.y)

            if v < 0 or u < 0 or v >= depth.shape[0] or u >= depth.shape[1]:
                continue

            half = self.window // 2
            u0, u1 = max(0, u - half), min(depth.shape[1], u + half + 1)
            v0, v1 = max(0, v - half), min(depth.shape[0], v + half + 1)

            patch = depth[v0:v1, u0:u1]
            if patch.size == 0:
                continue

            patch = patch.astype(np.float32)
            finite = patch[np.isfinite(patch)]
            if finite.size == 0:
                continue

            if np.nanmax(finite) > 100.0:
                finite = finite / 1000.0

            finite = finite[finite > 0.05]
            if finite.size == 0:
                continue

            Z = float(np.median(finite))
            X = float((u - cx) * Z / fx)
            Y = float((v - cy) * Z / fy)

            pose_cam = PoseStamped()
            pose_cam.header.stamp = depth_header.stamp
            pose_cam.header.frame_id = self.camera_frame
            pose_cam.pose.position.x = X
            pose_cam.pose.position.y = Y
            pose_cam.pose.position.z = Z
            pose_cam.pose.orientation.w = 1.0

            try:
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.camera_frame,
                    pose_cam.header.stamp,
                    timeout=Duration(seconds=0.3),
                )
                pose_base = tf2_geometry_msgs.do_transform_pose_stamped(pose_cam, tf)
            except Exception as e:
                self.get_logger().warn(f"TF failed: {e}", throttle_duration_sec=1.0)
                continue

            meas = {
                "class": cls,
                "score": score,
                "pose": {
                    "frame": self.target_frame,
                    "x": float(pose_base.pose.position.x),
                    "y": float(pose_base.pose.position.y),
                    "z": float(pose_base.pose.position.z),
                }
            }
            measurements.append(meas)

        self.update_tracks(measurements)

    # --------------------------------------------------
    # tracking
    # --------------------------------------------------

    def pose_dist(self, a, b):
        dx = float(a["x"]) - float(b["x"])
        dy = float(a["y"]) - float(b["y"])
        dz = float(a["z"]) - float(b["z"])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def new_track_id(self, cls: str) -> str:
        idx = self.next_track_idx.get(cls, 0)
        self.next_track_idx[cls] = idx + 1
        return f"{cls}_{idx}"

    def update_tracks(self, measurements):
        now_ns = self.get_clock().now().nanoseconds
        unmatched_track_ids = set(self.tracks.keys())

        for meas in measurements:
            cls = meas["class"]
            pose = meas["pose"]

            best_id = None
            best_dist = 1e9

            for tid, tr in self.tracks.items():
                if tr["class"] != cls:
                    continue
                d = self.pose_dist(pose, tr["pose"])
                if d < self.track_match_distance and d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is None:
                tid = self.new_track_id(cls)
                self.tracks[tid] = {
                    "id": tid,
                    "class": cls,
                    "score": float(meas["score"]),
                    "pose": dict(pose),
                    "last_seen_ns": now_ns,
                    "hits": 1,
                }
            else:
                tr = self.tracks[best_id]
                a = self.ema_alpha
                tr["pose"]["x"] = a * pose["x"] + (1.0 - a) * tr["pose"]["x"]
                tr["pose"]["y"] = a * pose["y"] + (1.0 - a) * tr["pose"]["y"]
                tr["pose"]["z"] = a * pose["z"] + (1.0 - a) * tr["pose"]["z"]
                tr["score"] = max(float(meas["score"]), 0.7 * tr["score"] + 0.3 * float(meas["score"]))
                tr["last_seen_ns"] = now_ns
                tr["hits"] += 1
                unmatched_track_ids.discard(best_id)

        timeout_ns = int(self.track_timeout_sec * 1e9)
        stale = [
            tid for tid, tr in self.tracks.items()
            if now_ns - int(tr["last_seen_ns"]) > timeout_ns
        ]
        for tid in stale:
            del self.tracks[tid]

    # --------------------------------------------------
    # publishing
    # --------------------------------------------------

    def publish_tracks(self):
        now_ns = self.get_clock().now().nanoseconds
        timeout_ns = int(self.track_timeout_sec * 1e9)

        objects = []
        for tid, tr in self.tracks.items():
            if now_ns - int(tr["last_seen_ns"]) > timeout_ns:
                continue

            obj = {
                "id": tr["id"],
                "class": tr["class"],
                "score": round(float(tr["score"]), 4),
                "pose": {
                    "frame": tr["pose"]["frame"],
                    "x": float(tr["pose"]["x"]),
                    "y": float(tr["pose"]["y"]),
                    "z": float(tr["pose"]["z"]),
                }
            }
            objects.append(obj)

            out = PoseStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = f'{self.target_frame}:{tr["class"]}:{tr["id"]}'
            out.pose.position.x = obj["pose"]["x"]
            out.pose.position.y = obj["pose"]["y"]
            out.pose.position.z = obj["pose"]["z"]
            out.pose.orientation.w = 1.0
            self.pub_pose.publish(out)

        objects.sort(key=lambda o: (o["class"], -o["score"], o["id"]))
        self.pub_json.publish(String(data=json.dumps({"objects": objects})))


def main():
    rclpy.init()
    node = BBoxToPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()