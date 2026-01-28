# =========================
# 3) fake_detector_node.py
# - Start marker just ABOVE table top (so you don't accidentally command inside table)
# - Keep publishing at 10 Hz
# =========================
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer


class FakeDetector(Node):
    def __init__(self):
        super().__init__("fake_detector_node")

        self.frame_id = "base_link"
        self.pub = self.create_publisher(PoseStamped, "/detected_object_pose", 10)

        # MUST MATCH scene_node.py (table top)
        self.table_top_z = 0.30

        # Store last pose (start above table)
        self.last_pose = PoseStamped()
        self.last_pose.header.frame_id = self.frame_id
        self.last_pose.pose.position.x = 0.35
        self.last_pose.pose.position.y = 0.30
        self.last_pose.pose.position.z = self.table_top_z + 0.08  # 8 cm above table
        self.last_pose.pose.orientation.w = 1.0

        self.server = InteractiveMarkerServer(self, "fake_object_marker")

        im = InteractiveMarker()
        im.header.frame_id = self.frame_id
        im.name = "object_1"
        im.description = "Detected Object"
        im.scale = 0.4
        im.pose = self.last_pose.pose

        self._add_visuals(im)
        self._add_6dof_controls(im)

        self.server.insert(im)
        self.server.setCallback(im.name, self.process_feedback)
        self.server.applyChanges()

        # publish continuously (10 Hz)
        self.timer = self.create_timer(0.1, self.publish_pose)

        self.get_logger().info("FakeDetector ready ")
        self.get_logger().info("Topic: /detected_object_pose (publishing at 10 Hz)")
        self.get_logger().info("RViz: Add InteractiveMarkers, topic /fake_object_marker/update")

    def publish_pose(self):
        self.last_pose.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.last_pose)

    def _add_visuals(self, im: InteractiveMarker):
        cube = Marker()
        cube.type = Marker.CUBE
        cube.scale.x = 0.05
        cube.scale.y = 0.05
        cube.scale.z = 0.05
        cube.color.r = 0.1
        cube.color.g = 0.1
        cube.color.b = 0.9
        cube.color.a = 1.0

        cube_ctrl = InteractiveMarkerControl()
        cube_ctrl.always_visible = True
        cube_ctrl.markers.append(cube)
        im.controls.append(cube_ctrl)

        text = Marker()
        text.type = Marker.TEXT_VIEW_FACING
        text.text = im.name
        text.pose.position.z = 0.08
        text.scale.z = 0.06
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0

        text_ctrl = InteractiveMarkerControl()
        text_ctrl.always_visible = True
        text_ctrl.markers.append(text)
        im.controls.append(text_ctrl)

    def _add_6dof_controls(self, im: InteractiveMarker):
        def add_ctrl(name, mode, ox, oy, oz):
            c = InteractiveMarkerControl()
            c.name = name
            c.interaction_mode = mode
            c.orientation.w = 1.0
            c.orientation.x = float(ox)
            c.orientation.y = float(oy)
            c.orientation.z = float(oz)
            im.controls.append(c)

        add_ctrl("move_x", InteractiveMarkerControl.MOVE_AXIS, 1, 0, 0)
        add_ctrl("move_y", InteractiveMarkerControl.MOVE_AXIS, 0, 1, 0)
        add_ctrl("move_z", InteractiveMarkerControl.MOVE_AXIS, 0, 0, 1)
        add_ctrl("rot_x", InteractiveMarkerControl.ROTATE_AXIS, 1, 0, 0)
        add_ctrl("rot_y", InteractiveMarkerControl.ROTATE_AXIS, 0, 1, 0)
        add_ctrl("rot_z", InteractiveMarkerControl.ROTATE_AXIS, 0, 0, 1)

    def process_feedback(self, feedback):
        self.last_pose.pose = feedback.pose


def main():
    rclpy.init()
    node = FakeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
