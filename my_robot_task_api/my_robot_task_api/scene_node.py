# =========================
# 1) scene_node.py
# - Fix table Z: pose.z is CENTER of the box, not the top
# - Expose table_top_z + thickness so your pick node can match it
# =========================
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class SceneNode(Node):
    def __init__(self):
        super().__init__("scene_node")
        self.cli = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.get_logger().info("Waiting for /apply_planning_scene ...")
        self.cli.wait_for_service()

        # ---- TABLE CONFIG (match this in your PickPlaceNode) ----
        self.table_size_x = 0.8
        self.table_size_y = 1.2
        self.table_thickness = 0.05

        self.table_center_x = 0.5
        self.table_center_y = 0.0

        # This is the *TOP SURFACE* height you want (tune to your real table)
        self.table_top_z = 0.20

        # Add defaults on startup
        self.add_table()

    def apply(self, collision_objects):
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = collision_objects

        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        ok = future.result().success if future.result() else False
        self.get_logger().info(f"ApplyPlanningScene success={ok}")

    def add_table(self):
        obj = CollisionObject()
        obj.id = "table"
        obj.header.frame_id = "base_link"
        obj.operation = CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [self.table_size_x, self.table_size_y, self.table_thickness]

        pose = Pose()
        pose.position.x = self.table_center_x
        pose.position.y = self.table_center_y

        # IMPORTANT: pose.z is the CENTER of the box
        pose.position.z = self.table_top_z - (self.table_thickness / 2.0)

        pose.orientation.w = 1.0

        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        self.apply([obj])

    def add_box(self, obj_id, size_xyz, pose_xyz):
        obj = CollisionObject()
        obj.id = obj_id
        obj.header.frame_id = "base_link"
        obj.operation = CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]

        pose = Pose()
        pose.position.x = float(pose_xyz[0])
        pose.position.y = float(pose_xyz[1])
        pose.position.z = float(pose_xyz[2])
        pose.orientation.w = 1.0

        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        self.apply([obj])


def main():
    rclpy.init()
    node = SceneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
