#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene


class SceneObjectAdder(Node):
    def __init__(self):
        super().__init__("scene_object_adder")

        self.cli = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene"
        )

        self.get_logger().info("Waiting for /apply_planning_scene ...")
        self.cli.wait_for_service()

        self.add_box()

    def add_box(self):
        box = CollisionObject()
        box.id = "demo_cube"
        box.header.frame_id = "base_link"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.05, 0.05, 0.05]  # 5 cm cube

        pose = Pose()
        pose.position.x = 0.45
        pose.position.y = 0.0
        pose.position.z = 0.05  # on table height
        pose.orientation.w = 1.0

        box.primitives.append(primitive)
        box.primitive_poses.append(pose)
        box.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(box)

        req = ApplyPlanningScene.Request()
        req.scene = scene

        self.cli.call_async(req)
        self.get_logger().info("Added demo cube to planning scene ✅")


def main():
    rclpy.init()
    node = SceneObjectAdder()
    rclpy.spin_once(node, timeout_sec=1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
