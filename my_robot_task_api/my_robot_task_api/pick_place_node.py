#!/usr/bin/env python3
import math
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

from my_robot_task_api.motion_node import MotionBase


class PickPlaceNode(MotionBase):
    def __init__(self):
        super().__init__(node_name="pick_place_node")

        self.busy = False
        self.latest_pose: PoseStamped | None = None

        self.create_subscription(PoseStamped, "/detected_object_pose", self.on_pose, 10)
        self.create_subscription(Bool, "/pick_now", self.on_pick_now, 10)

        # Place target 
        self.place_x = 0.25
        self.place_y = 0.30
        self.place_z = 0.35

        # Heights 
        self.approach_h = 0.10
        self.grasp_h = 0.009
        self.lift_h = 0.12

        self.get_logger().info("PickPlaceNode ready ")
        self.get_logger().info("1) Drag marker in RViz to set object pose.")
        self.get_logger().info("2) Trigger:")
        self.get_logger().info("   ros2 topic pub /pick_now std_msgs/msg/Bool '{data: true}' -1")

    def on_pose(self, msg: PoseStamped):
        self.latest_pose = msg

    def on_pick_now(self, msg: Bool):
        if not msg.data:
            return

        if self.busy:
            self.get_logger().warn("Already executing. Ignoring trigger.")
            return

        if self.latest_pose is None:
            self.get_logger().error("No /detected_object_pose received yet.")
            return

        self.busy = True
        pose_copy = self.latest_pose  # snapshot
        threading.Thread(target=self._run_sequence, args=(pose_copy,), daemon=True).start()

    def _run_sequence(self, obj_pose: PoseStamped):
        try:
            self.get_logger().info("Starting pick & place sequence...")

            # Optional: safe pre-pick pose (keeps robot consistent)
          #  ok = self.move_pose_via_ik_controller_sync(0.35, 0.0, 0.25, preset="carry", duration_sec=3.0)
            #if not ok:
             #   self.get_logger().error("Failed to move to safe pose ")
              #  return

            ok = self.execute_pick_place(obj_pose)
            if ok:
                self.get_logger().info("Pick & place done ")
            else:
                self.get_logger().error("Pick & place failed ")
        finally:
            self.busy = False
    def execute_pick_place(self, obj_pose: PoseStamped) -> bool:
        x = float(obj_pose.pose.position.x)
        y = float(obj_pose.pose.position.y)
        z = float(obj_pose.pose.position.z)

        self.get_logger().info(
            f"Object pose: x={x:.3f} y={y:.3f} z={z:.3f} frame={obj_pose.header.frame_id}"
        )

        def must(ok: bool, step: str) -> bool:
            if not ok:
                self.get_logger().error(f"Step failed : {step}")
                return False
            self.get_logger().info(f"Step OK : {step}")
            return True

        approach_z = z + self.approach_h
        grasp_z = z + self.grasp_h
        lift_z = z + self.lift_h
        align_z= approach_z + 0.1

    # --- PICK ---
        if not must(self.gripper_sync("open"), "gripper open"):
            return False

    # 1) Go above object with carry
        if not must(
            self.move_pose_via_ik_controller_sync(x, y, approach_z, preset="carry", duration_sec=3.0),
            "approach above object (carry)"
        ):
            return False

    # 2) Rotate tool DOWN at same XYZ
        if not must(
            self.align_down_with_yaw_search_sync(x, y, align_z, pitch=math.pi, duration_sec=2.0),
            "align gripper down (same XYZ)"
        ):
            return False

    # 3) Descend with gripper down
        if not must(
            self.move_pose_via_ik_controller_sync(x, y, grasp_z, preset="grasp_down", duration_sec=2.5),
            "descend to grasp (down)"
        ):
            return False

        if not must(self.gripper_sync("close"), "gripper close"):
            return False

    # 4) Lift back up (carry)
        #if not must(
         #   self.move_pose_via_ik_controller_sync(x, y, lift_z, preset="carry", duration_sec=3.0),
          #  "lift (carry)"
    #):
     #       return False

    # --- PLACE ---
        place_approach_z = self.place_z + self.approach_h
        place_grasp_z = self.place_z + self.grasp_h
        place_lift_z = self.place_z + self.lift_h
        place_align_z= place_approach_z + 0.05

    # 1) Go above place with carry
        if not must(
            self.move_pose_via_ik_controller_sync(self.place_x, self.place_y, place_approach_z,
                                              preset="carry", duration_sec=3.0),
            "move above place (carry)"
        ):
            return False

    # 2) Rotate tool DOWN at same XYZ (place)
        if not must(
            self.align_down_with_yaw_search_sync(self.place_x, self.place_y, place_align_z,pitch=math.pi, duration_sec=2.0),
            "align gripper down at place (same XYZ)"
        ):
            return False

    # 3) Descend to place with tool down
        if not must(
            self.move_pose_via_ik_controller_sync(self.place_x, self.place_y, place_grasp_z,
                                              preset="place_down", duration_sec=3.0),
            "lower to place (down)"
        ):
            return False

        if not must(self.gripper_sync("open"), "release"):
            return False

    # 4) Lift away (carry)
        if not must(
            self.move_pose_via_ik_controller_sync(self.place_x, self.place_y, place_lift_z,
                                              preset="carry", duration_sec=3.0),
            "retreat (carry)"
    ):
            return False

        return True


def main():
    rclpy.init()
    node = PickPlaceNode()

    # Multi-threaded executor prevents deadlocks with sync service/action calls
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
