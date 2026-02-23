#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    CollisionObject,
    RobotState
)
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import BoundingVolume
from sensor_msgs.msg import JointState
from rclpy.callback_groups import ReentrantCallbackGroup


class MoveGroupExecutor(Node):
    def __init__(self):
        super().__init__('task_executor')

        # IMPORTANT: match whatever your system uses.
        # If Gazebo publishes /clock and everything else uses sim time, set True.
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                                                     rclpy.parameter.Parameter.Type.BOOL,
                                                     True)])

        self.cb_group = ReentrantCallbackGroup()

        self.group_name = "arm"
        self.base_frame = "base_link"
        self.ee_link = "tool_frame"   # plan constraints for tool_frame

        # Hardcoded "top-down" quaternion (xyzw) from your tf2_echo
        self.TOPDOWN = (0.999, -0.039, 0.003, 0.027)

        # Scene
        self.scene = {"objects": []}
        self.sub_scene = self.create_subscription(
            String, '/scene/objects_json', self.cb_scene, 10, callback_group=self.cb_group)

        self.sub_plan = self.create_subscription(
            String, '/task_plan', self.cb_plan, 10, callback_group=self.cb_group)

        self.last_js = None
        self.create_subscription(
            JointState, '/joint_states', self.cb_js, 10, callback_group=self.cb_group)

        # Collision publishing
        self.collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)
        # small delay so publisher connects before first publish
        self.create_timer(0.5, self.add_table_collision, callback_group=self.cb_group)

        # MoveGroup action client
        self.client = ActionClient(self, MoveGroup, '/move_action', callback_group=self.cb_group)
        self.get_logger().info("Waiting for /move_action...")
        self.client.wait_for_server()
        self.get_logger().info("Connected to /move_action")

        # Sequence state machine
        self.busy = False
        self.seq = []
        self.step_idx = 0
        self.pending_goal = None
        self.pending_result = None
        self.timer = self.create_timer(0.05, self.tick, callback_group=self.cb_group)

    def cb_js(self, msg: JointState):
        self.last_js = msg

    def cb_scene(self, msg: String):
        try:
            self.scene = json.loads(msg.data)
        except Exception:
            return

    def select_object_pose(self, cls: str, index: int = 0):
        matches = [o for o in self.scene.get("objects", []) if o.get("class") == cls]
        if not matches:
            return None
        matches.sort(key=lambda o: o.get("score", 0.0), reverse=True)
        if index >= len(matches):
            index = 0
        return matches[index]["pose"]  # dict {frame,x,y,z}

    def add_table_collision(self):
        # Your SDF:
        # <pose>0.6 0.0 0.15 0 0 0</pose>
        # <size>0.8 0.6 0.65</size>
        # pose is CENTER of the box, so z=0.15 is correct center.

        co = CollisionObject()
        co.id = "gazebo_table"
        co.header.frame_id = self.base_frame

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.8, 0.6, 0.65]

        p = Pose()
        p.position.x = 0.6
        p.position.y = 0.0
        p.position.z = 0.15  # center
        p.orientation.w = 1.0

        co.primitives.append(prim)
        co.primitive_poses.append(p)
        co.operation = CollisionObject.ADD

        self.collision_pub.publish(co)
        self.get_logger().info("Published table collision object to MoveIt (/collision_object)")

    def make_pose_stamped(self, pose_dict, dz=0.0):
        ps = PoseStamped()
        x = float(pose_dict["x"])
        y = float(pose_dict["y"])
        z = float(pose_dict["z"]) + dz

        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()

        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        return ps

    def make_goal_constraints(self, pose_stamped: PoseStamped, use_topdown: bool):
        c = Constraints()

        # --- Position constraint: small sphere around target ---
        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.02]  # 2cm radius (tune)

        bv = BoundingVolume()
        bv.primitives.append(sphere)

        sphere_pose = Pose()
        sphere_pose.position = pose_stamped.pose.position
        sphere_pose.orientation.w = 1.0
        bv.primitive_poses.append(sphere_pose)

        pc.constraint_region = bv
        pc.weight = 1.0
        c.position_constraints.append(pc)

        # --- Orientation constraint (optional) ---
        if use_topdown:
            oc = OrientationConstraint()
            oc.header.frame_id = self.base_frame
            oc.link_name = self.ee_link
            oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = self.TOPDOWN

            # allow some tilt; yaw free
            oc.absolute_x_axis_tolerance = 0.35
            oc.absolute_y_axis_tolerance = 0.35
            oc.absolute_z_axis_tolerance = 3.14
            oc.weight = 1.0
            c.orientation_constraints.append(oc)

        return c

    def send_pose_goal_constrained(self, pose_stamped: PoseStamped, use_topdown: bool):
        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.pipeline_id = "ompl"
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0

        # Start state
        req.start_state = RobotState()
        req.start_state.is_diff = True
        if self.last_js is not None:
            req.start_state.joint_state = self.last_js
            req.start_state.is_diff = False

        req.goal_constraints = [self.make_goal_constraints(pose_stamped, use_topdown)]

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5

        return self.client.send_goal_async(goal)

    def pick_and_place(self, pick_pose, place_pose):
        # Table top from your SDF
        TABLE_TOP_Z = 0.15 + 0.65/2  # 0.475

        # Clamp pick z so you don't target below the table top
        pick_pose = dict(pick_pose)
        pick_pose["z"] = max(pick_pose["z"], TABLE_TOP_Z + 0.02)

        place_pose = dict(place_pose)
        place_pose["z"] = TABLE_TOP_Z + 0.02

        self.seq = [
            ("pregrasp", self.make_pose_stamped(pick_pose, dz=0.15), False),  # free orientation far away
            ("approach", self.make_pose_stamped(pick_pose, dz=0.05), False),   # enforce top-down near object
            ("down",     self.make_pose_stamped(pick_pose, dz=0.02), False),   # final descend
            ("lift",     self.make_pose_stamped(pick_pose, dz=0.15), False),
            ("preplace", self.make_pose_stamped(place_pose, dz=0.15), False),
            ("lower",    self.make_pose_stamped(place_pose, dz=0.03), False),
            ("retreat",  self.make_pose_stamped(place_pose, dz=0.15), False),
        ]
        self.step_idx = 0
        self.busy = True
        self.pending_goal = None
        self.pending_result = None
        self.get_logger().info("Sequence queued")

    def tick(self):
        if not self.busy:
            return
        if self.step_idx >= len(self.seq):
            self.get_logger().info("Sequence done")
            self.busy = False
            return

        name, ps, use_topdown = self.seq[self.step_idx]

        # send next goal
        if self.pending_goal is None and self.pending_result is None:
            self.get_logger().info(f"Step {self.step_idx+1}/{len(self.seq)}: {name}")
            self.pending_goal = self.send_pose_goal_constrained(ps, use_topdown)
            return

        # wait for goal handle then result
        if self.pending_goal is not None and self.pending_goal.done() and self.pending_result is None:
            gh = self.pending_goal.result()
            if gh is None or not gh.accepted:
                self.get_logger().error("MoveGroup goal rejected")
                self.busy = False
                return
            self.pending_result = gh.get_result_async()
            return

        if self.pending_result is not None and self.pending_result.done():
            res = self.pending_result.result()
            err = res.result.error_code.val if res else -999
            if err != 1:
                self.get_logger().error(f"MoveGroup failed, error_code={err}")
                self.busy = False
                return

            # TODO: insert real gripper actions here
            if name == "approach":
                self.get_logger().info("GRIPPER CLOSE (stub)")
            if name == "lower":
                self.get_logger().info("GRIPPER OPEN (stub)")

            self.pending_goal = None
            self.pending_result = None
            self.step_idx += 1

    def cb_plan(self, msg: String):
        try:
            plan = json.loads(msg.data)
        except Exception:
            self.get_logger().error("Invalid plan JSON")
            return

        if plan.get("intent") != "pick_and_place":
            return

        pick = plan["objects"]["pick"]
        place_on = plan["objects"]["place_on"]

        pick_pose = self.select_object_pose(pick["class"], pick.get("index", 0))
        if not pick_pose:
            self.get_logger().error(f"No object detected for class '{pick['class']}'")
            return

        if place_on["class"] == "table":
            place_pose = {"frame": self.base_frame, "x": 0.45, "y": -0.20, "z": 0.475}
        else:
            place_pose = self.select_object_pose(place_on["class"], place_on.get("index", 0))

        if not place_pose:
            self.get_logger().error(f"No place target for class '{place_on['class']}'")
            return

        self.pick_and_place(pick_pose, place_pose)


def main():
    rclpy.init()
    node = MoveGroupExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
