#!/usr/bin/env python3
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import tf2_ros

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState

from control_msgs.action import GripperCommand

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    CollisionObject,
    RobotState,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive


class MoveGroupExecutor(Node):
    def __init__(self):
        super().__init__("task_executor")

        self.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time",
                rclpy.parameter.Parameter.Type.BOOL,
                True
            )
        ])

        self.cb_group = ReentrantCallbackGroup()

        # MoveIt frames/links 
        self.group_name = "arm"
        self.base_frame = "base_link"
        self.ee_link = "tool_frame"

        # fingertip links 
        self.left_tip_link = "left_finger_dist_link"
        self.right_tip_link = "right_finger_dist_link"
        self._cached_tip_offset = None

        # compute once after TF is ready
        self.create_timer(1.0, self._init_tip_offset_once, callback_group=self.cb_group)

        # small bias to push grasp point slightly "into" the object along tool z/x if needed
        self.GRASP_BIAS_TOOL = (0.0, 0.0, 0.03)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Scene
        self.scene = {"objects": []}
        self.sub_scene = self.create_subscription(
            String, "/scene/objects_json", self.cb_scene, 10, callback_group=self.cb_group
        )
        self.sub_plan = self.create_subscription(
            String, "/task_plan", self.cb_plan, 10, callback_group=self.cb_group
        )

        # Joint states 
        self.last_js = None
        self.create_subscription(
            JointState, "/joint_states", self.cb_js, 10, callback_group=self.cb_group
        )

        # Gripper (GripperActionController)
        self.declare_parameter("gripper_action_name", "/gen3_lite_2f_gripper_controller/gripper_cmd")
        self.declare_parameter("gripper_open", 0.1)      # tune for your controller (often 0.5..1.0)
        self.declare_parameter("gripper_closed", 0.5)    # tune for your controller (often 0.0)
        self.declare_parameter("gripper_effort", 50.0)

        self.gripper_action_name = self.get_parameter("gripper_action_name").value
        self.gripper_open = float(self.get_parameter("gripper_open").value)
        self.gripper_closed = float(self.get_parameter("gripper_closed").value)
        self.gripper_effort = float(self.get_parameter("gripper_effort").value)

        self.gripper_client = ActionClient(
            self, GripperCommand, self.gripper_action_name, callback_group=self.cb_group
        )
        self.get_logger().info(f"Waiting for gripper action {self.gripper_action_name} ...")
        self.gripper_client.wait_for_server()
        self.get_logger().info("Connected to gripper action")

        # Collision publishing
        self.collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)

        # publish table once after startup
        self._published_table = False
        self.create_timer(1.0, self._publish_table_once, callback_group=self.cb_group)

        # MoveGroup action client
        self.client = ActionClient(self, MoveGroup, "/move_action", callback_group=self.cb_group)
        self.get_logger().info("Waiting for /move_action...")
        self.client.wait_for_server()
        self.get_logger().info("Connected to /move_action")

        # Sequence state
        self.busy = False
        self.seq = []
        self.step_idx = 0
        self.pending_goal = None
        self.pending_result = None
        self.timer = self.create_timer(0.05, self.tick, callback_group=self.cb_group)

    # callbacks 

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
        return matches[index]["pose"]

    # fingertip offset 

    def _init_tip_offset_once(self):
        if self._cached_tip_offset is not None:
            return
        off = self.compute_fingertip_midpoint_offset_tool()
        if off is not None:
            self._cached_tip_offset = off
            self.get_logger().info(
                "Computed tool->grasp offset from fingertip midpoint: "
                f"({off[0]:.3f}, {off[1]:.3f}, {off[2]:.3f}) in tool_frame"
            )

    def compute_fingertip_midpoint_offset_tool(self):
        """
        Returns (x,y,z) = midpoint between left/right fingertip DIST links,
        expressed in tool_frame coordinates.
        """
        try:
            tl = self.tf_buffer.lookup_transform(
                self.ee_link, self.left_tip_link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            tr = self.tf_buffer.lookup_transform(
                self.ee_link, self.right_tip_link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            xl, yl, zl = tl.transform.translation.x, tl.transform.translation.y, tl.transform.translation.z
            xr, yr, zr = tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z
            return ((xl + xr) * 0.5, (yl + yr) * 0.5, (zl + zr) * 0.5)
        except Exception as e:
            self.get_logger().warn(f"Failed to compute fingertip midpoint offset: {e}")
            return None

    # collision 

    def _publish_table_once(self):
        if self._published_table:
            return
        self.add_table_collision()
        self._published_table = True

    def add_table_collision(self):
        """
        Table collision matching your SDF:
          size: 0.8 x 0.6 x 0.04 (top slab)
          pose in base_link: x=0.6, y=0.0, z=0.58 (center of slab)
        NOTE: This assumes your table is indeed defined in base_link coordinates.
        If your table is in 'world', either change base_frame back to 'world'
        OR publish collision in 'world' and ensure MoveIt uses world planning frame.
        """
        co = CollisionObject()
        co.id = "gazebo_table_top"
        co.header.frame_id = self.base_frame

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.8, 0.6, 0.04]

        p = Pose()
        p.position.x = 0.6
        p.position.y = 0.0
        p.position.z = 0.58
        p.orientation.w = 1.0

        co.primitives.append(prim)
        co.primitive_poses.append(p)
        co.operation = CollisionObject.ADD

        self.collision_pub.publish(co)
        self.get_logger().info("Published table collision object (/collision_object)")

    #  math helpers 

    def quat_to_rot(self, q):
        x, y, z, w = q
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        return np.array([
            [1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy)],
            [2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)],
            [2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy)],
        ], dtype=float)

    def rot_to_quat(self, R):
        tr = float(np.trace(R))
        if tr > 0.0:
            S = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S

        q = np.array([x, y, z, w], dtype=float)
        q /= np.linalg.norm(q) + 1e-12
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def compute_tool_topdown_quat_world(self):
        """
        Force tool_frame +X to point DOWN (base_frame -Z).
        Preserve yaw by using current tool +Y projected onto base XY.
        This matches your setup where tool_frame +X points forward.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed for topdown: {e}")
            return None

        q_cur = (
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        )
        Rcur = self.quat_to_rot(q_cur)

    # Current tool +Y in base frame (helps preserve yaw / jaw orientation)
        y_cur = Rcur[:, 1]
        y_proj = np.array([y_cur[0], y_cur[1], 0.0], dtype=float)
        n = np.linalg.norm(y_proj)
        if n < 1e-6:
            y_proj = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            y_proj /= n

    # tool points down
        x_axis = np.array([0.0, 0.0, -1.0], dtype=float)

    # Make y orthogonal to x (Gram–Schmidt)
        y_axis = y_proj - np.dot(y_proj, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-12

    # Right-handed frame: z = x × y
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-12

        R = np.column_stack((x_axis, y_axis, z_axis))
        return self.rot_to_quat(R)


    # pose helpers 

    def make_pose_stamped(self, pose_dict, dz=0.0):
        """
        Assumes pose_dict is already in base_frame coordinates.
        If your detector publishes in another frame, transform it before calling this.
        """
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(pose_dict["x"])
        ps.pose.position.y = float(pose_dict["y"])
        ps.pose.position.z = float(pose_dict["z"]) + dz
        ps.pose.orientation.w = 1.0
        return ps

    #  gripper 

    def send_gripper(self, position: float, effort: float = None):
        if effort is None:
            effort = self.gripper_effort

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(effort)
        return self.gripper_client.send_goal_async(goal)

    def open_gripper(self):
        return self.send_gripper(self.gripper_open)

    def close_gripper(self):
        return self.send_gripper(self.gripper_closed)

    # MoveIt constraints / goal

    def make_goal_constraints(
        self,
        pose_stamped: PoseStamped,
        use_topdown: bool,
        radius: float,
        use_tip: bool,
        topdown_quat=None,
    ):
        c = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link

        if use_tip:
            off = self._cached_tip_offset
            if off is None:
                off = self.compute_fingertip_midpoint_offset_tool()
            if off is None:
                off = (0.0, 0.0, 0.0)

            off = (
                off[0] + self.GRASP_BIAS_TOOL[0],
                off[1] + self.GRASP_BIAS_TOOL[1],
                off[2] + self.GRASP_BIAS_TOOL[2],
            )
            pc.target_point_offset.x = float(off[0])
            pc.target_point_offset.y = float(off[1])
            pc.target_point_offset.z = float(off[2])
        else:
            pc.target_point_offset.x = 0.0
            pc.target_point_offset.y = 0.0
            pc.target_point_offset.z = 0.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(radius)]

        bv = BoundingVolume()
        bv.primitives.append(sphere)

        sphere_pose = Pose()
        sphere_pose.position = pose_stamped.pose.position
        sphere_pose.orientation.w = 1.0
        bv.primitive_poses.append(sphere_pose)

        pc.constraint_region = bv
        pc.weight = 1.0
        c.position_constraints.append(pc)

        if use_topdown:
            q = topdown_quat if topdown_quat is not None else self.compute_tool_topdown_quat_world()
            if q is not None:
                oc = OrientationConstraint()
                oc.header.frame_id = self.base_frame
                oc.link_name = self.ee_link
                oc.orientation.x = float(q[0])
                oc.orientation.y = float(q[1])
                oc.orientation.z = float(q[2])
                oc.orientation.w = float(q[3])
                # tolerances
                oc.absolute_x_axis_tolerance = 1.0
                oc.absolute_y_axis_tolerance = 1.0
                oc.absolute_z_axis_tolerance = 3.14
                oc.weight = 1.0
                c.orientation_constraints.append(oc)
            else:
                self.get_logger().warn("Topdown TF failed; skipping orientation constraint.")

        return c

    def send_pose_goal_constrained(self, pose_stamped: PoseStamped, use_topdown: bool, radius: float, use_tip: bool):
        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.pipeline_id = "ompl"
        req.num_planning_attempts = 20
        req.allowed_planning_time = 20.0

        # Seed start state from /joint_states to avoid stale/unknown start state
        req.start_state = RobotState()
        if self.last_js is not None:
            req.start_state.joint_state = self.last_js
            req.start_state.is_diff = False
        else:
            req.start_state.is_diff = True

        top_q = None

        req.goal_constraints = [
            self.make_goal_constraints(pose_stamped, use_topdown, radius, use_tip, top_q)
        ]
        

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5
        return self.client.send_goal_async(goal)

    # -task sequencing 

    def pick_and_place(self, pick_pose, place_pose):
        # Table geometry 
        TABLE_TOP_CENTER_Z = 0.58
        TABLE_THICKNESS = 0.04
        TABLE_TOP_Z = TABLE_TOP_CENTER_Z + TABLE_THICKNESS / 2.0

        pick_pose = dict(pick_pose)
        place_pose = dict(place_pose)

        # place slightly above table surface
        place_pose["z"] = TABLE_TOP_Z + 0.02

        self.seq = [
            ("pregrasp", self.make_pose_stamped(pick_pose, dz=0.10), True),
            ("approach", self.make_pose_stamped(pick_pose, dz=0.05), True),
            ("down",     self.make_pose_stamped(pick_pose, dz=0.05), True),
            ("gripper_close", None, False),
            ("lift",     self.make_pose_stamped(pick_pose, dz=0.15), True),

            ("preplace", self.make_pose_stamped(place_pose, dz=0.15), True),
            ("lower",    self.make_pose_stamped(place_pose, dz=0.05), True),
            ("gripper_open", None, False),
            ("retreat",  self.make_pose_stamped(place_pose, dz=0.15), True),
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

        # Gripper-only steps
        if name in ("gripper_close", "gripper_open"):
            if self.pending_goal is None and self.pending_result is None:
                self.get_logger().info(f"Step {self.step_idx+1}/{len(self.seq)}: {name}")
                self.pending_goal = self.close_gripper() if name == "gripper_close" else self.open_gripper()
                return

            if self.pending_goal is not None and self.pending_goal.done() and self.pending_result is None:
                gh = self.pending_goal.result()
                if gh is None or not gh.accepted:
                    self.get_logger().error("Gripper goal rejected")
                    self.busy = False
                    return
                self.pending_result = gh.get_result_async()
                return

            if self.pending_result is not None and self.pending_result.done():

                self.pending_goal = None
                self.pending_result = None
                self.step_idx += 1
            return


        use_tip = name in ("approach", "down", "lower")

        # Tight radius for precise contact steps, looser for travel steps
        if name in ("down", "lower"):
            radius = 0.01
        elif name == "approach":
            radius = 0.03
        else:
            radius = 0.08

        if self.pending_goal is None and self.pending_result is None:
            # Debug print tool/tip/target
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, self.ee_link, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                cx, cy, cz = tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z
                q = (
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w
                )
                R = self.quat_to_rot(q)
                off = self._cached_tip_offset if self._cached_tip_offset is not None else (0.0, 0.0, 0.0)
                if use_tip:
                    off = (off[0] + self.GRASP_BIAS_TOOL[0],
                           off[1] + self.GRASP_BIAS_TOOL[1],
                           off[2] + self.GRASP_BIAS_TOOL[2])
                else:
                    off = (0.0, 0.0, 0.0)
                tip = np.array([cx, cy, cz]) + R @ np.array(off, dtype=float)

                tx, ty, tz = ps.pose.position.x, ps.pose.position.y, ps.pose.position.z
                self.get_logger().info(
                    f"{name}: tool=({cx:.3f},{cy:.3f},{cz:.3f}) tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) "
                    f"target=({tx:.3f},{ty:.3f},{tz:.3f}) r={radius:.3f} topdown={use_topdown} use_tip={use_tip}"
                )
            except Exception:
                pass

            self.get_logger().info(f"Step {self.step_idx+1}/{len(self.seq)}: {name}")
            self.pending_goal = self.send_pose_goal_constrained(ps, use_topdown, radius, use_tip)
            return

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

            self.pending_goal = None
            self.pending_result = None
            self.step_idx += 1

    # plan callback 

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

        # If placing on table, use a fixed pose (in base_frame coordinates)
        if place_on["class"] == "table":
            place_pose = {"frame": self.base_frame, "x": 0.45, "y": -0.20, "z": 0.60}
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


if __name__ == "__main__":
    main()
