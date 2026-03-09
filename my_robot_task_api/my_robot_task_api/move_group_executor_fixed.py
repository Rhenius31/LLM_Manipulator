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
    PlanningScene,
    AttachedCollisionObject,
)
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive


class MoveGroupExecutor(Node):
    def __init__(self):
        super().__init__("task_executor")

        self.set_parameters([
            rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)
        ])

        self.cb_group = ReentrantCallbackGroup()

        # ---- Frames / links ----
        self.group_name = "arm"
        self.base_frame = "base_link"
        self.ee_link = "tool_frame"

        # fingertip links
        self.left_tip_link = "left_finger_dist_link"
        self.right_tip_link = "right_finger_dist_link"

        # Links allowed to touch object during grasp
        self.touch_links = [
            self.ee_link,
            "left_finger_link",
            "right_finger_link",
            "left_finger_dist_link",
            "right_finger_dist_link",
            "left_finger_prox_link",
            "right_finger_prox_link",
        ]

        self._cached_tip_offset = None
        self.GRASP_BIAS_TOOL = (0.0, 0.0, 0.02)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(1.0, self._init_tip_offset_once, callback_group=self.cb_group)

        # ---- Scene ----
        self.scene = {"objects": []}
        self.create_subscription(String, "/scene/objects_json", self.cb_scene, 10, callback_group=self.cb_group)
        self.create_subscription(String, "/task_plan", self.cb_plan, 10, callback_group=self.cb_group)

        self._last_scene_publish_ns = 0
        self.scene_publish_period_ns = int(0.5 * 1e9)  # 0.5s throttle
        self._scene_object_ids = set()
        self.publish_scene_collisions = True
        self._last_place_pose = None

        # Joint states
        self.last_js = None
        self.create_subscription(JointState, "/joint_states", self.cb_js, 10, callback_group=self.cb_group)

        # ---- Gripper ----
        self.declare_parameter("gripper_action_name", "/gen3_lite_2f_gripper_controller/gripper_cmd")
        self.declare_parameter("gripper_open", 0.10)
        self.declare_parameter("gripper_closed", 0.50)
        self.declare_parameter("gripper_effort", 50.0)

        self.gripper_action_name = self.get_parameter("gripper_action_name").value
        self.gripper_open = float(self.get_parameter("gripper_open").value)
        self.gripper_closed = float(self.get_parameter("gripper_closed").value)
        self.gripper_effort = float(self.get_parameter("gripper_effort").value)

        self.gripper_client = ActionClient(self, GripperCommand, self.gripper_action_name, callback_group=self.cb_group)
        self.get_logger().info(f"Waiting for gripper action {self.gripper_action_name} ...")
        self.gripper_client.wait_for_server()
        self.get_logger().info("Connected to gripper action")

        # ---- Planning Scene service ----
        self.ps_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=self.cb_group)
        self.get_logger().info("Waiting for /apply_planning_scene ...")
        self.ps_client.wait_for_service()
        self.get_logger().info("Connected to /apply_planning_scene")

        # Optional collision topic (some setups visualize this; PlanningScene is the real authority)
        self.collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)

        # ---- MoveGroup ----
        self.client = ActionClient(self, MoveGroup, "/move_action", callback_group=self.cb_group)
        self.get_logger().info("Waiting for /move_action...")
        self.client.wait_for_server()
        self.get_logger().info("Connected to /move_action")

        # Object collision defaults (box sizes)
        self.default_object_box = {
            "cup": (0.08, 0.08, 0.10),
            "box": (0.06, 0.06, 0.06),
            "tray": (0.30, 0.20, 0.03),
        }
        self.fallback_box = (0.06, 0.06, 0.06)

        # Active object being manipulated
        self.active_object_id = None
        self.active_object_prim = None

        # Track which scene_* object we removed as duplicate for the picked instance
        self._picked_scene_dup_id = None

        # ---- Sequence state ----
        self.busy = False
        self.seq = []
        self.step_idx = 0
        self.pending_goal = None
        self.pending_result = None
        self._step_attempt = 0

        # sync step timer
        self._sync_until_ns = 0

        self.timer = self.create_timer(0.05, self.tick, callback_group=self.cb_group)

        # table collision
        self.enable_table_collision = True
        if self.enable_table_collision:
            self._published_table = False
            self.create_timer(1.0, self._publish_table_once, callback_group=self.cb_group)

    # -------------------- callbacks --------------------

    def cb_js(self, msg: JointState):
        self.last_js = msg

    def cb_scene(self, msg: String):
        try:
            self.scene = json.loads(msg.data)
        except Exception:
            return

        for obj in self.scene.get("objects", []):
            pose = obj.get("pose", {})
            self.get_logger().info(
                f"DETECTED {obj.get('class')} pose: "
                f"frame={pose.get('frame')} "
                f"x={pose.get('x')} y={pose.get('y')} z={pose.get('z')}"
            )

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_scene_publish_ns < self.scene_publish_period_ns:
            return
        self._last_scene_publish_ns = now_ns

        if self.publish_scene_collisions and (not self.busy):
            self.publish_scene_objects_as_collisions()

    def select_object_pose(self, cls: str, index: int = 0):
        matches = [o for o in self.scene.get("objects", []) if o.get("class") == cls]
        if not matches:
            return None, None
        matches.sort(key=lambda o: o.get("score", 0.0), reverse=True)
        if index >= len(matches):
            index = 0
        return matches[index]["pose"], matches[index]

    # -------------------- scene collision publishing --------------------

    def publish_scene_objects_as_collisions(self):
        objs = self.scene.get("objects", [])
        if not objs:
            return

        collision_list = []

        for i, obj in enumerate(objs):
            cls = obj.get("class", "obj")
            pose = obj.get("pose", None)
            if not pose:
                continue

            # Skip table if it causes noise 
            if cls in ("table", "tray"):
                continue

            if pose.get("frame", self.base_frame) != self.base_frame:
                self.get_logger().warn(
                    f"Skipping {cls}_{i}: pose frame {pose.get('frame')} != base_frame {self.base_frame}"
                )
                continue

            obj_id = f"scene_{cls}_{i}"
            self._scene_object_ids.add(obj_id)

            co, _prim = self.make_object_collision(obj_id, cls, pose, shrink=0.01)
            self.get_logger().info(
                f"PLANNING SCENE {obj_id}: "
                f"class={cls} x={co.primitive_poses[0].position.x:.3f} "
                f"y={co.primitive_poses[0].position.y:.3f} "
                f"z={co.primitive_poses[0].position.z:.3f} "
                f"size={co.primitives[0].dimensions}"
            )
            collision_list.append(co)

        if collision_list:
            for co in collision_list:
                self.collision_pub.publish(co)
            self.apply_world_collision_objects_async(collision_list, label="scene_update")
            self.get_logger().info(f"PlanningScene updated with {len(collision_list)} detected objects")

    # -------------------- fingertip offset + picking duplicate lookup --------------------

    def _init_tip_offset_once(self):
        if self._cached_tip_offset is not None:
            return
        off = self.compute_fingertip_midpoint_offset_tool()
        if off is not None:
            self._cached_tip_offset = off
            self.get_logger().info(
                f"Computed tool->grasp offset from fingertip midpoint: "
                f"({off[0]:.3f}, {off[1]:.3f}, {off[2]:.3f}) in {self.ee_link}"
            )

    def _dist2_pose(self, a: dict, b: dict) -> float:
        dx = float(a["x"]) - float(b["x"])
        dy = float(a["y"]) - float(b["y"])
        dz = float(a["z"]) - float(b["z"])
        return dx*dx + dy*dy + dz*dz

    def find_closest_scene_object_id(self, cls: str, pick_pose: dict):
        """
        Returns the scene_* id for the closest detected object of class `cls`.
        Avoids relying on unstable indices from score sorting.
        """
        best_id = None
        best_d2 = 1e9
        for i, obj in enumerate(self.scene.get("objects", [])):
            if obj.get("class") != cls:
                continue
            pose = obj.get("pose")
            if not pose:
                continue
            if pose.get("frame", self.base_frame) != self.base_frame:
                continue
            d2 = self._dist2_pose(pose, pick_pose)
            if d2 < best_d2:
                best_d2 = d2
                best_id = f"scene_{cls}_{i}"
        return best_id

    def compute_fingertip_midpoint_offset_tool(self):
        try:
            tl = self.tf_buffer.lookup_transform(
                self.ee_link, self.left_tip_link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3)
            )
            tr = self.tf_buffer.lookup_transform(
                self.ee_link, self.right_tip_link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3)
            )
            xl, yl, zl = tl.transform.translation.x, tl.transform.translation.y, tl.transform.translation.z
            xr, yr, zr = tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z
            return ((xl + xr) * 0.5, (yl + yr) * 0.5, (zl + zr) * 0.5)
        except Exception as e:
            self.get_logger().warn(f"Failed to compute fingertip midpoint offset: {e}")
            return None

    # -------------------- apply_planning_scene (ASYNC) --------------------

    def _on_apply_done(self, future, label: str):
        try:
            resp = future.result()
            if resp is None or not resp.success:
                self.get_logger().warn(f"{label}: apply_planning_scene failed")
            else:
                self.get_logger().info(f"{label}: apply_planning_scene ok")
        except Exception as e:
            self.get_logger().warn(f"{label}: apply_planning_scene exception: {e}")

    def apply_world_collision_objects_async(self, collision_objects, label="world_objects"):
        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects = list(collision_objects)

        req = ApplyPlanningScene.Request()
        req.scene = ps
        fut = self.ps_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_apply_done(f, label))
        return fut

    def attach_object_async(self, obj_id: str, prim: SolidPrimitive):
        aco = AttachedCollisionObject()
        aco.link_name = self.ee_link
        aco.object = CollisionObject()
        aco.object.id = obj_id
        aco.object.header.frame_id = self.ee_link
        aco.object.operation = CollisionObject.ADD
        aco.touch_links = list(self.touch_links)

        aco.object.primitives = [prim]
        p = Pose()
        p.orientation.w = 1.0
        aco.object.primitive_poses = [p]

        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.attached_collision_objects = [aco]

        req = ApplyPlanningScene.Request()
        req.scene = ps
        fut = self.ps_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_apply_done(f, f"attach({obj_id})"))
        return fut

    def detach_object_async(self, obj_id: str):
        aco = AttachedCollisionObject()
        aco.object = CollisionObject()
        aco.object.id = obj_id
        aco.object.operation = CollisionObject.REMOVE

        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.attached_collision_objects = [aco]

        req = ApplyPlanningScene.Request()
        req.scene = ps
        fut = self.ps_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_apply_done(f, f"detach({obj_id})"))
        return fut

    # -------------------- collision --------------------

    def _publish_table_once(self):
        if getattr(self, "_published_table", False):
            return
        self.add_table_collision()
        self._published_table = True

    def add_table_collision(self):
        co = CollisionObject()
        co.id = "table_volume"
        co.header.frame_id = self.base_frame

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.8, 0.6, 0.58]

        p = Pose()
        p.position.x = 0.60
        p.position.y = 0.00
        p.position.z = 0.29
        p.orientation.w = 1.0

        co.primitives.append(prim)
        co.primitive_poses.append(p)
        co.operation = CollisionObject.ADD

        self.collision_pub.publish(co)
        self.apply_world_collision_objects_async([co], label="table")
        self.get_logger().info("Published table collision volume")

    def remove_world_object_async(self, obj_id: str):
        co = CollisionObject()
        co.id = obj_id
        co.header.frame_id = self.base_frame
        co.operation = CollisionObject.REMOVE
        return self.apply_world_collision_objects_async([co], label=f"world_remove({obj_id})")
    
    def attach_and_remove_world_async(self, obj_id: str, prim: SolidPrimitive):
        aco = AttachedCollisionObject()
        aco.link_name = self.ee_link
        aco.touch_links = list(self.touch_links)

        aco.object = CollisionObject()
        aco.object.id = obj_id
        aco.object.header.frame_id = self.ee_link
        aco.object.operation = CollisionObject.ADD
        aco.object.primitives = [prim]

        p = Pose()
        p.orientation.w = 1.0
        aco.object.primitive_poses = [p]

        world_remove = CollisionObject()
        world_remove.id = obj_id
        world_remove.header.frame_id = self.base_frame
        world_remove.operation = CollisionObject.REMOVE

        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.attached_collision_objects = [aco]
        ps.world.collision_objects = [world_remove]

        req = ApplyPlanningScene.Request()
        req.scene = ps
        fut = self.ps_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_apply_done(f, f"attach_and_remove({obj_id})"))
        return fut

    def make_object_collision(self, obj_id: str, cls: str, pose_dict: dict, shrink: float = 0.01):
        sx, sy, sz = self.default_object_box.get(cls, self.fallback_box)
        sx = max(0.01, sx - shrink)
        sy = max(0.01, sy - shrink)
        sz = max(0.01, sz - shrink)

        co = CollisionObject()
        co.id = obj_id
        co.header.frame_id = self.base_frame

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(sx), float(sy), float(sz)]

        p = Pose()
        p.position.x = float(pose_dict["x"])
        p.position.y = float(pose_dict["y"])
         # If detector gives top surface z, convert to object center z
        z_det = float(pose_dict["z"])

        if cls == "box":
        # box detection looks like top surface
            p.position.z = z_det - 0.5 * sz

        elif cls == "cup":
        # cup detection often lands near the top rim
            p.position.z = z_det - 0.5 * sz

        elif cls == "tray":
        # tray detection is probably seeing the visual rim, not the base slab
        # base thickness is 0.03, rim top is about 0.03 above tray center
            p.position.z = z_det - 0.045

        else:
            p.position.z = z_det - 0.5 * sz

        p.orientation.w = 1.0

        co.primitives = [prim]
        co.primitive_poses = [p]
        co.operation = CollisionObject.ADD
        return co, prim

    # -------------------- math helpers --------------------

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

    def compute_tool_topdown_quat(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3),
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

        y_cur = Rcur[:, 1]
        y_proj = np.array([y_cur[0], y_cur[1], 0.0], dtype=float)
        n = np.linalg.norm(y_proj)
        if n < 1e-6:
            y_proj = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            y_proj /= n

        x_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        y_axis = y_proj - np.dot(y_proj, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-12
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-12

        R = np.column_stack((x_axis, y_axis, z_axis))
        return self.rot_to_quat(R)

    # -------------------- pose helpers --------------------

    def make_pose_stamped(self, pose_dict, dz=0.0):
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(pose_dict["x"])
        ps.pose.position.y = float(pose_dict["y"])
        ps.pose.position.z = float(pose_dict["z"]) + float(dz)
        ps.pose.orientation.w = 1.0
        return ps
    
    def make_grasp_pose(self, cls: str, pose_dict: dict):
        pose = dict(pose_dict)
        return pose
    # -------------------- gripper --------------------

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

    # -------------------- MoveIt constraints / goal --------------------

    def make_goal_constraints(self, pose_stamped: PoseStamped, radius: float, use_tip: bool,
                              use_topdown: bool, topdown_quat, enforce_orientation: bool):
        c = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link

        if use_tip:
            off = self._cached_tip_offset or (0.0, 0.0, 0.0)
            off = (
                off[0] + self.GRASP_BIAS_TOOL[0],
                off[1] + self.GRASP_BIAS_TOOL[1],
                off[2] + self.GRASP_BIAS_TOOL[2],
            )
            pc.target_point_offset.x = float(off[0])
            pc.target_point_offset.y = float(off[1])
            pc.target_point_offset.z = float(off[2])

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

        if use_topdown and enforce_orientation:
            q = topdown_quat
            if q is not None:
                oc = OrientationConstraint()
                oc.header.frame_id = self.base_frame
                oc.link_name = self.ee_link
                oc.orientation.x = float(q[0])
                oc.orientation.y = float(q[1])
                oc.orientation.z = float(q[2])
                oc.orientation.w = float(q[3])
                oc.absolute_x_axis_tolerance = 0.8
                oc.absolute_y_axis_tolerance = 0.8
                oc.absolute_z_axis_tolerance = 3.14
                oc.weight = 1.0
                c.orientation_constraints.append(oc)

        return c

    def send_pose_goal_constrained(self, pose_stamped: PoseStamped,
                                   radius: float, use_tip: bool,
                                   use_topdown: bool, enforce_orientation: bool):

        req = MotionPlanRequest()
        req.group_name = self.group_name

        req.pipeline_id = ""
        req.planner_id = ""
        req.num_planning_attempts = 15
        req.allowed_planning_time = 15.0
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

        req.start_state = RobotState()
        if self.last_js is not None:
            req.start_state.joint_state = self.last_js
            req.start_state.is_diff = True
        else:
            req.start_state.is_diff = True

        top_q = self.compute_tool_topdown_quat() if use_topdown else None

        req.goal_constraints = [
            self.make_goal_constraints(
                pose_stamped=pose_stamped,
                radius=radius,
                use_tip=use_tip,
                use_topdown=use_topdown,
                topdown_quat=top_q,
                enforce_orientation=enforce_orientation,
            )
        ]

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2

        return self.client.send_goal_async(goal)

    # -------------------- task sequencing --------------------

    def pick_and_place(self, pick_pose, place_pose, pick_cls="obj", pick_index=0):
        # Keep scene collisions ON (so other objects remain collidable)
        self.publish_scene_collisions = True

        pick_pose = dict(pick_pose)
        grasp_pose = self.make_grasp_pose(pick_cls, pick_pose)
        place_pose = dict(place_pose)
        self._last_place_pose = dict(place_pose)
        place_pose["z"] = max(float(place_pose["z"]), 0.55)

        # Remove ONLY the duplicate scene_* object that corresponds to this picked instance
        self._picked_scene_dup_id = self.find_closest_scene_object_id(pick_cls, pick_pose)
        if self._picked_scene_dup_id:
            self.get_logger().info(f"Removing duplicate for picked object: {self._picked_scene_dup_id}")
            self.remove_world_object_async(self._picked_scene_dup_id)
            if self._picked_scene_dup_id in self._scene_object_ids:
                self._scene_object_ids.remove(self._picked_scene_dup_id)

        # Add a dedicated pick_* collision object for grasping
        obj_id = f"pick_{pick_cls}_{pick_index}"
        co, prim = self.make_object_collision(obj_id, pick_cls, pick_pose, shrink=0.01)
        self.collision_pub.publish(co)
        self.apply_world_collision_objects_async([co], label=f"add_obj({obj_id})")

        self.active_object_id = obj_id
        self.active_object_prim = prim

        self.seq = [
            #("approach", self.make_pose_stamped(grasp_pose, dz=0.04), True),
            ("down",     self.make_pose_stamped(grasp_pose, dz=0.01), True),
            ("gripper_close", None, False),
            ("sync_scene", None, False),
            ("lift",     self.make_pose_stamped(grasp_pose, dz=0.18), False),

            ("lower",    self.make_pose_stamped(place_pose, dz=0.04), True),
            ("gripper_open", None, False),
            ("sync_scene", None, False),
            ("retreat",  self.make_pose_stamped(place_pose, dz=0.18), False),
        ]

        self.step_idx = 0
        self._step_attempt = 0
        self.busy = True
        self.pending_goal = None
        self.pending_result = None
        self._sync_until_ns = 0
        self.get_logger().info("Sequence queued")

    def tick(self):
        if not self.busy:
            return

        if self.step_idx >= len(self.seq):
            self.get_logger().info("Sequence done")
            self.busy = False
            # when idle again, cb_scene will republish scene collisions (including the released object)
            return

        name, ps, use_topdown = self.seq[self.step_idx]

        # --- sync step (non-blocking wait) ---
        if name == "sync_scene":
            if self._sync_until_ns == 0:
                self._sync_until_ns = self.get_clock().now().nanoseconds + int(0.25 * 1e9)
                self.get_logger().info(f"Step {self.step_idx+1}/{len(self.seq)}: sync_scene (250ms)")
                return
            if self.get_clock().now().nanoseconds >= self._sync_until_ns:
                self._sync_until_ns = 0
                self.step_idx += 1
            return

        # --- Gripper-only steps ---
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
                if name == "gripper_close" and self.active_object_id and self.active_object_prim:
                    # attach then remove world copy so finger contact isn't a start-state collision
                    self.attach_and_remove_world_async(self.active_object_id, self.active_object_prim)

                if name == "gripper_open" and self.active_object_id:
    # 1) detach from robot
                    self.detach_object_async(self.active_object_id)

    # 2) put it back into the world at the place pose (so it’s collidable again)
                    if self._last_place_pose is not None and self.active_object_prim is not None:
                        co = CollisionObject()
                        co.id = self.active_object_id
                        co.header.frame_id = self.base_frame
                        co.operation = CollisionObject.ADD
                        co.primitives = [self.active_object_prim]

                        p = Pose()
                        p.position.x = float(self._last_place_pose["x"])
                        p.position.y = float(self._last_place_pose["y"])
        # place pose is usually on surface -> shift to center
                        p.position.z = float(self._last_place_pose["z"]) + 0.5 * float(self.active_object_prim.dimensions[2])
                        p.orientation.w = 1.0
                        co.primitive_poses = [p]

                        self.collision_pub.publish(co)
                        self.apply_world_collision_objects_async([co], label=f"world_add({self.active_object_id})")

    # 3) clear active object tracking
                    self.active_object_id = None
                    self.active_object_prim = None
                    # After release, do nothing special: next cb_scene publish will re-add scene_* objects.

                self.pending_goal = None
                self.pending_result = None
                self.step_idx += 1
                self._step_attempt = 0
            return

        # --- Motion steps ---
        use_tip = name in ("approach", "down", "lower")

        if name == "down":
            radius = 0.012
        elif name in ("approach", "lower"):
            radius = 0.06
        else:
            radius = 0.08

        enforce_orientation = (name == "down")

        if self._step_attempt >= 1:
            enforce_orientation = False
            radius = max(radius, 0.08)

        if self.pending_goal is None and self.pending_result is None:
            self.get_logger().info(
                f"Step {self.step_idx+1}/{len(self.seq)}: {name} "
                f"(attempt {self._step_attempt+1}, r={radius:.3f}, topdown={use_topdown}, "
                f"enforce_ori={enforce_orientation}, use_tip={use_tip})"
            )
            self.pending_goal = self.send_pose_goal_constrained(
                ps, radius=radius, use_tip=use_tip, use_topdown=use_topdown, enforce_orientation=enforce_orientation
            )
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
                self.get_logger().warn(f"MoveGroup failed at '{name}', error_code={err}")

                if self._step_attempt < 1:
                    self.get_logger().warn(f"Retrying step '{name}' with relaxed constraints...")
                    self._step_attempt += 1
                    self.pending_goal = None
                    self.pending_result = None
                    return

                self.busy = False
                return

            self.pending_goal = None
            self.pending_result = None
            self.step_idx += 1
            self._step_attempt = 0

    # -------------------- plan callback --------------------

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

        pick_pose, _ = self.select_object_pose(pick["class"], int(pick.get("index", 0)))
        if not pick_pose:
            self.get_logger().error(f"No object detected for class '{pick['class']}'")
            return

        if place_on["class"] == "table":
            place_pose = {"frame": self.base_frame, "x": 0.45, "y": -0.20, "z": 0.60}
        else:
            place_pose, _ = self.select_object_pose(place_on["class"], int(place_on.get("index", 0)))

        if not place_pose:
            self.get_logger().error(f"No place target for class '{place_on['class']}'")
            return

        self.pick_and_place(
            pick_pose=pick_pose,
            place_pose=place_pose,
            pick_cls=pick["class"],
            pick_index=int(pick.get("index", 0)),
        )


def main():
    rclpy.init()
    node = MoveGroupExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()