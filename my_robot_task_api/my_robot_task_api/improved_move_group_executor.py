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

        self.group_name = "arm"
        self.base_frame = "base_link"
        self.ee_link = "tool_frame"
        self.left_tip_link = "left_finger_dist_link"
        self.right_tip_link = "right_finger_dist_link"

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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(1.0, self._init_tip_offset_once, callback_group=self.cb_group)

        self.scene = {"objects": []}
        self.last_plan = None
        self.last_js = None

        self.create_subscription(String, "/scene/objects_json", self.cb_scene, 10, callback_group=self.cb_group)
        self.create_subscription(String, "/task_plan", self.cb_plan, 10, callback_group=self.cb_group)
        self.create_subscription(JointState, "/joint_states", self.cb_js, 10, callback_group=self.cb_group)

        self._last_scene_publish_ns = 0
        self.scene_publish_period_ns = int(0.5 * 1e9)
        self._scene_object_ids = set()
        self.publish_scene_collisions = True

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

        self.ps_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=self.cb_group)
        self.get_logger().info("Waiting for /apply_planning_scene ...")
        self.ps_client.wait_for_service()

        self.collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)

        self.client = ActionClient(self, MoveGroup, "/move_action", callback_group=self.cb_group)
        self.get_logger().info("Waiting for /move_action...")
        self.client.wait_for_server()

        self.default_object_box = {
            "cup": (0.08, 0.08, 0.10),
            "box": (0.06, 0.06, 0.06),
            "box_2": (0.06, 0.06, 0.06),
            "tray": (0.30, 0.20, 0.03),
        }
        self.fallback_box = (0.06, 0.06, 0.06)

        self.enable_table_collision = True
        if self.enable_table_collision:
            self._published_table = False
            self.create_timer(1.0, self._publish_table_once, callback_group=self.cb_group)

        self.active_object_id = None
        self.active_object_prim = None
        self.active_pick_object = None
        self._picked_scene_dup_id = None
        self._last_place_pose = None

        self.busy = False
        self.seq = []
        self.step_idx = 0
        self.pending_goal = None
        self.pending_result = None
        self._step_attempt = 0
        self._sync_until_ns = 0

        self.timer = self.create_timer(0.05, self.tick, callback_group=self.cb_group)

    # --------------------------------------------------
    # basic callbacks
    # --------------------------------------------------

    def cb_js(self, msg: JointState):
        self.last_js = msg

    def cb_scene(self, msg: String):
        try:
            self.scene = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("Invalid scene JSON")
            return

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_scene_publish_ns < self.scene_publish_period_ns:
            return
        self._last_scene_publish_ns = now_ns

        if self.publish_scene_collisions and (not self.busy):
            self.publish_scene_objects_as_collisions()

    def cb_plan(self, msg: String):
        if self.busy:
            self.get_logger().warn("Executor busy, ignoring new plan")
            return

        try:
            plan = json.loads(msg.data)
        except Exception:
            self.get_logger().error("Invalid plan JSON")
            return

        self.last_plan = plan

        steps = plan.get("steps", [])
        if steps:
            if self.execute_symbolic_steps(plan):
                return

        if plan.get("intent") == "pick_and_place":
            pick = plan["objects"]["pick"]
            place_on = plan["objects"]["place_on"]

            pick_obj = self.select_object(pick.get("id"), pick["class"], int(pick.get("index", 0)))
            if pick_obj is None:
                self.get_logger().error(f"No pick object for {pick}")
                return

            if place_on["class"] == "table":
                place_pose = {"frame": self.base_frame, "x": 0.45, "y": -0.20, "z": 0.60}
            else:
                place_obj = self.select_object(place_on.get("id"), place_on["class"], int(place_on.get("index", 0)))
                if place_obj is None:
                    self.get_logger().error(f"No place target for {place_on}")
                    return
                place_pose = dict(place_obj["pose"])

            self.queue_pick_place_from_objects(pick_obj, place_pose)
            return

        self.get_logger().warn(f"Unsupported intent: {plan.get('intent')}")

    # --------------------------------------------------
    # object selection
    # --------------------------------------------------

    def select_object(self, object_id=None, cls=None, index=0):
        objs = self.scene.get("objects", [])

        if object_id:
            for o in objs:
                if o.get("id") == object_id:
                    return o

        if cls is None:
            return None

        matches = [o for o in objs if o.get("class") == cls]
        if not matches:
            return None

        matches.sort(key=lambda o: o.get("score", 0.0), reverse=True)
        if index >= len(matches):
            index = 0
        return matches[index]

    def select_object_pose(self, cls: str, index: int = 0):
        obj = self.select_object(None, cls, index)
        if obj is None:
            return None, None
        return obj["pose"], obj

    # --------------------------------------------------
    # symbolic step execution
    # --------------------------------------------------

    def execute_symbolic_steps(self, plan: dict) -> bool:
        steps = plan.get("steps", [])
        if not steps:
            return False

        low_level = []

        carrying = None
        last_place_pose = None

        for step in steps:
            skill = step.get("skill")
            args = step.get("args", {})

            if skill == "pick":
                obj = self.select_object(
                    object_id=args.get("object_id"),
                    cls=args.get("class"),
                    index=int(args.get("index", 0))
                )
                if obj is None:
                    self.get_logger().error(f"pick target not found: {args}")
                    return False

                low_level.extend(self.expand_pick(obj))
                carrying = obj

            elif skill == "place":
                if carrying is None:
                    self.get_logger().error("place requested but no object is being carried")
                    return False

                if args.get("class") == "table" or args.get("target_id") == "table_virtual_0":
                    place_pose = {"frame": self.base_frame, "x": 0.45, "y": -0.20, "z": 0.60}
                else:
                    tgt = self.select_object(
                        object_id=args.get("target_id"),
                        cls=args.get("class"),
                        index=int(args.get("index", 0))
                    )
                    if tgt is None:
                        self.get_logger().error(f"place target not found: {args}")
                        return False
                    place_pose = dict(tgt["pose"])

                low_level.extend(self.expand_place(place_pose))
                last_place_pose = dict(place_pose)

            elif skill == "move_home":
                low_level.extend(self.expand_move_home())

            elif skill == "open_gripper":
                low_level.append({"name": "gripper_open"})

            elif skill == "close_gripper":
                low_level.append({"name": "gripper_close"})

            else:
                self.get_logger().warn(f"Unsupported symbolic skill: {skill}")
                return False

        if not low_level:
            return False

        if last_place_pose is not None:
            self._last_place_pose = dict(last_place_pose)

        self.queue_sequence(low_level)
        return True

    # --------------------------------------------------
    # skill expansion
    # --------------------------------------------------

    def expand_pick(self, obj: dict):
        pick_pose = dict(obj["pose"])
        pick_cls = obj["class"]

        grasp_pose = self.make_grasp_pose(pick_cls, pick_pose)

        self.prepare_active_pick_object(obj)

        return [
            #{"name": "gripper_open"},
            #{"name": "approach", "pose": self.make_pose_stamped(grasp_pose, dz=0.01), "topdown": True},
            {"name": "down", "pose": self.make_pose_stamped(grasp_pose, dz=0.02), "topdown": True},
            {"name": "gripper_close"},
            {"name": "sync_scene"},
            {"name": "lift", "pose": self.make_pose_stamped(grasp_pose, dz=0.1), "topdown": False},
        ]

    def expand_place(self, place_pose: dict):
        pp = dict(place_pose)
        pp["z"] = max(float(pp["z"]), 0.55)

        return [
            {"name": "move_to_place", "pose": self.make_pose_stamped(pp, dz=0.04), "topdown": True},
            #{"name": "lower", "pose": self.make_pose_stamped(pp, dz=0.04), "topdown": True},
            {"name": "gripper_open"},
            {"name": "sync_scene"},
            {"name": "retreat", "pose": self.make_pose_stamped(pp, dz=0.18), "topdown": True},
        ]

    def expand_move_home(self):
        home = {"x": 0.35, "y": 0.0, "z": 0.75}
        return [{"name": "home", "pose": self.make_pose_stamped(home, dz=0.0), "topdown": False}]

    def queue_pick_place_from_objects(self, pick_obj: dict, place_pose: dict):
        low_level = []
        low_level.extend(self.expand_pick(pick_obj))
        low_level.extend(self.expand_place(place_pose))
        self._last_place_pose = dict(place_pose)
        self.queue_sequence(low_level)

    def queue_sequence(self, seq):
        self.seq = seq
        self.step_idx = 0
        self.busy = True
        self.pending_goal = None
        self.pending_result = None
        self._step_attempt = 0
        self._sync_until_ns = 0
        self.get_logger().info(f"Queued sequence with {len(seq)} steps")

    # --------------------------------------------------
    # pick object scene prep
    # --------------------------------------------------

    def prepare_active_pick_object(self, obj: dict):
        pick_cls = obj["class"]
        pick_pose = obj["pose"]
        pick_id = obj.get("id", f"{pick_cls}_0")

        self.publish_scene_collisions = True

        self._picked_scene_dup_id = self.find_closest_scene_object_id(pick_cls, pick_pose)
        if self._picked_scene_dup_id:
            self.remove_world_object_async(self._picked_scene_dup_id)
            if self._picked_scene_dup_id in self._scene_object_ids:
                self._scene_object_ids.remove(self._picked_scene_dup_id)

        obj_id = f"pick_{pick_id}"
        co, prim = self.make_object_collision(obj_id, pick_cls, pick_pose, shrink=0.01)
        self.collision_pub.publish(co)
        self.apply_world_collision_objects_async([co], label=f"add_obj({obj_id})")

        self.active_object_id = obj_id
        self.active_object_prim = prim
        self.active_pick_object = obj

    # --------------------------------------------------
    # tick loop
    # --------------------------------------------------

    def tick(self):
        if not self.busy:
            return

        if self.step_idx >= len(self.seq):
            self.get_logger().info("Sequence done")
            self.busy = False
            self.seq = []
            return

        step = self.seq[self.step_idx]
        name = step["name"]
        ps = step.get("pose")
        use_topdown = bool(step.get("topdown", False))

        if name == "sync_scene":
            if self._sync_until_ns == 0:
                self._sync_until_ns = self.get_clock().now().nanoseconds + int(0.25 * 1e9)
                self.get_logger().info(f"Step {self.step_idx+1}/{len(self.seq)}: sync_scene")
                return
            if self.get_clock().now().nanoseconds >= self._sync_until_ns:
                self._sync_until_ns = 0
                self.step_idx += 1
            return

        if name in ("gripper_close", "gripper_open"):
            self.tick_gripper_step(name)
            return

        self.tick_motion_step(name, ps, use_topdown)

    def tick_gripper_step(self, name):
        if self.pending_goal is None and self.pending_result is None:
            self.get_logger().info(f"Step {self.step_idx+1}/{len(self.seq)}: {name}")
            self.pending_goal = self.close_gripper() if name == "gripper_close" else self.open_gripper()
            return

        if self.pending_goal is not None and self.pending_goal.done() and self.pending_result is None:
            gh = self.pending_goal.result()
            if gh is None or not gh.accepted:
                self.get_logger().error("Gripper goal rejected")
                self.abort_sequence()
                return
            self.pending_result = gh.get_result_async()
            return

        if self.pending_result is not None and self.pending_result.done():
            if name == "gripper_close" and self.active_object_id and self.active_object_prim:
                self.attach_and_remove_world_async(self.active_object_id, self.active_object_prim)

            if name == "gripper_open" and self.active_object_id:
                self.detach_object_async(self.active_object_id)

                if self._last_place_pose is not None and self.active_object_prim is not None:
                    co = CollisionObject()
                    co.id = self.active_object_id
                    co.header.frame_id = self.base_frame
                    co.operation = CollisionObject.ADD
                    co.primitives = [self.active_object_prim]

                    p = Pose()
                    p.position.x = float(self._last_place_pose["x"])
                    p.position.y = float(self._last_place_pose["y"])
                    p.position.z = float(self._last_place_pose["z"]) + 0.5 * float(self.active_object_prim.dimensions[2])
                    p.orientation.w = 1.0
                    co.primitive_poses = [p]

                    self.collision_pub.publish(co)
                    self.apply_world_collision_objects_async([co], label=f"world_add({self.active_object_id})")

                self.active_object_id = None
                self.active_object_prim = None
                self.active_pick_object = None

            self.pending_goal = None
            self.pending_result = None
            self.step_idx += 1
            self._step_attempt = 0

    def tick_motion_step(self, name, ps, use_topdown):
        use_tip = name in ("approach", "down", "lower", "move_to_place")
        if name == "down":
            radius = 0.025
        elif name in ("approach", "lower", "move_to_place"):
            radius = 0.025
        else:
            radius = 0.04

        enforce_orientation = name in ("down", "lower", "approach", "move_to_place")

        if self._step_attempt >= 1:
            enforce_orientation = False
            radius = max(radius, 0.025)

        if self.pending_goal is None and self.pending_result is None:
            self.get_logger().info(
                f"Step {self.step_idx+1}/{len(self.seq)}: {name} "
                f"(attempt {self._step_attempt+1}, r={radius:.3f}, topdown={use_topdown})"
            )
            self.pending_goal = self.send_pose_goal_constrained(
                ps, radius=radius, use_tip=use_tip,
                use_topdown=use_topdown,
                enforce_orientation=enforce_orientation
            )
            return

        if self.pending_goal is not None and self.pending_goal.done() and self.pending_result is None:
            gh = self.pending_goal.result()
            if gh is None or not gh.accepted:
                self.get_logger().error("MoveGroup goal rejected")
                self.abort_sequence()
                return
            self.pending_result = gh.get_result_async()
            return

        if self.pending_result is not None and self.pending_result.done():
            res = self.pending_result.result()
            err = res.result.error_code.val if res else -999

            if err != 1:
                self.get_logger().warn(f"MoveGroup failed at '{name}', error_code={err}")
                if self._step_attempt < 1:
                    self._step_attempt += 1
                    self.pending_goal = None
                    self.pending_result = None
                    return
                self.abort_sequence()
                return

            self.pending_goal = None
            self.pending_result = None
            self.step_idx += 1
            self._step_attempt = 0

    def abort_sequence(self):
        self.busy = False
        self.seq = []
        self.pending_goal = None
        self.pending_result = None
        self._step_attempt = 0
        self._sync_until_ns = 0
        self.get_logger().error("Sequence aborted")

    # --------------------------------------------------
    # scene collision publishing
    # --------------------------------------------------

    def publish_scene_objects_as_collisions(self):
        objs = self.scene.get("objects", [])
        if not objs:
            return

        collision_list = []

        for i, obj in enumerate(objs):
            cls = obj.get("class", "obj")
            pose = obj.get("pose")
            if not pose:
                continue

            if cls in ("table"):
                continue

            if pose.get("frame", self.base_frame) != self.base_frame:
                continue

            obj_id = obj.get("id", f"scene_{cls}_{i}")
            scene_id = f"scene_{obj_id}"
            self._scene_object_ids.add(scene_id)

            co, _prim = self.make_object_collision(scene_id, cls, pose, shrink=0.01)
            self.get_logger().info(
                f"Adding collision object: id={scene_id}, class={cls}, "
                f"pose=({p.pose.position.x if False else pose['x']}, {pose['y']}, {pose['z']})"
            )
            collision_list.append(co)

        if collision_list:
            for co in collision_list:
                self.collision_pub.publish(co)
            self.apply_world_collision_objects_async(collision_list, label="scene_update")

    # --------------------------------------------------
    # geometry / pose helpers
    # --------------------------------------------------

    def _init_tip_offset_once(self):
        if self._cached_tip_offset is not None:
            return
        off = self.compute_fingertip_midpoint_offset_tool()
        if off is not None:
            self._cached_tip_offset = off

    def _dist2_pose(self, a: dict, b: dict) -> float:
        dx = float(a["x"]) - float(b["x"])
        dy = float(a["y"]) - float(b["y"])
        dz = float(a["z"]) - float(b["z"])
        return dx * dx + dy * dy + dz * dz

    def find_closest_scene_object_id(self, cls: str, pick_pose: dict):
        best_id = None
        best_d2 = 1e9
        for obj in self.scene.get("objects", []):
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
                best_id = f"scene_{obj.get('id', cls)}"
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

        pose["y"] -= 0.05
        if cls == "cup":
            pose["z"] = float(pose["z"]) + 0.00
        elif cls in ("box","box_2"):
            pose["x"] -= 0.05 
            pose["z"] = float(pose["z"]) + 0.00
        return pose

    # --------------------------------------------------
    # gripper
    # --------------------------------------------------

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

    # --------------------------------------------------
    # MoveIt goal generation
    # --------------------------------------------------

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

    def make_goal_constraints(self, pose_stamped: PoseStamped,
                              radius: float, use_tip: bool,
                              use_topdown: bool, topdown_quat,
                              enforce_orientation: bool):
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

        bv = BoundingVolume()
        sph = SolidPrimitive()
        sph.type = SolidPrimitive.SPHERE
        sph.dimensions = [float(radius)]
        bv.primitives = [sph]

        p = Pose()
        p.position.x = pose_stamped.pose.position.x
        p.position.y = pose_stamped.pose.position.y
        p.position.z = pose_stamped.pose.position.z
        p.orientation.w = 1.0
        bv.primitive_poses = [p]

        pc.constraint_region = bv
        pc.weight = 1.0
        c.position_constraints.append(pc)

        if enforce_orientation and use_topdown and topdown_quat is not None:
            oc = OrientationConstraint()
            oc.header.frame_id = self.base_frame
            oc.link_name = self.ee_link
            oc.orientation.x = topdown_quat[0]
            oc.orientation.y = topdown_quat[1]
            oc.orientation.z = topdown_quat[2]
            oc.orientation.w = topdown_quat[3]
            oc.absolute_x_axis_tolerance = 0.15
            oc.absolute_y_axis_tolerance = 0.15
            oc.absolute_z_axis_tolerance = 3.14
            oc.weight = 1.0
            c.orientation_constraints.append(oc)

        return c

    def send_pose_goal_constrained(self, pose_stamped: PoseStamped,
                                   radius: float, use_tip: bool,
                                   use_topdown: bool, enforce_orientation: bool):

        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.num_planning_attempts = 200
        req.allowed_planning_time = 180.0
        req.max_velocity_scaling_factor = 0.4
        req.max_acceleration_scaling_factor = 0.4

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

    # --------------------------------------------------
    # planning scene helpers
    # --------------------------------------------------

    def _on_apply_done(self, future, label: str):
        try:
            resp = future.result()
            if resp is None or not resp.success:
                self.get_logger().warn(f"{label}: apply_planning_scene failed")
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
        p.position.x = 0.0
        p.position.y = 0.0
        p.position.z = 0.0
        p.orientation.w = 1.0
        aco.object.primitive_poses = [p]

        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.is_diff = True
        ps.robot_state.attached_collision_objects = [aco]

        req = ApplyPlanningScene.Request()
        req.scene = ps
        fut = self.ps_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_apply_done(f, f"attach({obj_id})"))
        return fut

    def detach_object_async(self, obj_id: str):
        aco = AttachedCollisionObject()
        aco.link_name = self.ee_link
        aco.object.id = obj_id
        aco.object.operation = CollisionObject.REMOVE

        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.is_diff = True
        ps.robot_state.attached_collision_objects = [aco]

        req = ApplyPlanningScene.Request()
        req.scene = ps
        fut = self.ps_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_apply_done(f, f"detach({obj_id})"))
        return fut

    def remove_world_object_async(self, obj_id: str):
        co = CollisionObject()
        co.id = obj_id
        co.header.frame_id = self.base_frame
        co.operation = CollisionObject.REMOVE
        return self.apply_world_collision_objects_async([co], label=f"remove_world({obj_id})")

    def attach_and_remove_world_async(self, obj_id: str, prim: SolidPrimitive):
        self.attach_object_async(obj_id, prim)
        self.remove_world_object_async(obj_id)

    def _publish_table_once(self):
        if self._published_table:
            return
        self._published_table = True

        co = CollisionObject()
        co.id = "fixed_table"
        co.header.frame_id = self.base_frame
        co.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.8, 0.6, 0.65]

        p = Pose()
        p.position.x = 0.6
        p.position.y = 0.0
        p.position.z = 0.29
        p.orientation.w = 1.0

        co.primitives = [prim]
        co.primitive_poses = [p]

        self.collision_pub.publish(co)
        self.apply_world_collision_objects_async([co], label="table")

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
        p.position.y = float(pose_dict["y"])-0.05
        p.position.z = float(pose_dict["z"])

        if cls in ("box","box_2"):
            p.position.x = float(pose_dict["x"])-0.05
            p.position.z = float(pose_dict["z"])+0.04

        if cls == "cup":
            p.position.z = float(pose_dict["z"])+0.01
        p.orientation.w = 1.0


        co.primitives = [prim]
        co.primitive_poses = [p]
        co.operation = CollisionObject.ADD
        return co, prim


def main():
    rclpy.init()
    node = MoveGroupExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()