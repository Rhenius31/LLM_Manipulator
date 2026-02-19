#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.task import Future
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from moveit_msgs.msg import MotionPlanRequest, WorkspaceParameters
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState, JointConstraint
from sensor_msgs.msg import JointState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


class MoveGroupExecutor(Node):
    def __init__(self):
        super().__init__('task_executor')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, False)])

        self.group_name = "arm"
        self.base_frame = "base_link"
        self.collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)
        self.add_table_collision()


        self.scene = {"objects": []}
        self.cb_group = ReentrantCallbackGroup()



        self.sub_scene = self.create_subscription(String, '/scene/objects_json', self.cb_scene, 10, callback_group=self.cb_group)
        self.sub_plan  = self.create_subscription(String, '/task_plan', self.cb_plan, 10, callback_group=self.cb_group)

        self.busy = False
        self.seq = []
        self.step_idx = 0
        self.current_pose = None
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz state machine
        self.pending_ik = None
        self.pending_goal = None
        self.pending_result = None


        

        self.group_name = "arm"
        self.base_frame = "base_link"
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik', callback_group=self.cb_group)
        self.get_logger().info("Waiting for /compute_ik ...")
        self.ik_cli.wait_for_service()
        self.get_logger().info("Connected to /compute_ik")

        # MoveGroup action client
        self.client = ActionClient(self, MoveGroup, '/move_action', callback_group=self.cb_group)
        self.get_logger().info("Waiting for /move_action...")
        self.client.wait_for_server()
        self.get_logger().info("Connected to /move_action (MoveIt)")

        from sensor_msgs.msg import JointState
        self.last_js = None
        self.create_subscription(JointState, '/joint_states', self.cb_js, 10, callback_group=self.cb_group)

    def cb_js(self, msg: JointState):
        self.last_js = msg

        

    def add_table_collision(self):
        co = CollisionObject()
        co.id = "gazebo_table"
        co.header.frame_id = self.base_frame  # use "base_link" (since your TF has world->base_link as identity)

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.8, 0.6, 0.68]  # x, y, z

        p = Pose()
        p.position.x = 0.6
        p.position.y = 0.0
        p.position.z = 0.15
        p.orientation.w = 1.0

        co.primitives.append(prim)
        co.primitive_poses.append(p)
        co.operation = CollisionObject.ADD

        self.collision_pub.publish(co)
        self.get_logger().info("Published table collision object to MoveIt")

    def send_joint_goal(self, joint_map: dict):
        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.pipeline_id = "ompl"
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.start_state.is_diff = True

        if self.last_js is not None:
            req.start_state.joint_state = self.last_js
            req.start_state.is_diff = False
        else:
            req.start_state.is_diff = True


        c = Constraints()
        for name, pos in joint_map.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)

        req.goal_constraints = [c]

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5

        return self.client.send_goal_async(goal)



    def cb_scene(self, msg: String):
        try:
            self.scene = json.loads(msg.data)
        except Exception:
            pass

    def select_object_pose(self, cls: str, index: int = 0):
        matches = [o for o in self.scene.get("objects", []) if o.get("class") == cls]
        if not matches:
            return None
        matches.sort(key=lambda o: o.get("score", 0.0), reverse=True)
        if index >= len(matches):
            index = 0
        return matches[index]["pose"]  # dict {frame,x,y,z}

    def make_pose_stamped(self, pose_dict, dz=0.0):
        ps = PoseStamped()

        ps.pose.position.x = float(pose_dict["x"])
        ps.pose.position.y = float(pose_dict["y"])
        ps.pose.position.z = float(pose_dict["z"]) + dz

        # For now keep neutral orientation; later we’ll set top-down quaternion
        ps.pose.orientation.x = -0.002
        ps.pose.orientation.y = -0.006
        ps.pose.orientation.z = -0.263
        ps.pose.orientation.w =  0.965
        return ps


    def send_pose_goal(self, pose_stamped: PoseStamped):
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()

        joints = self.compute_ik(pose_stamped)
        if joints is None:
            return None
        return self.send_joint_goal(joints)



    def send_pose_goal_blocking(self, pose_stamped: PoseStamped, timeout_sec=10.0):
        goal_future = self.send_pose_goal(pose_stamped)
        if goal_future is None:
            self.get_logger().error("Skipping goal because IK failed")
            return False

        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=timeout_sec)
        goal_handle = goal_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveGroup goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        result = result_future.result()

        if result is None:
            self.get_logger().error("No result received (timeout?)")
            return False

        err = result.result.error_code.val
        if err != 1:
            self.get_logger().error(f"MoveGroup failed, error_code={err}")
            return False

        return True

    
    def pick_and_place(self, pick_pose, place_pose):
        self.seq = [
        ("pregrasp", self.make_pose_stamped(pick_pose, dz=0.12)),
        ("approach", self.make_pose_stamped(pick_pose, dz=0.03)),
        ("lift",     self.make_pose_stamped(pick_pose, dz=0.18)),
        ("preplace", self.make_pose_stamped(place_pose, dz=0.15)),
        ("lower",    self.make_pose_stamped(place_pose, dz=0.06)),
        ("retreat",  self.make_pose_stamped(place_pose, dz=0.18)),
    ]
        self.step_idx = 0
        self.busy = True
        self.get_logger().info("Sequence queued")

    def tick(self):
        if not self.busy:
            return
        if self.step_idx >= len(self.seq):
            self.get_logger().info("Sequence done ")
            self.busy = False
            return

        name, ps = self.seq[self.step_idx]

    # If nothing in-flight, start IK for this step
        if self.pending_ik is None and self.pending_goal is None and self.pending_result is None:
            self.get_logger().info(f"Step {self.step_idx+1}/{len(self.seq)}: {name}")
            self.start_ik(ps)
            return

    # If IK finished and produced a goal future, wait for goal handle
        if self.pending_goal is not None and self.pending_goal.done() and self.pending_result is None:
            goal_handle = self.pending_goal.result()
            if goal_handle is None or not goal_handle.accepted:
                self.get_logger().error("MoveGroup goal rejected")
                self.fail_step(name)
                return
            self.pending_result = goal_handle.get_result_async()
            return

    # If action result finished, advance step
        if self.pending_result is not None and self.pending_result.done():
            res = self.pending_result.result()
            if res is None:
                self.get_logger().error("No action result")
                self.fail_step(name)
                return
            err = res.result.error_code.val
            if err != 1:
                self.get_logger().error(f"MoveGroup failed, error_code={err}")
                self.fail_step(name)
                return

            if name == "approach":
                self.get_logger().info("GRIPPER CLOSE (stub)")
            if name == "lower":
                self.get_logger().info("GRIPPER OPEN (stub)")

        # clear and advance
            self.pending_ik = None
            self.pending_goal = None
            self.pending_result = None
            self.step_idx += 1

    def fail_step(self, name):
        self.get_logger().error(f"Step failed: {name}")
        self.busy = False
        self.pending_ik = None
        self.pending_goal = None
        self.pending_result = None
        
    def start_ik(self, pose_stamped: PoseStamped):
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(
        f"IK target: x={pose_stamped.pose.position.x:.3f} "
        f"y={pose_stamped.pose.position.y:.3f} "
        f"z={pose_stamped.pose.position.z:.3f}"
)


        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = "tool_frame"
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.timeout.sec = 3
        req.ik_request.avoid_collisions = True
        req.ik_request.robot_state = RobotState()
        req.ik_request.robot_state.is_diff = True
        if self.last_js is not None:
            req.ik_request.robot_state.joint_state = self.last_js

        self.pending_ik = self.ik_cli.call_async(req)
        self.pending_ik.add_done_callback(self.on_ik_done)

    def on_ik_done(self, fut):
        try:
            resp = fut.result()
        except Exception as e:
            self.get_logger().error(f"IK call exception: {e}")
            self.fail_step("ik")
            return

        if resp is None or resp.error_code.val != 1:
            code = None if resp is None else resp.error_code.val
            self.get_logger().error(f"IK failed/timeout error_code={code}")
            self.fail_step("ik")
            return

        js = resp.solution.joint_state
        arm_joints = {"joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"}
        joint_map = {n:p for n,p in zip(js.name, js.position) if n in arm_joints}
        self.get_logger().info(f"IK OK joints: {list(joint_map.keys())}")


    # Send MoveGroup goal async (store future)
        self.pending_goal = self.send_joint_goal(joint_map)
        self.pending_ik = None



    def cb_plan(self, msg: String):
        try:
            plan = json.loads(msg.data)
        except Exception:
            self.get_logger().error("Invalid plan JSON")
            return

        if plan.get("intent") != "pick_and_place":
            self.get_logger().warn(f"Unhandled intent: {plan.get('intent')}")
            return

        pick = plan["objects"]["pick"]
        place_on = plan["objects"]["place_on"]

        pick_pose = self.select_object_pose(pick["class"], pick.get("index", 0))
        if not pick_pose:
            self.get_logger().error(f"No object detected for class '{pick['class']}'")
            return

        if place_on["class"] == "table":
            # demo fixed spot on table (tune z to your table height)
            place_pose = {"frame": self.base_frame, "x": 0.45, "y": -0.20, "z": 0.675}
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