#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Quaternion
from trajectory_msgs.msg import JointTrajectoryPoint

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState

from control_msgs.action import FollowJointTrajectory
from control_msgs.action import GripperCommand



GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0
GRIPPER_EFFORT = 100.0


def quat_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


ORIENTATIONS = {
    "carry": quat_from_rpy(0.0, 0.0, 0.0),
    "grasp_down": quat_from_rpy(0.0, math.pi, 0.0),
    "place_down": quat_from_rpy(0.0, math.pi, 0.0),
}


class MotionBase(Node):
    """
    Reusable motion base:
      - Pose -> IK (/compute_ik) -> joint trajectory controller execution
      - Gripper command action

    This avoids MoveIt ExecuteTrajectory problems and uses the proven controller:
      /joint_trajectory_controller/follow_joint_trajectory
    """
    def __init__(self, node_name="motion_base"):
        super().__init__(node_name)

        # Robot config
        self.group_name = "arm"
        self.planning_frame = "base_link"
        self.eef_link = "end_effector_link"
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

        # Interfaces
        self.ik_service = "/compute_ik"
        self.jtc_action = "/joint_trajectory_controller/follow_joint_trajectory"
        self.gripper_action = "/gen3_lite_2f_gripper_controller/gripper_cmd"

        # Callback group to avoid deadlocks (re-entrant)
        self.cb_group = ReentrantCallbackGroup()

        # Joint state seed for IK
        self._last_joint_state: JointState | None = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 50)

        # Clients
        self.ik_srv = self.create_client(GetPositionIK, self.ik_service, callback_group=self.cb_group)
        self.jtc_client = ActionClient(self, FollowJointTrajectory, self.jtc_action, callback_group=self.cb_group)
        self.gripper_client = ActionClient(self, GripperCommand, self.gripper_action, callback_group=self.cb_group)

        # Wait for servers
        self.get_logger().info(f"Waiting for IK service {self.ik_service} ...")
        self.ik_srv.wait_for_service()
        self.get_logger().info("Connected to /compute_ik ")

        self.get_logger().info(f"Waiting for controller action {self.jtc_action} ...")
        self.jtc_client.wait_for_server()
        self.get_logger().info("Connected to joint_trajectory_controller ")

        self.get_logger().info(f"Waiting for gripper action {self.gripper_action} ...")
        self.gripper_client.wait_for_server()
        self.get_logger().info("Connected to gripper controller ")

    def _on_joint_state(self, msg: JointState):
        self._last_joint_state = msg

    # -
    # Pose helper
    # -
    def pose_from_xyz_preset(self, x, y, z, preset="carry") -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.planning_frame
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)
        ps.pose.orientation.w = 1.0   # neutral quaternion

        #ps.pose.orientation = ORIENTATIONS.get(preset, ORIENTATIONS["carry"])
        return ps
    def move_xyz_unconstrained_sync(self, x, y, z, duration_sec=3.0) -> bool:
        """
        Move to XYZ with orientation unconstrained (identity quaternion).
        """
        self.get_logger().info(f"[POSE->IK FREE] x={x:.3f} y={y:.3f} z={z:.3f}")

        pose = PoseStamped()
        pose.header.frame_id = self.planning_frame  # keep consistent
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.w = 1.0  # unconstrained orientation

        joints = self.compute_ik_sync(pose)
        if joints is None:
            return False

        return self.execute_joints_controller_sync(joints, duration_sec=duration_sec)

    # -
    # IK (compute_ik) 
    # -
    def compute_ik_sync(self, pose_stamped: PoseStamped):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.eef_link
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.avoid_collisions = False


        req.ik_request.timeout.sec = 2
        req.ik_request.timeout.nanosec = 0

        # Provide robot_state seed (prevents "Found empty JointState message")
        req.ik_request.robot_state = RobotState()
        if self._last_joint_state is not None and len(self._last_joint_state.name) > 0:
            req.ik_request.robot_state.joint_state = self._last_joint_state
        else:
            self.get_logger().warn("[IK] No /joint_states received yet. IK may fail.")

        self.get_logger().info(
            f"[IK] Request frame={pose_stamped.header.frame_id} "
            f"x={pose_stamped.pose.position.x:.3f} "
            f"y={pose_stamped.pose.position.y:.3f} "
            f"z={pose_stamped.pose.position.z:.3f}"
        )

        fut = self.ik_srv.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()

        if res is None:
            self.get_logger().error("[IK] No response from /compute_ik")
            return None

        if res.error_code.val != 1:
            self.get_logger().error(f"[IK] Failed. error_code={res.error_code.val}")
            return None

        js = res.solution.joint_state
        if not js.name or not js.position:
            self.get_logger().error("[IK] Solution joint_state empty.")
            return None

        name_to_pos = dict(zip(js.name, js.position))
        missing = [j for j in self.joint_names if j not in name_to_pos]
        if missing:
            self.get_logger().error(f"[IK] Missing joints in solution: {missing}")
            self.get_logger().error(f"[IK] Returned joints: {list(js.name)}")
            return None

        joints = [float(name_to_pos[j]) for j in self.joint_names]
        self.get_logger().info("[IK] Success ")
        return joints


    # Execute joints via controller
    def execute_joints_controller_sync(self, joints, duration_sec=3.0) -> bool:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)

        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in joints]
        p.time_from_start.sec = int(duration_sec)
        p.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)

        goal.trajectory.points = [p]

        self.get_logger().info("[JTC] Sending FollowJointTrajectory ...")
        fut = self.jtc_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()

        if gh is None or not gh.accepted:
            self.get_logger().error("[JTC] Goal rejected.")
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        res_wrap = res_fut.result()
        if res_wrap is None:
            self.get_logger().error("[JTC] No result.")
            return False

        res = res_wrap.result
        # FollowJointTrajectory: error_code 0 = success
        if res.error_code != 0:
            self.get_logger().error(f"[JTC] Failed. error_code={res.error_code} msg='{res.error_string}'")
            return False

        self.get_logger().info("[JTC] Success ")
        return True
    def align_down_with_yaw_search_sync(self, x, y, z, pitch=math.pi, duration_sec=2.0) -> bool:
        pitches = [math.pi, -math.pi/2, math.pi/2] 
        yaws = [0.0, math.pi/2, -math.pi/2, math.pi, math.pi/4, -math.pi/4]
        for pitch in pitches:
            for yaw in yaws:
                ps = PoseStamped()
                ps.header.frame_id = self.planning_frame
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(y)
                ps.pose.position.z = float(z)
                ps.pose.orientation = quat_from_rpy(0.0, pitch, yaw)

                joints = self.compute_ik_sync(ps)
                if joints is not None:
                    self.get_logger().info(f"[ALIGN-DOWN] Success with yaw={yaw:.3f}")
                    return self.execute_joints_controller_sync(joints, duration_sec=duration_sec)

        self.get_logger().error("[ALIGN-DOWN] Failed for all yaw options.")
        return False


    # Pose -> IK -> controller execute

    def move_pose_via_ik_controller_sync(self, x, y, z, preset="carry", duration_sec=3.0) -> bool:
        self.get_logger().info(
            f"[POSE->IK] x={x:.3f} y={y:.3f} z={z:.3f} preset={preset}"
        )

        pose = self.pose_from_xyz_preset(x, y, z, preset=preset)
        joints = self.compute_ik_sync(pose)
        if joints is None:
            return False

        self.get_logger().info(f"[POSE->IK] joints={[round(j, 3) for j in joints]}")
        return self.execute_joints_controller_sync(joints, duration_sec=duration_sec)


    # Gripper
    def send_gripper(self, position: float, max_effort: float = GRIPPER_EFFORT):
        g = GripperCommand.Goal()
        g.command.position = float(position)
        g.command.max_effort = float(max_effort)
        return self.gripper_client.send_goal_async(g)

    def gripper_sync(self, state: str) -> bool:
        pos = GRIPPER_OPEN if state == "open" else GRIPPER_CLOSED
        self.get_logger().info(f"[GRIPPER] {state} pos={pos}")
        fut = self.send_gripper(pos)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error("[GRIPPER] Goal rejected.")
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        self.get_logger().info("[GRIPPER] Done ")
        return True


def main():
    rclpy.init()
    node = MotionBase("motion_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
