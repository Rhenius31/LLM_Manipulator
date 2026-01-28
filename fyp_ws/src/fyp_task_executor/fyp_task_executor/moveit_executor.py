#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint

from control_msgs.action import GripperCommand

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0
GRIPPER_EFFORT = 100.0


class MoveItExecutor(Node):
    def __init__(self):
        super().__init__("moveit_executor")

        self.group_name = "arm"

        # MoveIt actions
        self.move_action = "/move_action"
        self.exec_action = "/execute_trajectory"

        # Gripper action (from your controller name)
        self.gripper_action = "/gen3_lite_2f_gripper_controller/gripper_cmd"

        self.move_client = ActionClient(self, MoveGroup, self.move_action)
        self.exec_client = ActionClient(self, ExecuteTrajectory, self.exec_action)
        self.gripper_client = ActionClient(self, GripperCommand, self.gripper_action)

        self.get_logger().info(f"Waiting for {self.move_action} ...")
        self.move_client.wait_for_server()
        self.get_logger().info("Connected to MoveGroup ")

        self.get_logger().info(f"Waiting for {self.exec_action} ...")
        self.exec_client.wait_for_server()
        self.get_logger().info("Connected to ExecuteTrajectory ")

        self.get_logger().info(f"Waiting for {self.gripper_action} ...")
        self.gripper_client.wait_for_server()
        self.get_logger().info("Connected to GripperCommand ")

        # Arm joints
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

        # Demo sequence
        self.sequence = [
            ("arm", "home",     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("grip", "open",    GRIPPER_OPEN),   # try 0.08 open
            ("arm", "pregrasp", [0.3, -1.1, 1.1, 0.2, 1.2, 0.0]),
            ("grip", "close",   GRIPPER_CLOSED),   # try 0.00 close
            ("arm", "place",    [-0.3, -1.0, 1.2, -0.2, 1.1, 0.0]),
            ("grip", "open",    GRIPPER_OPEN),
            ("arm", "home_end", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        self.idx = 0
        self.run_next()

    #  Build a joint-goal request 
    def build_joint_goal(self, joints, tolerance_rad=0.01):
        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.num_planning_attempts = 3
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

        c = Constraints()
        c.name = "joint_goal"

        for name, pos in zip(self.joint_names, joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = float(tolerance_rad)
            jc.tolerance_below = float(tolerance_rad)
            jc.weight = 1.0
            c.joint_constraints.append(jc)

        req.goal_constraints = [c]
        return req

    #  Plan with MoveGroup (plan_only=True) 
    def send_plan(self, request: MotionPlanRequest):
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return self.move_client.send_goal_async(goal)

    #  Execute trajectory 
    def send_execute(self, robot_trajectory):
        g = ExecuteTrajectory.Goal()
        g.trajectory = robot_trajectory
        return self.exec_client.send_goal_async(g)

    #  Gripper command 
    def send_gripper(self, position: float, max_effort: float = GRIPPER_EFFORT):
        g = GripperCommand.Goal()
        g.command.position = float(position)
        g.command.max_effort = float(max_effort)
        return self.gripper_client.send_goal_async(g)

    # Sequence runner
    def run_next(self):
        if self.idx >= len(self.sequence):
            self.get_logger().info("Sequence done ")
            return

        kind, name, value = self.sequence[self.idx]

        if kind == "arm":
            joints = value
            self.get_logger().info(f"[{self.idx+1}/{len(self.sequence)}] ARM: {name}  {joints}")
            req = self.build_joint_goal(joints)
            fut = self.send_plan(req)
            fut.add_done_callback(self._on_plan_goal)

        else:
            pos = float(value)
            self.get_logger().info(f"[{self.idx+1}/{len(self.sequence)}] GRIPPER: {name}  pos={pos}")
            fut = self.send_gripper(pos)
            fut.add_done_callback(self._on_gripper_goal)

    # Callbacks: planning
    def _on_plan_goal(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("MoveGroup goal rejected ")
            return
        gh.get_result_async().add_done_callback(self._on_plan_result)

    def _on_plan_result(self, future):
        res = future.result().result
        if res.error_code.val != 1:
            self.get_logger().error(f"Planning failed. error_code={res.error_code.val}")
            return

        traj = res.planned_trajectory
        npts = len(traj.joint_trajectory.points)
        self.get_logger().info(f"Plan OK points={npts}")

        if npts == 0:
            self.get_logger().error("Empty trajectory, cannot execute.")
            return

        fut = self.send_execute(traj)
        fut.add_done_callback(self._on_exec_goal)

    # Callbacks: execution
    def _on_exec_goal(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("ExecuteTrajectory rejected ")
            return
        gh.get_result_async().add_done_callback(self._on_exec_result)

    def _on_exec_result(self, future):
        res = future.result().result
        if res.error_code.val == 1:
            self.get_logger().info("Arm execution SUCCESS ")
            self.idx += 1
            self.run_next()
        else:
            self.get_logger().error(f"Arm execution failed. error_code={res.error_code.val}")

    # Callbacks: gripper
    def _on_gripper_goal(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("Gripper goal rejected ")
            return
        gh.get_result_async().add_done_callback(self._on_gripper_result)

    def _on_gripper_result(self, future):
        # GripperCommand result doesn't have MoveItErrorCodes; just proceed if action completed
        self.get_logger().info("Gripper action done ")
        self.idx += 1
        self.run_next()


def main():
    rclpy.init()
    node = MoveItExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
