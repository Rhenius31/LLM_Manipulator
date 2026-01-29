from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory("my_robot_task_api")

    world = os.path.join(pkg_share, "worlds", "gen3_world.sdf")
    urdf  = os.path.join(pkg_share, "urdf", "gen3_sim.urdf")

    return LaunchDescription([
        # Gazebo
        ExecuteProcess(
            cmd=["gz", "sim", "-r", world],
            output="screen"
        ),

        # robot_state_publisher
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "publish_robot_description": True,
                "robot_description": open(urdf, "r").read(),
            }],
        ),

        # Spawn robot (delay a bit so robot_description is definitely available)
        TimerAction(
            period=2.0,
            actions=[
                ExecuteProcess(
                    cmd=["ros2", "run", "ros_gz_sim", "create",
                         "-name", "gen3_lite",
                         "-topic", "/robot_description"],
                    output="screen"
                )
            ]
        ),

        # Controllers (delay a bit so controller_manager exists)
        TimerAction(
            period=4.0,
            actions=[
                ExecuteProcess(
                    cmd=["ros2", "run", "controller_manager", "spawner",
                         "joint_state_broadcaster", "--controller-manager", "/controller_manager"],
                    output="screen"
                ),
                ExecuteProcess(
                    cmd=["ros2", "run", "controller_manager", "spawner",
                         "joint_trajectory_controller", "--controller-manager", "/controller_manager"],
                    output="screen"
                ),
                ExecuteProcess(
                    cmd=["ros2", "run", "controller_manager", "spawner",
                         "gen3_lite_2f_gripper_controller", "--controller-manager", "/controller_manager"],
                    output="screen"
                ),
            ]
        ),
    ])
