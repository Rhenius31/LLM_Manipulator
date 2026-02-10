from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Build MoveIt config from Kinova's moveit_config package (no edits to Kinova pkg)
    moveit_config = (
        MoveItConfigsBuilder(
            "gen3_lite_gen3_lite_2f",
            package_name="kinova_gen3_lite_moveit_config",
        )
        .to_moveit_configs()
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
            # ✅ Increase execution wait time so MoveIt doesn't CONTROL_FAILED early
            {"trajectory_execution": {
                "allowed_execution_duration_scaling": 20.0,
                "allowed_goal_duration_margin": 30.0,
                "allowed_start_tolerance": 0.01,
            }},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        move_group,
    ])
