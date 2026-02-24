from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

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
            {"trajectory_execution": {
                "allowed_execution_duration_scaling": 20.0,
                "allowed_goal_duration_margin": 30.0,
                "allowed_start_tolerance": 0.20,
            }},
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        output="screen",
        additional_env={"LIBGL_ALWAYS_SOFTWARE": "1"},
        parameters=[
            # pass ONLY what RViz needs explicitly
            {"robot_description": moveit_config.robot_description["robot_description"]},
            {"robot_description_semantic": moveit_config.robot_description_semantic["robot_description_semantic"]},
            {"robot_description_kinematics": moveit_config.robot_description_kinematics["robot_description_kinematics"]},
            {"use_sim_time": use_sim_time},
        ],
        # optional if you have a config file:
        # arguments=["-d", str(moveit_config.package_path / "config" / "moveit.rviz")],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        move_group,
        TimerAction(period=2.0, actions=[rviz]),  # delay helps a lot
    ])
