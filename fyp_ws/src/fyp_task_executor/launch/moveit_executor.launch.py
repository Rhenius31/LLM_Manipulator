from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    moveit_config = (
        MoveItConfigsBuilder(
            "gen3_lite_gen3_lite_2f",
            package_name="kinova_gen3_lite_moveit_config",
        )
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    executor = Node(
        package="fyp_task_executor",
        executable="moveit_executor",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock"
        ),
        executor,
    ])
