import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import xacro


def generate_launch_description():
    pkg = get_package_share_directory("kinova_gen3_lite_moveit_config")

    controllers_yaml = os.path.join(pkg, "config", "ros2_controllers.yaml")
    urdf_xacro = os.path.join(pkg, "config", "gen3_lite_gen3_lite_2f.urdf.xacro")

    doc = xacro.process_file(
        urdf_xacro,
        mappings={
            "use_fake_hardware": "false",
            "sim_gazebo": "true",
            "moveit_active": "false",
            "robot_ip": "0.0.0.0",
            "ros2_control_name": "controller_manager",
        },
    )
    robot_description = {"robot_description": doc.toxml()}

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True, "publish_robot_description": True}],
    )


    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        name="controller_manager",
        output="screen",
        parameters=[robot_description, controllers_yaml, {"use_sim_time": True}],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    joint_traj_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gen3_lite_2f_gripper_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    robot_desc_pub = Node(
    package="my_robot_task_api",
    executable="robot_description_publisher",
    output="screen",
    parameters=[robot_description, {"use_sim_time": True}],
)


    return LaunchDescription(
        [
            rsp_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            joint_traj_spawner,
            gripper_spawner,
            robot_desc_pub,
        ]
    )
