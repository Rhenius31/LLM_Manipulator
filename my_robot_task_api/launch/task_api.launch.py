from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package="my_robot_task_api",
            executable="fake_detector_node",
            name="fake_detector_node",
            output="screen",
        ),
        Node(
            package="my_robot_task_api",
            executable="pick_place_node",
            name="pick_place_node",
            output="screen",
        ),
        Node(
            package="my_robot_task_api",
            executable="scene_node",
            name="scene_node",
            output="screen",
        ),
    ])
