from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    with open('/home/rhenius/workspace/my_robot_task_api/urdf/gen3_final.urdf', 'r') as inf:
        robot_desc = inf.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[
                {'use_sim_time': True},
                {'robot_description': robot_desc}
            ]
        )
    ])
