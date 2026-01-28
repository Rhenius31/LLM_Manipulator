from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_task_api'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # Required index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Package.xml
        ('share/' + package_name, ['package.xml']),
        #  INSTALL LAUNCH FILES
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rhenius',
    maintainer_email='rhenius@todo.todo',
    description='Task-level API for MoveIt + LLM control',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_node = my_robot_task_api.motion_node:main',
            'pick_place_node = my_robot_task_api.pick_place_node:main',
            'fake_detector_node = my_robot_task_api.fake_detector_node:main',
            'scene_node = my_robot_task_api.scene_node:main',
            "robot_description_publisher = my_robot_task_api.robot_description_publisher:main",

        ],
    },
)
