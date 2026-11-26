"""Show the saved house map in RViz."""
import os

from launch import LaunchDescription
from launch.actions import UnsetEnvironmentVariable
from launch_ros.actions import Node

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def generate_launch_description():
    return LaunchDescription([
        UnsetEnvironmentVariable('GTK_PATH'),  # VS Code's snap value crashes RViz
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen',
             parameters=[{'yaml_filename': os.path.join(ROOT, 'maps', 'turtlebot3_house.yaml')}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='map_lifecycle',
             parameters=[{'autostart': True, 'node_names': ['map_server']}]),
        # RViz needs the fixed frame to exist in TF.
        Node(package='tf2_ros', executable='static_transform_publisher', name='map_frame',
             arguments=['--frame-id', 'map', '--child-frame-id', 'map_view']),
        Node(package='rviz2', executable='rviz2', prefix=os.path.join(ROOT, 'tools', 'nogpu.sh'), output='screen',
             arguments=['-d', os.path.join(ROOT, 'config', 'view_map.rviz')]),
    ])
