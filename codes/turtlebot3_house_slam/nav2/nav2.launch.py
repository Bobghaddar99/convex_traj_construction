"""Planner, DWB controller and AMCL on the saved map (no recoveries, no velocity smoother: pure DWA)."""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))


def generate_launch_description():
    params = [LaunchConfiguration('params'), {'use_sim_time': True}]
    node = lambda pkg, exe, name, **kw: Node(package=pkg, executable=exe, name=name,
                                             parameters=params, output='screen', **kw)
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=os.path.join(HERE, 'params_paper.yaml')),
        node('nav2_map_server', 'map_server', 'map_server'),
        node('nav2_amcl', 'amcl', 'amcl'),
        node('nav2_planner', 'planner_server', 'planner_server'),
        node('nav2_controller', 'controller_server', 'controller_server',
             remappings=[('cmd_vel', '/house_sim/cmd_vel')]),
        # started late: otherwise its first service reply can be lost during discovery
        TimerAction(period=8.0, actions=[Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_dwa',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': ['map_server', 'amcl', 'planner_server', 'controller_server']}])]),
    ])
