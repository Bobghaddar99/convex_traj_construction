"""Gazebo (software rendering, RTF 0.5), robot, bridge and odometry for the Nav2 trials. No SLAM."""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, UnsetEnvironmentVariable
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)


def generate_launch_description():
    sim_time = {'use_sim_time': True}
    tf = lambda child, x, z: Node(
        package='tf2_ros', executable='static_transform_publisher', name=f'tf_{child}', parameters=[sim_time],
        arguments=['--x', str(x), '--z', str(z), '--frame-id', 'base_footprint', '--child-frame-id', child])
    return LaunchDescription([
        UnsetEnvironmentVariable('GTK_PATH'),
        SetEnvironmentVariable('GZ_PARTITION', 'house_slam'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(ROOT, 'models')),
        ExecuteProcess(cmd=[os.path.join(ROOT, 'tools', 'nogpu.sh'), 'gz', 'sim', '-r', '-s', '--headless-rendering',
                            '-v1', os.path.join(HERE, 'house_half_speed.world')], output='screen'),
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-file', os.path.join(ROOT, 'models', 'turtlebot3_burger', 'model.sdf'),
                        '-name', 'burger', '-x', '-2.008', '-y', '1.0', '-z', '0.01']),
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             parameters=[sim_time, {'config_file': os.path.join(HERE, 'bridge.yaml')}]),
        ExecuteProcess(cmd=['/usr/bin/python3', os.path.join(ROOT, 'tools', 'gyro_odom.py'),
                            '--ros-args', '-p', 'use_sim_time:=true'], output='screen'),
        tf('base_scan', -0.032, 0.171),
        tf('base_link', 0.0, 0.0),
    ])
