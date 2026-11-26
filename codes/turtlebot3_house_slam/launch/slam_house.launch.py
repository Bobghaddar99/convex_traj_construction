"""TurtleBot3 Burger in the house world (Gazebo Harmonic) with SLAM Toolbox."""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, UnsetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def path(*parts):
    return os.path.join(ROOT, *parts)


def generate_launch_description():
    sim_time = {'use_sim_time': True}
    gui = LaunchConfiguration('gui')
    world = LaunchConfiguration('world')
    resources = os.pathsep.join(
        filter(None, [path('models'), os.environ.get('GZ_SIM_RESOURCE_PATH')]))

    nogpu = path('tools', 'nogpu.sh')  # Gazebo and RViz never touch the NVIDIA GPU

    def static_tf(child, x, z):
        return Node(
            package='tf2_ros', executable='static_transform_publisher', name=f'tf_{child}',
            parameters=[sim_time],
            arguments=['--x', str(x), '--z', str(z),
                       '--frame-id', 'base_footprint', '--child-frame-id', child])

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('bridge', default_value=path('config', 'bridge.yaml')),
        DeclareLaunchArgument('world', default_value=path('worlds', 'turtlebot3_house_closed.world')),
        DeclareLaunchArgument(
            'robot', default_value=path('models', 'turtlebot3_burger', 'model.sdf')),
        # VS Code's snap leaks these: GTK_PATH crashes RViz, and SNAP_COMMON makes
        # slam_toolbox save pose graphs under /var/snap.
        UnsetEnvironmentVariable('GTK_PATH'),
        UnsetEnvironmentVariable('SNAP_COMMON'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resources),
        ExecuteProcess(cmd=[nogpu, 'gz', 'sim', '-r', '-v1', world],
                       condition=IfCondition(gui), output='screen'),
        ExecuteProcess(cmd=[nogpu, 'gz', 'sim', '-r', '-s', '--headless-rendering', '-v1', world],
                       condition=UnlessCondition(gui), output='screen'),
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-file', LaunchConfiguration('robot'), '-name', 'burger',
                        '-x', '-2.0', '-y', '1.0', '-z', '0.01']),
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             parameters=[sim_time, {'config_file': LaunchConfiguration('bridge')}]),
        ExecuteProcess(cmd=['/usr/bin/python3', path('tools', 'gyro_odom.py'),
                            '--ros-args', '-p', 'use_sim_time:=true'], output='screen'),
        ExecuteProcess(cmd=['/usr/bin/python3', path('tools', 'scan_filter.py')], output='screen'),
        # The lidar link is at the model origin; the sensor inside it carries the offset.
        static_tf('base_scan', -0.032, 0.171),
        static_tf('base_link', 0.0, 0.0),
        Node(package='slam_toolbox', executable='sync_slam_toolbox_node', name='slam_toolbox',
             parameters=[path('config', 'slam.yaml'), sim_time], output='screen'),
        Node(package='rviz2', executable='rviz2', prefix=nogpu, condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', path('config', 'slam.rviz')], parameters=[sim_time]),
    ])
