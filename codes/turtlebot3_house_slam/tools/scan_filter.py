#!/usr/bin/python3
"""Let "nothing hit" lidar rays clear free space in SLAM Toolbox.

Karto ignores readings >= range_max, so an infinite ray clears nothing and floor under
tables stays unknown. Such rays are republished as a finite reading beyond the mapping
range (max_laser_range): Karto then marks them free up to that range, with no obstacle.
"""
import math

import rclpy
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan

NO_HIT = 5.0      # must be above max_laser_range and below RANGE_MAX
RANGE_MAX = 12.0

if __name__ == '__main__':
    rclpy.init()
    node = rclpy.create_node('scan_filter')
    pub = node.create_publisher(LaserScan, 'scan_slam', 50)

    def on_scan(m):
        m.ranges = [NO_HIT if math.isinf(r) and r > 0 else r for r in m.ranges]
        m.range_max = RANGE_MAX
        pub.publish(m)

    node.create_subscription(LaserScan, 'scan', on_scan, QoSProfile(depth=50))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
