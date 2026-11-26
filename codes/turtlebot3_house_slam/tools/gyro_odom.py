#!/usr/bin/python3
"""Odometry as on a real TurtleBot3: distance from the wheels, heading from the gyro."""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


class GyroOdom(Node):
    def __init__(self):
        super().__init__('gyro_odom')
        self.x = self.y = self.yaw = self.rate = 0.0
        self.last_imu = self.last_wheel = None
        self.pub = self.create_publisher(Odometry, 'odom', 20)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Imu, 'imu', self.on_imu, QoSProfile(depth=1000))
        self.create_subscription(Odometry, 'wheel_odom', self.on_wheel, 100)

    def on_imu(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.rate = m.angular_velocity.z
        if self.last_imu is not None and t > self.last_imu:
            self.yaw += self.rate * (t - self.last_imu)
        self.last_imu = t

    def on_wheel(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        a = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if self.last_wheel is not None:
            x0, y0, a0 = self.last_wheel
            mid = a0 + math.atan2(math.sin(a - a0), math.cos(a - a0)) / 2
            ds = (p.x - x0) * math.cos(mid) + (p.y - y0) * math.sin(mid)
            self.x += ds * math.cos(self.yaw)
            self.y += ds * math.sin(self.yaw)
        self.last_wheel = (p.x, p.y, a)

        o = Odometry()
        o.header.stamp = m.header.stamp
        o.header.frame_id, o.child_frame_id = 'odom', 'base_footprint'
        o.pose.pose.position.x, o.pose.pose.position.y = self.x, self.y
        o.pose.pose.orientation.z = math.sin(self.yaw / 2)
        o.pose.pose.orientation.w = math.cos(self.yaw / 2)
        o.twist.twist.linear.x = m.twist.twist.linear.x
        o.twist.twist.angular.z = self.rate
        self.pub.publish(o)

        t = TransformStamped()
        t.header, t.child_frame_id = o.header, o.child_frame_id
        t.transform.translation.x, t.transform.translation.y = self.x, self.y
        t.transform.rotation = o.pose.pose.orientation
        self.tf.sendTransform(t)


if __name__ == '__main__':
    rclpy.init()
    try:
        rclpy.spin(GyroOdom())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
