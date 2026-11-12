#!/usr/bin/env python3
"""
Reads the Carter's current odometry and sends it to AMCL as the initial pose.
Works because SLAM's map frame started exactly on top of the odom frame,
and Isaac Sim has not been reset since, so odom pose ~= map pose.
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data          # Best Effort QoS (works with any publisher)
from nav_msgs.msg import Odometry                       # message type of /chassis/odom
from geometry_msgs.msg import PoseWithCovarianceStamped # message type AMCL listens to on /initialpose


class InitialPoseFromOdom(Node):
    def __init__(self):
        super().__init__('initial_pose_from_odom')
        # Publisher to AMCL's initial-pose topic (the same topic RViz's "2D Pose Estimate" uses)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        # Subscribe to odometry with sensor-data QoS, avoiding the Reliable/Best Effort
        # mismatch you saw with RViz2 and /scan
        self.sub = self.create_subscription(
            Odometry, '/chassis/odom', self.on_odom, qos_profile_sensor_data)
        self.done = False

    def on_odom(self, odom: Odometry):
        if self.done:
            return
        # Wait until AMCL is actually listening, otherwise the message is lost
        if self.pub.get_subscription_count() == 0:
            self.get_logger().info('Waiting for AMCL to subscribe to /initialpose...')
            return

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'            # the pose is expressed in the map frame
        msg.header.stamp = odom.header.stamp   # use sim time from the odom message
        msg.pose.pose = odom.pose.pose         # copy position + orientation from odometry
        msg.pose.pose.position.z = 0.0         # Nav2 is 2D; ignore any height

        # Covariance = "how unsure am I?" It sets how widely AMCL scatters its particles.
        # Index 0 = x variance, 7 = y variance, 35 = yaw variance (6x6 matrix, row-major)
        msg.pose.covariance[0] = 0.25    # +/- 0.5 m in x
        msg.pose.covariance[7] = 0.25    # +/- 0.5 m in y
        msg.pose.covariance[35] = 0.068  # +/- ~15 degrees in heading

        self.pub.publish(msg)

        # Convert the quaternion to a yaw angle just to print something readable
        q = odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        p = odom.pose.pose.position
        self.get_logger().info(
            f'Initial pose sent: x={p.x:.2f} m, y={p.y:.2f} m, yaw={math.degrees(yaw):.1f} deg')
        self.done = True


def main():
    rclpy.init()
    node = InitialPoseFromOdom()
    # Process callbacks until the pose has been sent once
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.5)
    # Spin a little longer so the message actually leaves before shutdown
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
