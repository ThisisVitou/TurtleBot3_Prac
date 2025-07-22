#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
import numpy as np

class ForceFieldNode(Node):
    def __init__(self):
        super().__init__('force_field_node')

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        # Goal
        self.goal_x = 0.0
        self.goal_y = 0.0

        # Obstacle list (x, y, radius)
        self.obstacles = [
            (-2.78, -5.58, 1.57),
            (-4.96, -0.76, 0.5),
            (-1.40, 0.28, 0.5),
            (-4.56, 3.71, 0.5),
            (1.6, 4.0, 0.5),
            (0.47, -2.07, 0.5),
            (2.63, 0.16, 0.5),
            (5.04, 2.13, 0.5),
            (4.36, -2.57, 0.5),
            (2.08, -5.53, 0.5)
        ]

        # State
        self.current_x = -6.0
        self.current_y = -9.0
        self.yaw = 0.0

        self.distance_to_goal_threshold = 0.3

        self.get_logger().info('Force field node initialized.')

    def odom_callback(self, msg):
        """Update current position and orientation from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        rot_q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

    def reached_goal(self):
        distance = np.hypot(self.goal_x - self.current_x, self.goal_y - self.current_y)
        return distance < self.distance_to_goal_threshold

    def compute_force(self):
        # Strong attractive force to goal
        ka = 3.0 
        dx = self.goal_x - self.current_x 
        dy = self.goal_y - self.current_y
        fx = ka * dx
        fy = ka * dy

        # Repulsive force from obstacles
        kr = 5.0  # Repulsive gain
        influence_distance = 3.0  # Obstacles within this distance (from surface) repel
        min_distance = 0.05  # Avoid division by zero and extreme forces

        for ox, oy, radius in self.obstacles:
            # Distance from robot to obstacle center
            dx_o = self.current_x - ox
            dy_o = self.current_y - oy
            dist_to_center = np.hypot(dx_o, dy_o)
            dist = dist_to_center - radius  # Distance from obstacle surface

            if dist < influence_distance and dist > min_distance:
                # Repulsive force magnitude: inverse linear for smoother behavior
                force_magnitude = kr * radius * (1.0 / dist - 1.0 / influence_distance)

                # Normalize direction vector to unit length
                if dist_to_center > 0:  # Avoid division by zero
                    unit_dx = dx_o / dist_to_center
                    unit_dy = dy_o / dist_to_center
                    fx += force_magnitude * unit_dx
                    fy += force_magnitude * unit_dy

        return fx, fy

    def control_loop(self):
        twist = Twist()

        if self.reached_goal():
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info("Reached goal!")
        else:
            fx, fy = self.compute_force()
            desired_angle = math.atan2(fy, fx)
            angle_error = desired_angle - self.yaw
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

            # Distance to goal
            distance_to_goal = np.hypot(self.goal_x - self.current_x, self.goal_y - self.current_y)

            # Stronger forward motion
            twist.linear.x = min(0.5, distance_to_goal * 0.5)  # Scales with distance
            twist.angular.z = max(-1.0, min(1.0, angle_error * 2.0))

            self.get_logger().info(f"Moving. Angle error: {math.degrees(angle_error):.2f}°, Force: ({fx:.2f}, {fy:.2f})")

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ForceFieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ForceField node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
