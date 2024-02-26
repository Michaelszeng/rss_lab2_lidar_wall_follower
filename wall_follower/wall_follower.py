#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA

from wall_follower.visualization_tools import *


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", -1)  # +1 = left wall; -1 = right wall
        self.declare_parameter("velocity", 4.0)
        self.declare_parameter("desired_distance", 0.275)
        self.declare_parameter("wheelbase", 0.325)
        self.declare_parameter("update_pose_rate", 0.02)

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        self.WHEELBASE = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.UPDATE_POSE_RATE = self.get_parameter('update_pose_rate').get_parameter_value().double_value
		
        # Initialize publishers and subscribers
        self.subscription = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.laser_scan_cb,
            10)  # queue size 10
        self.subscription  # prevent unused variable warning

        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)  # queue size 10
        self.visualization_publisher_ = self.create_publisher(Marker, 'visualization', 10)
        timer_period = self.UPDATE_POSE_RATE
        self.timer = self.create_timer(timer_period, self.steering_cb)


    def plot_marker(self, frame_id, x, y):
        """
        Helper function to plot points in RViz
        """
        marker = Marker()
        marker.header = Header(frame_id=frame_id)  # Set the frame ID here
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        # Size of the point
        marker.scale.x = 0.1
        marker.scale.y = 0.1

        # Color and alpha
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

        p = Point(x=x, y=y)
        marker.points.append(p)

        self.visualization_publisher_.publish(marker)


    def polar_to_cartesian(self, ranges, angle_min, angle_max, angle_increment):
        # Calculate the angles for each range
        angles = np.arange(angle_min, angle_max, angle_increment)
        
        # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
        x_coordinates = ranges * np.cos(angles)
        y_coordinates = ranges * np.sin(angles)
        
        # Combine x and y coordinates into a single array
        coordinates = np.vstack((x_coordinates, y_coordinates)).T
        
        return coordinates


    def laser_scan_cb(self, msg):
        lidar_ranges = msg.ranges
        lidar_angle_min = msg.angle_min  # -2.3550000190734863
        lidar_angle_max = msg.angle_max  # 2.3550000190734863
        angle_increment = msg.angle_increment

        if self.SIDE == 1:  # left
            angle_min = 0.261799
            angle_max = 1.570796  # pi/2
        else:  # right
            angle_min = -1.570796
            angle_max = -0.261799 # pi/2

        ranges = lidar_ranges[int((angle_min - lidar_angle_min)/angle_increment) : int(((angle_min - lidar_angle_min)/angle_increment) + ((angle_max - angle_min)/angle_increment))+1]

        # Convert ranges to x,y coordinates
        coordinates = self.polar_to_cartesian(ranges, angle_min, angle_max, angle_increment)
        for coord in coordinates:
            self.plot_marker("base_link", coord[0], coord[1])

        # Fit line to x,y coordinates


    def steering_cb(self):
        drive_msg = AckermannDriveStamped()
        # drive_msg.drive.steering_angle = 
        # drive_msg.drive.steering_angle_velocity = 
        # drive_msg.drive.speed = 
        # drive_msg.drive.acceleration = 
        # drive_msg.drive.jerk = 
        self.publisher_.publish(drive_msg)
        # self.get_logger().info(f"Publishing: {1}")


def main():
    
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
