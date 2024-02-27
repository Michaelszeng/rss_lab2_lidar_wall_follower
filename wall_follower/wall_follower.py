#!/usr/bin/env python3
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros
import tf_transformations

from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", 1)  # +1 = left wall; -1 = right wall
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

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)  # To get velocity/speed data to determine lookahead distance

        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)  # queue size 10
        self.visualization_publisher_ = self.create_publisher(MarkerArray, 'visualization', 10)
        self.wall_visualization_publisher_ = self.create_publisher(Marker, 'wall_visualization', 10)
        self.lookeahead_visualization_publisher_ = self.create_publisher(Marker, 'lookahead_visualization', 10)
        timer_period = self.UPDATE_POSE_RATE
        self.timer = self.create_timer(timer_period, self.steering_cb)

        self.m = None  # in base_link frame
        self.b = None  # in base_link frame
        self.min_L = 2
        self.max_L = 4
        self.L_ratio = 1  # ratio from v to L
        self.L_offset = self.WHEELBASE

        # Non-constant attributes
        self.speed = 0
        self.heading = 0


    def plot_wall_markers(self, frame_id, coords, m, b):
        """
        Helper function to plot points in RViz.

        coords should be Nx2
        """
        marker_array = MarkerArray()

        for i in range(np.shape(coords)[0]):
            x = coords[i,0]
            y = coords[i,1]
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.ns = "wall_markers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            # Size of the point
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker_array.markers.append(marker)

        # Draw best fit line
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.05  # Line width
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        # Define start and end points of the line
        start_x = coords[0,0]
        start_y = m*start_x + b
        start_point = Point(x=start_x, y=start_y, z=0.0)
        end_x = coords[-1,0]
        end_y = m*end_x + b
        end_point = Point(x=end_x, y=end_y, z=0.0)

        marker.points.append(start_point)
        marker.points.append(end_point)

        self.wall_visualization_publisher_.publish(marker)
        self.visualization_publisher_.publish(marker_array)

    
    def plot_lookahead_marker(self, frame_id, lookahead_pt):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_pt"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = lookahead_pt[0]
        marker.pose.position.y = lookahead_pt[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        # Size of the point
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25

        self.lookeahead_visualization_publisher_.publish(marker)

    def polar_to_cartesian(self, ranges, angle_min, angle_max, angle_increment):
        # Calculate the angles for each range
        angles = np.arange(angle_min, angle_max, angle_increment)
        
        # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
        x_coordinates = ranges * np.cos(angles)
        y_coordinates = ranges * np.sin(angles)
        
        # Combine x and y coordinates into a single array
        coordinates = np.vstack((x_coordinates, y_coordinates)).T
        
        return coordinates
    

    def best_fit_line(self, coords):
        # Extract x and y coordinates
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Calculate the necessary sums
        N = len(coords)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        # Calculate the slope (m) and y-intercept (b)
        m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)
        b = (sum_y - m * sum_x) / N
        
        return m, b  # in base_link frame
    

    def find_lookahead_points(self, r, m, b):
        # Coefficients for the quadratic equation Ax^2 + Bx + C = 0
        A = 1 + m**2
        B = 2*m*b
        C = b**2 - r**2
        
        # Solve the quadratic equation
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            return []  # No real intersections
        else:
            # Calculate both solutions for x
            x1 = (-B + np.sqrt(discriminant)) / (2*A)
            x2 = (-B - np.sqrt(discriminant)) / (2*A)
            
            # Calculate corresponding y values
            y1 = m*x1 + b
            y2 = m*x2 + b
            
            # Return intersection points
            if discriminant == 0:
                return [(x1, y1)]  # One intersection (tangent)
            else:
                return [(x1, y1), (x2, y2)]  # Two intersections


    def odom_callback(self, msg):
        linear_velocity = msg.twist.twist.linear
        angular_velocity = msg.twist.twist.angular

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf_transformations.euler_from_quaternion(orientation_list)
        self.heading = yaw

        self.get_logger().info(
            'Linear Velocity: x={:.2f} m/s, y={:.2f} m/s, z={:.2f} m/s'.format(
                linear_velocity.x, linear_velocity.y, linear_velocity.z))

        # self.get_logger().info(
        #     'Angular Velocity: x={:.2f} rad/s, y={:.2f} rad/s, z={:.2f} rad/s'.format(
        #         angular_velocity.x, angular_velocity.y, angular_velocity.z))
        
        self.speed = math.sqrt(linear_velocity.x**2 + linear_velocity.y**2 + linear_velocity.z**2)


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
        coordinates = self.polar_to_cartesian(ranges, angle_min, angle_max, angle_increment)  # Nx2
        # self.get_logger().info(f"coordinates: {coordinates}")

        # Fit line to x,y coordinates
        m, b = self.best_fit_line(coordinates)

        self.plot_wall_markers("base_link", coordinates, m, b)

        self.m = m
        self.b = b


    def steering_cb(self):
        if self.m != None and self.b != None:
            L = max(self.min_L, min(self.max_L, self.L_ratio*self.speed))  # bound between min_L and max_L
            lookahead_pts = self.find_lookahead_points(L, self.m, self.b)  # in base_link frame
            if len(lookahead_pts) > 0:
                lookahead_pt = max(lookahead_pts, key=lambda point: point[0])
                self.plot_lookahead_marker('base_link', lookahead_pt)
                self.get_logger().info(f"lookahead_pt: {lookahead_pt}")
            else:
                self.get_logger().info(f"No lookahead point found.")

        # steering_angle = np.arctan2((self.WHEELBASE*np.sin(eta)) / ((0.5*L) + (self.L_offset*np.cos(eta))))

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
    
