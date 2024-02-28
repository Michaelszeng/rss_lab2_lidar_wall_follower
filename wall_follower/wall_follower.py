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

        self.get_logger().info(f"self.SIDE: {self.SIDE}")
        self.get_logger().info(f"self.VELOCITY: {self.VELOCITY}")
        self.get_logger().info(f"self.DESIRED_DISTANCE: {self.DESIRED_DISTANCE}")

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

        self.publisher_ = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)  # queue size 10
        self.visualization_publisher_ = self.create_publisher(MarkerArray, 'visualization', 10)
        self.wall_visualization_publisher_ = self.create_publisher(Marker, 'wall_visualization', 10)
        timer_period = self.UPDATE_POSE_RATE
        self.timer = self.create_timer(timer_period, self.steering_cb)

        
        # Constants
        self.L_offset = self.WHEELBASE

        # Hyperparameters
        self.forward_angle = -0.2
        self.back_angle = 2.2
        self.ransac_thresh = 0.175
        self.ransac_success_thresh = 0.2
        self.range_thresh = 4  # ignore any range measurements further than this
        self.linear_regression_theshold_dist = 2.0 * self.DESIRED_DISTANCE
        self.kp = 0.8/(self.VELOCITY**2)
        self.kd = 0.2666*(self.VELOCITY**2)
        # self.kd = 0.0

        # Non-constant attributes
        self.speed = 0
        self.m = None  # in base_link frame
        self.b = None  # in base_link frame


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
        if m is not None and b is not None:
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

    def polar_to_cartesian(self, ranges, angle_min, angle_max, angle_increment):
        # Calculate the angles for each range
        angles = np.arange(angle_min, angle_max+angle_increment, angle_increment)  # +angle_increment to include endpoint in arange

        valid_indices = np.where(ranges != -1)[0]  # indices in ranges not equal to -1

        ranges = ranges[valid_indices]
        angles = angles[valid_indices]
        
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


    # def best_fit_line(self, coords):
    #     # Extract x and y coordinates
    #     x = coords[:, 0]
    #     y = coords[:, 1]
        
    #     # Calculate the quartiles and IQR for y
    #     Q1 = np.percentile(y, 25)
    #     Q3 = np.percentile(y, 75)
    #     IQR = Q3 - Q1
        
    #     # Determine the outlier boundaries
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
        
    #     # Filter out outlier points
    #     non_outlier_indices = (y >= lower_bound) & (y <= upper_bound)
    #     x_filtered = x[non_outlier_indices]
    #     y_filtered = y[non_outlier_indices]
        
    #     # Calculate the necessary sums with filtered data
    #     N = len(x_filtered)
    #     sum_x = np.sum(x_filtered)
    #     sum_y = np.sum(y_filtered)
    #     sum_xy = np.sum(x_filtered * y_filtered)
    #     sum_x2 = np.sum(x_filtered ** 2)
        
    #     # Calculate m and b with filtered data
    #     m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)
    #     b = (sum_y - m * sum_x) / N
        
    #     return m, b  # in base_link frame

    def best_fit_line_ransac(self, coords, iters=100):
        best_m = None
        best_b = None
        best_inliers = None
        best_num_inliers = 0
        
        for _ in range(iters):
            # Randomly select 2 points
            try:
                indices = np.random.choice(len(coords), size=2, replace=False)
            except:
                return self.m, self.b
            sample = coords[indices]
            
            # Fit a line to these 2 points
            x_sample = sample[:, 0]
            y_sample = sample[:, 1]
            m = (y_sample[1] - y_sample[0]) / (x_sample[1] - x_sample[0])
            b = y_sample[0] - m * x_sample[0]
            
            # Calculate distances from all points to the line
            distances = np.abs(coords[:, 1] - (m * coords[:, 0] + b))
            
            # Count inliers
            inliers = coords[distances < self.ransac_thresh]
            num_inliers = len(inliers)
            
            # Update best fit if necessary
            if num_inliers > best_num_inliers:
                best_m = m
                best_b = b
                best_inliers = inliers
                best_num_inliers = num_inliers
        
        # if num_inliers < self.ransac_success_thresh * len(coords):
        #     self.get_logger().info("Too many outliers for RANSAC; deferring to linear regression.")
        #     # Very large proportion of outliers,so just defer to normal linear regressoin
        #     return self.best_fit_line(coords)
                
        return best_m, best_b
            

    def pd(self, e, angle):
        """
        PD controller for steering command. Angle is used as derivative, since
        it is proportional to the rate of change of e.
        """
        if self.SIDE == -1:
            return -(self.kp * e) + (self.kd * angle)
        else:
            return (self.kp * e) + (self.kd * angle)


    def odom_callback(self, msg):
        linear_velocity = msg.twist.twist.linear
        angular_velocity = msg.twist.twist.angular

        # self.get_logger().info(
        #     'Linear Velocity: x={:.2f} m/s, y={:.2f} m/s, z={:.2f} m/s'.format(
        #         linear_velocity.x, linear_velocity.y, linear_velocity.z))

        # self.get_logger().info(
        #     'Angular Velocity: x={:.2f} rad/s, y={:.2f} rad/s, z={:.2f} rad/s'.format(
        #         angular_velocity.x, angular_velocity.y, angular_velocity.z))
        
        self.speed = math.sqrt(linear_velocity.x**2 + linear_velocity.y**2 + linear_velocity.z**2)


    def laser_scan_cb(self, msg):
        # For grading:
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        lidar_ranges = msg.ranges
        lidar_angle_min = msg.angle_min  # -2.3550000190734863
        lidar_angle_max = msg.angle_max  # 2.3550000190734863
        angle_increment = msg.angle_increment
        
        if self.SIDE == 1:  # left
            angle_min = self.forward_angle
            angle_max = self.back_angle
        else:  # right
            angle_min = -self.back_angle  # pi/2
            angle_max = -self.forward_angle 

        ranges = np.array(lidar_ranges[int((angle_min - lidar_angle_min)/angle_increment) : int(((angle_min - lidar_angle_min)/angle_increment) + ((angle_max - angle_min)/angle_increment))+1])
        ranges[ranges > self.range_thresh] = -1  # remove range meaurements too far away

        # Convert ranges to x,y coordinates
        coordinates = self.polar_to_cartesian(ranges, angle_min, angle_max, angle_increment)  # Nx2
        # self.get_logger().info(f"coordinates: {coordinates}")

        # Fit line to x,y coordinates
        # If there are many nearby points in front of the car, do simple linear regression
        ranges_front_of_car = list(lidar_ranges[int((-0.6 - lidar_angle_min)/angle_increment) : int(((-0.6 - lidar_angle_min)/angle_increment) + (0.8/angle_increment))])
        # self.get_logger().info(f"int((-0.6 - lidar_angle_min)/angle_increment): {int((-0.6 - lidar_angle_min)/angle_increment)}")
        # self.get_logger().info(f"int(((angle_min - lidar_angle_min)/angle_increment) + (0.8/angle_increment)): {int(((-0.6 - lidar_angle_min)/angle_increment) + (0.8/angle_increment))}")
        # self.get_logger().info(f"ranges_front_of_car: {ranges_front_of_car}")
        # self.get_logger().info(f"len(ranges_front_of_car): {(ranges_front_of_car)}")
        if sum(ranges_front_of_car)/(len(ranges_front_of_car) + 1e-6) < self.linear_regression_theshold_dist:
            self.get_logger().info("Using linear regression instead of RANSAC.")
            # Duplicate the points in front of the car to give them more weight
            front_coordinates = coordinates[int((-0.6 - lidar_angle_min)/angle_increment) : int(((-0.6 - lidar_angle_min)/angle_increment) + (0.8/angle_increment)),:]  # Mx2
            coordinates = np.vstack([coordinates, front_coordinates, front_coordinates, front_coordinates, front_coordinates])
            m, b = self.best_fit_line(coordinates)
        else:  # Otherwise, use RANSAC
            m, b = self.best_fit_line_ransac(coordinates)
        
        if len(coordinates > 0):
            self.plot_wall_markers("base_link", coordinates, m, b)

        self.m = m
        self.b = b


    def steering_cb(self):
        if self.m != None and self.b != None:

            cur_wall_dist = abs(self.b) / math.sqrt(1 + self.m**2)
            angle_to_wall = math.atan(self.m)
            steering_angle = self.pd(cur_wall_dist - self.DESIRED_DISTANCE, angle_to_wall)


            # L = max(self.min_L, min(self.max_L, self.L_ratio*self.speed))  # bound between min_L and max_L
            # self.get_logger().info(f"L: {L}")
            # lookahead_pts = self.find_lookahead_points(L, self.m, self.b)  # in base_link frame
            # lookahead_pt = max(lookahead_pts, key=lambda point: point[0])  # in base_link frame
            # self.plot_lookahead_marker('base_link', lookahead_pt)
            # self.get_logger().info(f"lookahead_pt: {lookahead_pt}")

            # eta = np.arctan2(lookahead_pt[1], lookahead_pt[0])  # angle relative to base_link x-axis

            # steering_angle = math.atan((self.WHEELBASE*np.sin(eta)) / ((0.5*L) + (self.L_offset*np.cos(eta))))

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering_angle
            drive_msg.drive.speed = self.VELOCITY
            self.publisher_.publish(drive_msg)
            # self.get_logger().info(f"Publishing steering_angle: {drive_msg.drive.steering_angle} and velocity: {drive_msg.drive.speed}")


def main():
    
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
