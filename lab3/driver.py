#!/usr/bin/env python3

# Bill Smart, smartw@oregonstate.edu
#
# driver.py
# Drive the robot towards a goal, going around an object


# Every Python node in ROS2 should include these lines.  rclpy is the basic Python
# ROS2 stuff, and Node is the class we're going to use to set up the node.
import rclpy
from rclpy.node import Node

# Velocity commands are given with Twist messages, from geometry_msgs
from geometry_msgs.msg import Twist, PoseStamped

# math stuff
from math import atan2, tanh, sqrt, pi, fabs, cos, sin
import numpy as np

# Header for the twist message
from std_msgs.msg import Header

# The twist command and the goal
from geometry_msgs.msg import TwistStamped, PointStamped

# For publishing markers to rviz
from visualization_msgs.msg import Marker

# The laser scan message type
from sensor_msgs.msg import LaserScan

# These are all for setting up the action server/client
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle

# This is the format of the message sent by the client - it is another node under lab 2
from nav_targets.action import NavTarget

# These are for transforming points/targets in the world into a point in the robot's coordinate space
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_geometry_msgs import do_transform_point

# This sets up multi-threading so the laser scan can happen at the same time we're processing the target goal
from rclpy.executors import MultiThreadedExecutor


class Lab3Driver(Node):
    def __init__(self, threshold=0.1):
        """ We have parameters this time
        @param threshold - how close do you have to be before saying you're at the goal? Set to width of robot
        """
        # Initialize the parent class, giving it a name.  The idiom is to use the
        # super() class.
        super().__init__('driver')

        # Goal will be set later. The action server will set the goal; you don't set it directly
        self.goal = None
        # A controllable parameter for how close you have to be to the goal to say "I'm there"
        self.threshold = threshold

        # Make a Marker to put in RViz to show the current goal/target the robot is aiming for
        self.target_marker = None

        # Publisher before subscriber
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 1)
        # Publish the current target as a marker (so RViz can show it)
        self.target_pub = self.create_publisher(Marker, 'current_target', 1)

        # Subscriber after publisher; this is the laser scan
        self.sub = self.create_subscription(LaserScan, 'base_scan', self.scan_callback, 10)

        # Create a buffer to put the transform data in
        self.tf_buffer = Buffer()
        
        # This sets up a listener for all of the transform types created
        self.transform_listener = TransformListener(self.tf_buffer, self)

        # Action client for passing "target" messages/state around
        # An action has a goal, feedback, and a result. This class (the driver) will have the action server side, and be
        #   responsible for sending feed back and result
        # The SendPoints class will have the action client - it will send the goals and cancel the goal and send another when 
        #    the server says it has completed the goal
        # There is an initial call and response (are you ready for a target?) followed by the target itself
        #   goal_accept_callback handles accepting the goal
        #   cancel_callback is called if the goal is actually canceled by the action client
        #   execute_callback actually starts moving toward the goal
        self.action_server = ActionServer(node=self,
                                    action_type=NavTarget,
                                    action_name="nav_target",
                                    callback_group=ReentrantCallbackGroup(),
                                    goal_callback=self.goal_accept_callback,
                                    cancel_callback=self.cancel_callback,
                                    execute_callback=self.action_callback)

        # This is the goal in the robot's coordinate system, calculated in set_target
        self.target = PointStamped()
        self.target.point.x = 0.0
        self.target.point.y = 0.0

        # GUIDE: Declare any variables here
    
        self.target_dist = None
        self.target_angle = None

        self.avoiding = False
        self.avoid_dir = 0  #+1 prefer left, -1 = prefer right, 0 = none
        self.avoid_clear_count = 0
        self.avoid_turn_bias = 0.48
        self.avoid_speed = 0.15

        self.angle_gain = 2.5   
        self.prev_linear_x = 0.0  
        self.prev_angular_z = 0.0 
        self.alpha = 0.85         

        # Timer to make sure we publish the target marker (once we get a goal)
        self.marker_timer = self.create_timer(1.0, self._marker_callback)

        self.count_since_last_scan = 0
        self.print_twist_messages = False
        self.print_distance_messages = False

    def zero_twist(self):
        """This is a helper class method to create and zero-out a twist"""
        # Don't really need to do this - the default values are zero - but can't hurt
        t = TwistStamped()
        t.header.frame_id = 'base_link'
        t.header.stamp = self.get_clock().now().to_msg()
        t.twist.linear.x = 0.0
        t.twist.linear.y = 0.0
        t.twist.linear.z = 0.0
        t.twist.angular.x = 0.0
        t.twist.angular.y = 0.0
        t.twist.angular.z = 0.0

        return t

    def _marker_callback(self):
        """Publishes the target so it shows up in RViz"""
        goal = self.goal

        if not goal:
            # No goal, get rid of marker if there is one
            if self.target_marker:
                self.target_marker.action = Marker.DELETE
                self.target_pub.publish(self.target_marker)
                self.target_marker = None
                self.get_logger().info(f"Driver: Had an existing target marker; removing")
            return
        
        # If we do not currently have a marker, make one
        if not self.target_marker:
            self.target_marker = Marker()
            self.target_marker.header.frame_id = goal.header.frame_id
            self.target_marker.id = 0
        
            self.get_logger().info(f"Driver: Creating Marker")

        # Build a marker for the target point
        #   - this prints out the green dot in RViz (the current target)
        self.target_marker.header.stamp = self.get_clock().now().to_msg()
        self.target_marker.header.frame_id = goal.header.frame_id
        self.target_marker.type = Marker.SPHERE
        self.target_marker.action = Marker.ADD
        self.target_marker.pose.position = goal.point
        self.target_marker.scale.x = 0.3
        self.target_marker.scale.y = 0.3
        self.target_marker.scale.z = 0.3
        self.target_marker.color.r = 0.0
        self.target_marker.color.g = 1.0
        self.target_marker.color.b = 0.0
        self.target_marker.color.a = 1.0

        # Publish the marker
        self.target_pub.publish(self.target_marker)

        # Turn off the timer so we don't just keep making and deleting the target Marker
        #   Will get turned back on when we get an goal request
        self.marker_timer.cancel()

    def goal_accept_callback(self, goal_request : ServerGoalHandle):
        """Accept a request for a new goal"""
        self.get_logger().info("Received a goal request")

        # Timer to make sure we publish the new target
        self.marker_timer.reset()

        # Accept all goals. You can use this (in the future) to NOT accept a goal if you want
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle : ServerGoalHandle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received a cancel request')

        # Make sure our goal is removed
        self.goal = None

        # ...and robot stops
        t = self.zero_twist()
        self.cmd_pub.publish(t)
        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0
                
        # Timer to make sure we remove the current target (if there is one)
        self.marker_timer.reset()

        return CancelResponse.ACCEPT
    
    def close_enough(self):
        """ Return true if close enough to goal. This will be used in action_callback to stop moving toward the goal
        @ return true/false """
        if self.target_dist is None:
            return False

        return self.target_dist < self.threshold

    def distance_to_target(self):
        """ Communicate with send points - set to distance to target"""
        if self.target_dist is None:
            return float('inf')
        return self.target_dist
    
    # Respond to the action request.
    def action_callback(self, goal_handle : ServerGoalHandle):
        """ This gets called when the new goal is sent by SendPoints
        @param goal_handle - this has the new goal
        @return a NavTarget return when done """

        self.get_logger().info(f'Received an execute goal request... {goal_handle.request.goal.point}')
    
        # Save the new goal as a stamped point
        self.goal = PointStamped()
        self.goal.header = goal_handle.request.goal.header
        self.goal.point = goal_handle.request.goal.point
        
        # Build a result to send back
        result = NavTarget.Result()
        result.success = False

        # Reset target
        self.set_target()

        # Keep publishing feedback, then sleeping (so the laser scan can happen)
        # GUIDE: If you aren't making progress, stop the while loop and mark the goal as failed
        best_dist = self.target_dist if self.target_dist is not None else 1e9
        no_progress_loops = 0
        rate = self.create_rate(2.0)
        while not self.close_enough():
            if not self.goal:
                self.get_logger().info(f"Goal was canceled")
                return result

            # Recompute target after motion updates from scan callback
            self.set_target()

            if self.target_dist is not None:
                if self.target_dist < best_dist - 0.05:
                    best_dist = self.target_dist
                    no_progress_loops = 0
                else:
                    no_progress_loops += 1
            
            feedback = NavTarget.Feedback()
            feedback.distance.data = self.distance_to_target()
            
            # Publish feedback - this gets sent back to send_points
            goal_handle.publish_feedback(feedback)

            # If progress stalls for too long, fail the goal so send_points can replan
            if no_progress_loops > 12:
                self.get_logger().info("Not making progress, failing current goal")
                self.goal = None
                t = self.zero_twist()
                self.cmd_pub.publish(t)
                self.prev_linear_x = 0.0
                self.prev_angular_z = 0.0
                return result

            # sleep so we can process the next scan
            rate.sleep()
            
        # Timer to make sure we remove the current target
        self.marker_timer.reset()

        # Don't keep processing goals
        self.goal = None 

        # Publish the zero twist
        t = self.zero_twist()
        self.cmd_pub.publish(t)
        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0

        self.get_logger().info(f"Completed goal")

        # Set the succeed value on the handle
        goal_handle.succeed()

        # Set the result to True and return
        result.success = True
        return result

    def set_target(self):
        """ Convert the goal into an x,y position (target) in the ROBOT's coordinate space
        @return the new target as a Point """

        goal = self.goal

        if goal:
            # Transforms for all coordinate frames in the robot are stored in a transform tree
            #  odom is the coordinate frame of the "world", base_link is the base link of the robot
            # A transform stores a rotation/translation to go from one coordinate system to the other
            transform = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))

            # This applies the transform to the Stamped Point
            #    Note: This does not work, for reasons that are unclear to me
            self.target = do_transform_point(goal, transform)

            euler_ang = -atan2(2 * transform.transform.rotation.z * transform.transform.rotation.w,
                               1.0 - 2 * transform.transform.rotation.z * transform.transform.rotation.z)

            x = goal.point.x - transform.transform.translation.x
            y = goal.point.y - transform.transform.translation.y

            rot_x = x * cos(euler_ang) - y * sin(euler_ang)
            rot_y = x * sin(euler_ang) + y * cos(euler_ang)

            self.target_dist = sqrt(rot_x * rot_x + rot_y * rot_y)
            self.target_angle = atan2(rot_y, rot_x)

            self.target.point.x = rot_x
            self.target.point.y = rot_y
            if self.print_distance_messages:
                self.get_logger().info(f'Target relative to robot: ({self.target.point.x:.2f}, {self.target.point.y:.2f}), orig ({goal.point.x, goal.point.y})')
            
        else:
            if self.print_distance_messages:
                self.get_logger().info(f'No target to get distance to')
            self.target = None  
            self.target_dist = None
            self.target_angle = None    
        
        # GUIDE: Calculate any additional variables here
        #  Remember that the target's location is in its own coordinate frame at 0,0, angle 0 (x-axis)
        #

        return self.target

    def scan_callback(self, scan):
        """ Lidar scan callback
        @param scan - has information about the scan, and the distances (see stopper.py in lab1)"""
    
        if self.print_twist_messages:
            self.get_logger().info("In scan callback")
        # Got a scan - set back to zero
        self.count_since_last_scan = 0

        # If we have a goal, then act on it, otherwise stay still
        if self.goal:
            # Recalculate the target point (assumes we've moved)
            self.set_target()

            # Call the method to actually calculate the twist
            t = self.get_twist(scan)
        else:
            t = self.zero_twist()
            #t.twist.linear.x = 0.1
            if self.print_twist_messages:
                self.get_logger().info(f"No goal, sitting still")

        # Publish the new twist
        self.cmd_pub.publish(t)

    def get_obstacle(self, scan):
        """ check if an obstacle
        @param scan - the lidar scan
        @return Currently True/False and speed, angular turn"""

        if not self.target:
            return False, 0.0, 0.0, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')
        
        # GUIDE: Use this method to collect obstacle information - is something in front of, to the left, or to 
        # the right of the robot? Start with your stopper code from Lab1
        ranges = np.array(scan.ranges, dtype = float)
        n = len(ranges)
        thetas = np.linspace(scan.angle_min, scan.angle_max, n, dtype = float)

        #define regions
        front_mask = np.abs(thetas) < 0.38
        left_mask = (thetas > 0.15) & (thetas < 1.45)
        right_mask = (thetas < -0.15) & (thetas > -1.45)
        front_left_mask = (thetas >= -0.05) & (thetas < 0.72)
        front_right_mask = (thetas <= 0.05) & (thetas > -0.72)
        back_mask = np.abs(np.abs(thetas) - np.pi) < 0.45
        back_left_mask = (thetas > 2.20) & (thetas < 3.05)
        back_right_mask = (thetas < -2.20) & (thetas > -3.05)
        
        def min_dist(mask):
            vals = ranges[mask]
            vals = vals[(vals > 0.0) & (~np.isinf(vals))]
            return float(np.min(vals)) if len(vals) > 0 else float('inf')

        front_dist = min_dist(front_mask)
        left_dist = min_dist(left_mask)
        right_dist = min_dist(right_mask)
        front_left_dist = min_dist(front_left_mask)
        front_right_dist = min_dist(front_right_mask)
        back_dist = min_dist(back_mask)
        back_left_dist = min_dist(back_left_mask)
        back_right_dist = min_dist(back_right_mask)

        #detection
        obstacle_threshold = 0.75 #meters
        obstacle_detected = front_dist < obstacle_threshold

        #decide consistent turn direction
        if abs(left_dist - right_dist) < 0.12:
            #prefer left
            obs_turn_dir = +1
        elif left_dist > right_dist:
            obs_turn_dir = +1    #turn left
        else:
            obs_turn_dir = -1    #turn right

        #set obs speed (small creep forward while avoiding)
        obs_speed = self.avoid_speed if obstacle_detected else 0.0

        obs_turn = float(self.avoid_turn_bias * obs_turn_dir)

        return obstacle_detected, obs_speed, obs_turn, front_dist, left_dist, right_dist, front_left_dist, front_right_dist, back_dist, back_left_dist, back_right_dist

    def get_twist(self, scan):
        """This is the method that calculate the twist
        @param scan - a LaserScan message with the current data from the LiDAR.  Use this for obstacle avoidance. 
            This is the same as your lab1 go and stop code
        @return a twist command"""
        
        t = self.zero_twist()

        # GUIDE:
        #  Step 1) Calculate the angle the robot has to turn to in order to point at the target
        #  Step 2) Set your speed based on how far away you are from the target, as before
        #  Step 3) Add code that veers left (or right) to avoid an obstacle in front of it
        # Reminder: t.linear.x = 0.1    sets the forward speed to 0.1
        #           t.angular.z = pi/2   sets the angular speed to 90 degrees per sec
        # Reminder 2: target is in self.target 
        #  Note: If the target is behind you, might turn first before moving
        #  Note: 0.4 is a good speed if nothing is in front of the robot
        if self.target is None or self.target_dist is None:
            return t

        angle = self.target_angle
        dist = self.target_dist

        min_speed = 0.06
        max_speed = 0.60
        max_turn = np.pi * 0.4

        # speed toward target
        speed = 0.9 * dist
        speed = max(min_speed, min(max_speed, speed))

        # check obstacles
        obstacle_detected, obs_speed, obs_turn_raw, front_dist, left_dist, right_dist, front_left_dist, front_right_dist, back_dist, back_left_dist, back_right_dist = self.get_obstacle(scan)

        # obs_turn isn't larger than max_turn
        obs_turn = float(max(-max_turn, min(max_turn, obs_turn_raw)))

        cmd_v = 0.0
        cmd_w = 0.0
        nearest_forward = min(front_dist, front_left_dist, front_right_dist)

        # if we're getting close enough, stop
        if self.close_enough():
            self.avoiding = False
            self.avoid_dir = 0
            self.avoid_clear_count = 0
            cmd_v = 0.0
            cmd_w = 0.0
        else:
            side_clearance = 0.24
            hard_side_clearance = 0.18
            safe_stop = 0.46
            front_escape = 0.28

            too_close_left = left_dist < side_clearance
            too_close_right = right_dist < side_clearance
            trapped_left = left_dist < hard_side_clearance
            trapped_right = right_dist < hard_side_clearance
            front_left_blocked = front_left_dist < 0.56
            front_right_blocked = front_right_dist < 0.56
            corner_trapped = front_dist < 0.40 and (too_close_left or too_close_right)

            blocking = obstacle_detected and (front_dist < self.target_dist + 0.08)
            # Slow down smoothly as we get closer to things in front of the robot.
            if nearest_forward < 0.85:
                scale = max(0.0, min(1.0, (nearest_forward - 0.20) / 0.65))
                speed = max(0.0, speed * scale)

            # if blocking iand not avoiding, start avoiding
            if (blocking or too_close_left or too_close_right) and not self.avoiding:
                self.avoiding = True
                self.avoid_clear_count = 0
                # Pick one side and stick with it instead of flipping every scan.
                if trapped_left and not trapped_right:
                    self.avoid_dir = -1
                elif trapped_right and not trapped_left:
                    self.avoid_dir = 1
                elif too_close_left and not too_close_right:
                    self.avoid_dir = -1
                elif too_close_right and not too_close_left:
                    self.avoid_dir = 1
                elif front_left_blocked and not front_right_blocked:
                    self.avoid_dir = -1
                elif front_right_blocked and not front_left_blocked:
                    self.avoid_dir = 1
                elif abs(left_dist - right_dist) < 0.12:
                    self.avoid_dir = 1 if (left_dist >= right_dist) else -1
                else:
                    self.avoid_dir = 1 if (left_dist > right_dist) else -1

            # if not blocking and not in avoiding state, normal behavior
            if self.avoiding:
                release_clear_dist = 0.62

                if front_dist > release_clear_dist and min(left_dist, right_dist) > 0.28:
                    # Require a few clear scans in a row before leaving avoidance mode.
                    self.avoid_clear_count += 1
                    if self.avoid_clear_count >= 4:
                        self.avoiding = False
                        self.avoid_dir = 0
                        self.avoid_clear_count = 0
                else:
                    self.avoid_clear_count = 0
                    if front_dist < front_escape or corner_trapped:
                        cmd_v = 0.0
                    elif trapped_left or trapped_right:
                        cmd_v = 0.07
                    else:
                        cmd_v = float(min(self.avoid_speed, 0.22))
                    cmd_w = float(max(-max_turn, min(max_turn, 1.05 * self.avoid_turn_bias * self.avoid_dir)))

            if not self.avoiding:
                if front_dist < safe_stop:
                    # force avoidance state
                    self.avoiding = True
                    self.avoid_clear_count = 0
                    # set avoid_dir if unknown
                    if too_close_left and not too_close_right:
                        self.avoid_dir = -1
                    elif too_close_right and not too_close_left:
                        self.avoid_dir = 1
                    elif front_left_blocked and not front_right_blocked:
                        self.avoid_dir = -1
                    elif front_right_blocked and not front_left_blocked:
                        self.avoid_dir = 1
                    elif self.avoid_dir == 0:
                        self.avoid_dir = 1 if (left_dist > right_dist) else -1
                    cmd_v = 0.0
                    cmd_w = float(max(-max_turn, min(max_turn, self.avoid_turn_bias * self.avoid_dir)))
                else:
                    # normal navigation
                    # angle_gain = 2.5
                    turn = self.angle_gain * angle 
                    cmd_w = float(max(-max_turn, min(max_turn, turn)))

                    # allow forward motion if target is in front
                    angle_threshold_for_foward = 0.8 
                    if abs(angle) < angle_threshold_for_foward:
                        cmd_v = float(speed)
                    else:
                        cmd_v = 0.0

                    need_wide_turn = (
                        abs(angle) > 1.2 and
                        (too_close_left or too_close_right or front_left_blocked or front_right_blocked)
                    )
                    if need_wide_turn and front_dist > 0.45:
                        cmd_v = max(cmd_v, 0.05)
                        if left_dist > right_dist + 0.05:
                            cmd_w = max(cmd_w, 0.45)
                        elif right_dist > left_dist + 0.05:
                            cmd_w = min(cmd_w, -0.45)

                    if too_close_left or too_close_right:
                        cmd_v = min(cmd_v, 0.16)
                        if too_close_left and not too_close_right:
                            cmd_w = min(cmd_w, -0.55)
                        elif too_close_right and not too_close_left:
                            cmd_w = max(cmd_w, 0.55)

                    if front_left_blocked and not front_right_blocked:
                        cmd_v = min(cmd_v, 0.11)
                        cmd_w = min(cmd_w, -0.55)
                    elif front_right_blocked and not front_left_blocked:
                        cmd_v = min(cmd_v, 0.11)
                        cmd_w = max(cmd_w, 0.55)

                    if corner_trapped:
                        cmd_v = 0.0
                        if too_close_left and not too_close_right:
                            cmd_w = max(cmd_w, 0.65)
                        elif too_close_right and not too_close_left:
                            cmd_w = min(cmd_w, -0.65)

        # Last safety clamp before smoothing/publish.
        if nearest_forward < 0.32:
            cmd_v = 0.0
        elif nearest_forward < 0.42:
            cmd_v = min(cmd_v, 0.06)
        elif nearest_forward < 0.58:
            cmd_v = min(cmd_v, 0.15)

        if min(left_dist, right_dist) < 0.20:
            cmd_v = min(cmd_v, 0.08)

        turn_ratio = abs(cmd_w) / max_turn if max_turn > 0 else 0.0
        if turn_ratio > 0.75:
            # Big turns get a smaller forward speed so the robot does not swing into obstacles.
            cmd_v = min(cmd_v, 0.05)
            if nearest_forward < 0.55 or min(left_dist, right_dist) < 0.26:
                cmd_v = min(cmd_v, 0.03)
        elif turn_ratio > 0.45:
            cmd_v = min(cmd_v, 0.16)

        rear_tight = min(back_dist, back_left_dist, back_right_dist) < 0.28
        if abs(cmd_w) > 0.45 and rear_tight:
            # If the robot is rotating with something close behind it, be much more careful.
            if front_dist > 0.45 and nearest_forward > 0.40:
                cmd_v = max(cmd_v, 0.05)
                cmd_w = max(-0.45, min(0.45, cmd_w))
            else:
                cmd_v = 0.0
                cmd_w = max(-0.28, min(0.28, cmd_w))

        final_linear_x = self.alpha * cmd_v + (1 - self.alpha) * self.prev_linear_x
        final_angular_z = self.alpha * cmd_w + (1 - self.alpha) * self.prev_angular_z

        # Save current smoothed commands for the next frame
        self.prev_linear_x = final_linear_x
        self.prev_angular_z = final_angular_z

        # Assign to the actual twist message
        t.twist.linear.x = float(final_linear_x)
        t.twist.angular.z = float(final_angular_z)

        if self.print_twist_messages:
            self.get_logger().info(f"Setting twist forward {t.twist.linear.x} angle {t.twist.angular.z}")
        
        return t
    
# The idiom in ROS2 is to use a function to do all of the setup and work.  This
# function is referenced in the setup.py file as the entry point of the node when
# we're running the node with ros2 run.  The function should have one argument, for
# passing command line arguments, and it should default to None.
def main(args=None):
    # Initialize rclpy.  We should do this every time.
    rclpy.init(args=args)

    # Make a node class.  The idiom in ROS2 is to encapsulte everything in a class
    # that derives from Node.
    driver = Lab3Driver()

    # Multi-threaded execution
    executor = MultiThreadedExecutor()
    executor.add_node(driver)
    executor.spin()
    
    # Make sure we shutdown everything cleanly.  This should happen, even if we don't
    # include this line, but you should do it anyway.
    rclpy.shutdown()
    

# If we run the node as a script, then we're going to start here.
if __name__ == '__main__':
    # The idiom in ROS2 is to set up a main() function and to call it from the entry
    # point of the script.
    main()
