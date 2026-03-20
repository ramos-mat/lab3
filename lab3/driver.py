#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from math import atan2, sqrt, pi, cos, sin
import numpy as np

from geometry_msgs.msg import TwistStamped, PointStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle

from nav_targets.action import NavTarget

from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_geometry_msgs import do_transform_point

from rclpy.executors import MultiThreadedExecutor


class Lab3Driver(Node):
    def __init__(self, threshold=0.15):
        super().__init__('driver')

        self.goal = None
        self.threshold = threshold

        self.target_marker = None

        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 1)
        self.target_pub = self.create_publisher(Marker, 'current_target', 1)

        self.sub = self.create_subscription(LaserScan, 'base_scan', self.scan_callback, 10)

        self.tf_buffer = Buffer()
        self.transform_listener = TransformListener(self.tf_buffer, self)

        self.action_server = ActionServer(
            node=self,
            action_type=NavTarget,
            action_name="nav_target",
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_accept_callback,
            cancel_callback=self.cancel_callback,
            execute_callback=self.action_callback
        )

        self.target = PointStamped()
        self.target.point.x = 0.0
        self.target.point.y = 0.0

        self.target_dist = None
        self.target_angle = None

        self.avoiding = False
        self.avoid_dir = 0
        self.avoid_turn_bias = 0.8
        self.avoid_speed = 0.08

        self.angle_gain = 2.5
        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0
        self.alpha = 0.85

        self.marker_timer = self.create_timer(1.0, self._marker_callback)

        self.count_since_last_scan = 0
        self.print_twist_messages = False
        self.print_distance_messages = False

    def zero_twist(self):
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
    local_goal = self.goal

    if local_goal is None:
        # No goal, get rid of marker if there is one
        if self.target_marker:
            self.target_marker.action = Marker.DELETE
            self.target_pub.publish(self.target_marker)
            self.target_marker = None
            self.get_logger().info("Driver: Had an existing target marker; removing")
        return

    # If we do not currently have a marker, make one
    if not self.target_marker:
        self.target_marker = Marker()
        self.target_marker.header.frame_id = local_goal.header.frame_id
        self.target_marker.id = 0
        self.get_logger().info("Driver: Creating Marker")

    # Build a marker for the target point
    self.target_marker.header.stamp = self.get_clock().now().to_msg()
    self.target_marker.header.frame_id = local_goal.header.frame_id
    self.target_marker.type = Marker.SPHERE
    self.target_marker.action = Marker.ADD
    self.target_marker.pose.position = local_goal.point
    self.target_marker.scale.x = 0.3
    self.target_marker.scale.y = 0.3
    self.target_marker.scale.z = 0.3
    self.target_marker.color.r = 0.0
    self.target_marker.color.g = 1.0
    self.target_marker.color.b = 0.0
    self.target_marker.color.a = 1.0

    # Publish the marker
    self.target_pub.publish(self.target_marker)

    # Turn off the timer so we do not keep making and deleting the target Marker
    self.marker_timer.cancel()

    def goal_accept_callback(self, goal_request: ServerGoalHandle):
        self.get_logger().info("Received a goal request")
        self.marker_timer.reset()
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle: ServerGoalHandle):
        self.get_logger().info('Received a cancel request')

        self.goal = None
        self.avoiding = False
        self.avoid_dir = 0

        t = self.zero_twist()
        self.cmd_pub.publish(t)

        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0

        self.marker_timer.reset()
        return CancelResponse.ACCEPT

    def close_enough(self):
        if self.target_dist is None:
            return False
        return self.target_dist < self.threshold

    def distance_to_target(self):
        if self.target_dist is None:
            return float('inf')
        return self.target_dist

    def action_callback(self, goal_handle: ServerGoalHandle):
        self.get_logger().info(f'Received an execute goal request... {goal_handle.request.goal.point}')

        self.goal = PointStamped()
        self.goal.header = goal_handle.request.goal.header
        self.goal.point = goal_handle.request.goal.point

        result = NavTarget.Result()
        result.success = False

        self.set_target()

        best_dist = self.target_dist if self.target_dist is not None else 1e9
        no_progress_loops = 0
        rate = self.create_rate(2.0)

        while not self.close_enough():
            if not self.goal:
                self.get_logger().info("Goal was canceled")
                return result

            self.set_target()

            if self.target_dist is not None:
                if self.target_dist < best_dist - 0.05:
                    best_dist = self.target_dist
                    no_progress_loops = 0
                else:
                    no_progress_loops += 1

            feedback = NavTarget.Feedback()
            feedback.distance.data = self.distance_to_target()
            goal_handle.publish_feedback(feedback)

            if no_progress_loops > 12:
                self.get_logger().info("Not making progress, failing current goal")
                self.goal = None
                self.avoiding = False
                self.avoid_dir = 0

                t = self.zero_twist()
                self.cmd_pub.publish(t)

                self.prev_linear_x = 0.0
                self.prev_angular_z = 0.0
                return result

            rate.sleep()

        self.marker_timer.reset()

        self.goal = None
        self.avoiding = False
        self.avoid_dir = 0

        t = self.zero_twist()
        self.cmd_pub.publish(t)

        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0

        self.get_logger().info("Completed goal")

        goal_handle.succeed()
        result.success = True
        return result

    def set_target(self):
        if self.goal:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            self.target = do_transform_point(self.goal, transform)

            euler_ang = -atan2(
                2 * transform.transform.rotation.z * transform.transform.rotation.w,
                1.0 - 2 * transform.transform.rotation.z * transform.transform.rotation.z
            )

            x = self.goal.point.x - transform.transform.translation.x
            y = self.goal.point.y - transform.transform.translation.y

            rot_x = x * cos(euler_ang) - y * sin(euler_ang)
            rot_y = x * sin(euler_ang) + y * cos(euler_ang)

            self.target_dist = sqrt(rot_x * rot_x + rot_y * rot_y)
            self.target_angle = atan2(rot_y, rot_x)

            self.target.point.x = rot_x
            self.target.point.y = rot_y

            if self.print_distance_messages:
                self.get_logger().info(
                    f'Target relative to robot: ({self.target.point.x:.2f}, {self.target.point.y:.2f}), '
                    f'orig {(self.goal.point.x, self.goal.point.y)}'
                )
        else:
            if self.print_distance_messages:
                self.get_logger().info('No target to get distance to')
            self.target = None
            self.target_dist = None
            self.target_angle = None

        return self.target

    def scan_callback(self, scan):
        if self.print_twist_messages:
            self.get_logger().info("In scan callback")

        self.count_since_last_scan = 0

        if self.goal:
            self.set_target()
            t = self.get_twist(scan)
        else:
            t = self.zero_twist()
            if self.print_twist_messages:
                self.get_logger().info("No goal, sitting still")

        self.cmd_pub.publish(t)

    def get_obstacle(self, scan):
        if not self.target:
            return False, 0.0, 0.0, float('inf'), float('inf'), float('inf')

        ranges = np.array(scan.ranges, dtype=float)
        n = len(ranges)
        thetas = np.linspace(scan.angle_min, scan.angle_max, n, dtype=float)

        front_mask = np.abs(thetas) < 0.25
        left_mask = (thetas > 0.25) & (thetas < 1.2)
        right_mask = (thetas < -0.25) & (thetas > -1.2)

        def min_dist(mask):
            vals = ranges[mask]
            vals = vals[(vals > 0.0) & (~np.isinf(vals))]
            return float(np.min(vals)) if len(vals) > 0 else float('inf')

        front_dist = min_dist(front_mask)
        left_dist = min_dist(left_mask)
        right_dist = min_dist(right_mask)

        obstacle_threshold = 0.6
        obstacle_detected = front_dist < obstacle_threshold

        if abs(left_dist - right_dist) < 0.12:
            obs_turn_dir = +1
        elif left_dist > right_dist:
            obs_turn_dir = +1
        else:
            obs_turn_dir = -1

        obs_speed = self.avoid_speed if obstacle_detected else 0.0
        obs_turn = float(self.avoid_turn_bias * obs_turn_dir)

        return obstacle_detected, obs_speed, obs_turn, front_dist, left_dist, right_dist

    def get_twist(self, scan):
        t = self.zero_twist()

        if self.target is None or self.target_dist is None:
            return t

        angle = self.target_angle
        dist = self.target_dist

        min_speed = 0.06
        max_speed = 0.35
        max_turn = np.pi * 0.4

        speed = 0.9 * dist
        speed = max(min_speed, min(max_speed, speed))

        obstacle_detected, obs_speed, obs_turn_raw, front_dist, left_dist, right_dist = self.get_obstacle(scan)
        obs_turn = float(max(-max_turn, min(max_turn, obs_turn_raw)))

        cmd_v = 0.0
        cmd_w = 0.0

        if self.close_enough():
            self.avoiding = False
            self.avoid_dir = 0
            cmd_v = 0.0
            cmd_w = 0.0
        else:
            side_clearance = 0.22
            hard_side_clearance = 0.16
            safe_stop = 0.55
            front_escape = 0.32

            too_close_left = left_dist < side_clearance
            too_close_right = right_dist < side_clearance
            trapped_left = left_dist < hard_side_clearance
            trapped_right = right_dist < hard_side_clearance

            blocking = obstacle_detected and (front_dist < self.target_dist + 0.08)

            if (blocking or too_close_left or too_close_right) and not self.avoiding:
                self.avoiding = True
                if trapped_left and not trapped_right:
                    self.avoid_dir = -1
                elif trapped_right and not trapped_left:
                    self.avoid_dir = 1
                elif too_close_left and not too_close_right:
                    self.avoid_dir = -1
                elif too_close_right and not too_close_left:
                    self.avoid_dir = 1
                elif abs(left_dist - right_dist) < 0.12:
                    self.avoid_dir = 1 if (left_dist >= right_dist) else -1
                else:
                    self.avoid_dir = 1 if (left_dist > right_dist) else -1

            if self.avoiding:
                release_clear_dist = 0.7

                if front_dist > release_clear_dist and min(left_dist, right_dist) > 0.28:
                    self.avoiding = False
                    self.avoid_dir = 0
                else:
                    if front_dist < front_escape or trapped_left or trapped_right:
                        cmd_v = 0.0
                    else:
                        cmd_v = float(self.avoid_speed)

                    cmd_w = float(max(-max_turn, min(max_turn, 1.1 * self.avoid_turn_bias * self.avoid_dir)))

            if not self.avoiding:
                if front_dist < safe_stop:
                    self.avoiding = True
                    if too_close_left and not too_close_right:
                        self.avoid_dir = -1
                    elif too_close_right and not too_close_left:
                        self.avoid_dir = 1
                    elif self.avoid_dir == 0:
                        self.avoid_dir = 1 if (left_dist > right_dist) else -1

                    cmd_v = 0.0
                    cmd_w = float(max(-max_turn, min(max_turn, self.avoid_turn_bias * self.avoid_dir)))
                else:
                    turn = self.angle_gain * angle
                    cmd_w = float(max(-max_turn, min(max_turn, turn)))

                    angle_threshold_for_forward = 0.8
                    if abs(angle) < angle_threshold_for_forward:
                        cmd_v = float(speed)
                    else:
                        cmd_v = 0.0

                    if too_close_left or too_close_right:
                        cmd_v = min(cmd_v, 0.12)
                        if too_close_left and not too_close_right:
                            cmd_w = min(cmd_w, -0.45)
                        elif too_close_right and not too_close_left:
                            cmd_w = max(cmd_w, 0.45)

        final_linear_x = self.alpha * cmd_v + (1 - self.alpha) * self.prev_linear_x
        final_angular_z = self.alpha * cmd_w + (1 - self.alpha) * self.prev_angular_z

        self.prev_linear_x = final_linear_x
        self.prev_angular_z = final_angular_z

        t.twist.linear.x = float(final_linear_x)
        t.twist.angular.z = float(final_angular_z)

        if self.print_twist_messages:
            self.get_logger().info(f"Setting twist forward {t.twist.linear.x} angle {t.twist.angular.z}")

        return t


def main(args=None):
    rclpy.init(args=args)

    driver = Lab3Driver()

    executor = MultiThreadedExecutor()
    executor.add_node(driver)
    executor.spin()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
