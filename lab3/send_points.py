#!/usr/bin/env python3

# Bill Smart, smartw@oregonstate.edu
#
# send_points.py
# Send navigation targets to the robot

import rclpy
from rclpy.node import Node
import numpy as np
from threading import Lock

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from nav_targets.action import NavTarget
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from nav_msgs.msg import OccupancyGrid

from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer

from lab3.path_planning import dijkstra, is_free
from lab3.exploring import find_all_possible_goals, find_best_point, find_waypoints


class SendPoints(Node):
    def __init__(self, points):
        """Initialize way points
        @param points: iterable list of x,y tuples in map coordinates
        """
        super().__init__('send_points')

        self.mutex = Lock()

        self.action_client = ActionClient(node=self, action_type=NavTarget, action_name='nav_target')

        self.next_goal_index = 0
        self.goal_points = [p for p in points]
        self.last_distance = 1e30

        self._goal_handle = None
        self._send_goal_future = None
        self._result_future = None
        self._cancel_future = None

        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.tf_buffer = Buffer()
        self.transform_listener = TransformListener(self.tf_buffer, self)

        self.start_timer = self.create_timer(1.0, self._start_action_client)

        self.goal_markers = None
        self.path_markers = None
        self.reachable_markers = None

        self.goal_marker_pub = self.create_publisher(MarkerArray, 'goal_points', 1)
        self.path_marker_pub = self.create_publisher(MarkerArray, 'path_points', 1)
        self.reachable_marker_pub = self.create_publisher(MarkerArray, 'reachable_points', 1)

        # Extra state for final project behavior
        self.need_replan = True
        self.latest_map = None
        self.last_plan_time = 0.0
        self.min_plan_period = 1.5

        # Remember places we already tried so we do not keep looping
        self.visited_goal_points = []
        self.visited_goal_radius = 0.75

        self.exploration_done = False
        self.current_goal_failed = False

        self._set_goal_markers()

    def _start_action_client(self):
        """Called by the timer whenever a new goal needs to be kicked off."""
        self.start_timer.cancel()

        if self.exploration_done:
            self.get_logger().info("Exploration complete - no more goals to send")
            return

        if self.next_goal_index == 0:
            self.get_logger().info("Start driver.py to get started")
            self.action_client.wait_for_server()

        if self.next_goal_index >= len(self.goal_points):
            self.get_logger().info("No more points to send right now")
            self.need_replan = True
            return

        if self.next_goal_index == 0:
            self._set_goal_markers()

        pt = self.goal_points[self.next_goal_index]
        self.next_goal_index += 1

        goal = NavTarget.Goal()
        goal.goal.header.frame_id = 'odom'
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.point.x = float(pt[0])
        goal.goal.point.y = float(pt[1])
        goal.goal.point.z = 0.0

        self.get_logger().info(
            f"Sending goal request... {self.next_goal_index - 1} of {len(self.goal_points)} {(pt[0], pt[1])}"
        )

        self._send_goal_future = self.action_client.send_goal_async(
            goal=goal,
            feedback_callback=self._feedback_callback
        )
        self._send_goal_future.add_done_callback(self._goal_sent_callback)

    def _goal_sent_callback(self, future: Future):
        """Called when the server says it got the goal."""
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().warn("Action server not available; did you kill driver.py?")
        else:
            self.get_logger().info("Goal accepted")
            self._result_future = self._goal_handle.get_result_async()
            self._result_future.add_done_callback(self._goal_done_callback)

    def _goal_done_callback(self, future: Future):
        """Called when the server says it finished the goal."""
        result = future.result().result

        # Goal that was just attempted
        finished_index = max(0, self.next_goal_index - 1)
        if finished_index < len(self.goal_points):
            tried_pt = self.goal_points[finished_index]
            self._remember_goal_point(tried_pt)

        if result.success:
            self.get_logger().info(f"Got to goal {self.next_goal_index}")

            if self.next_goal_index < len(self.goal_points):
                self.start_timer.reset()
            else:
                self.get_logger().info("Finished current waypoint list, need new plan")
                self.need_replan = True
        else:
            self.get_logger().info(f"Did not get to goal {self.next_goal_index}, requesting replan")
            self.current_goal_failed = True
            self.need_replan = True

        self._send_goal_future = None
        self._result_future = None
        self._cancel_future = None
        self._goal_handle = None

    def _feedback_callback(self, feedback):
        """Feedback from driver while moving toward current target."""
        self.last_distance = feedback.feedback.distance.data

    def _cancel_response_callback(self, future: Future):
        """Called when the driver confirms a goal cancel."""
        cancel_response = future.result()
        self.get_logger().info(f'Cancel request accepted by server: {cancel_response.return_code}')
        self._send_goal_future = None
        self._result_future = None
        self._cancel_future = None
        self._goal_handle = None

        # If cancel happened because we are replanning, start fresh from new list
        if not self.exploration_done:
            self.start_timer.reset()

    def skip_current_goal(self):
        """Cancels the current goal and moves to the next."""
        if not self._goal_handle:
            self.get_logger().info("No active goals to skip")
        elif self._cancel_future:
            self.get_logger().info("Already skipping goal, wait for this to finish first")
        else:
            self.get_logger().info(f"Skipping to next goal {self.next_goal_index} of {len(self.goal_points)}")
            self._cancel_future = self._goal_handle.cancel_goal_async()
            self._cancel_future.add_done_callback(self._cancel_response_callback)

    def completed_all_goals(self):
        """Returns True if all current goals have been completed."""
        return self.next_goal_index >= len(self.goal_points)

    def add_more_goal_points(self, goal_pts: list):
        """Append more goals to current list."""
        for pt in goal_pts:
            self.goal_points.append((pt[0], pt[1]))

        self._set_goal_markers()

        if self._result_future is None:
            self.start_timer.reset()

    def replace_goal_points(self, goal_pts: list, skip_current: bool):
        """Replace the current list of goals."""
        self.next_goal_index = 0
        self.goal_points = [(p[0], p[1]) for p in goal_pts]
        self._set_goal_markers()

        if skip_current:
            self.skip_current_goal()
        elif self._result_future is None and not self.exploration_done:
            self.start_timer.reset()

    def _remember_goal_point(self, pt_xy):
        """Store tried goal areas so we do not repeat them."""
        for old_pt in self.visited_goal_points:
            if np.hypot(pt_xy[0] - old_pt[0], pt_xy[1] - old_pt[1]) < self.visited_goal_radius:
                return
        self.visited_goal_points.append((pt_xy[0], pt_xy[1]))

    def _already_tried_goal(self, pt_xy):
        """True if goal is too close to a previously tried goal."""
        for old_pt in self.visited_goal_points:
            if np.hypot(pt_xy[0] - old_pt[0], pt_xy[1] - old_pt[1]) < self.visited_goal_radius:
                return True
        return False

    def _set_goal_markers(self):
        """Update goal markers whenever goals change."""
        if self.goal_markers is None:
            self.goal_markers = MarkerArray()

        with self.mutex:
            line_marker = Marker()
            line_marker.header.frame_id = 'odom'
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.id = 0
            line_marker.scale.x = 0.1
            line_marker.scale.y = 0.1
            line_marker.scale.z = 0.1
            line_marker.color.r = 0.0
            line_marker.color.g = 0.0
            line_marker.color.b = 1.0
            line_marker.color.a = 1.0
            line_marker.points = []

            for p in self.goal_points:
                pt = Point()
                pt.x = p[0]
                pt.y = p[1]
                pt.z = 0.0
                line_marker.points.append(pt)

            self.goal_markers.markers = [line_marker]

            for indx, point in enumerate(self.goal_points):
                marker = Marker()
                marker.header.frame_id = 'odom'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = line_marker.id + indx + 1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                self.goal_markers.markers.append(marker)

        self.goal_marker_pub.publish(self.goal_markers)

    def _set_path_markers(self, path_list, skip=5):
        """Update path markers in map coordinates."""
        if self.path_markers is None:
            self.path_markers = MarkerArray()

        with self.mutex:
            line_marker = Marker()
            line_marker.header.frame_id = 'odom'
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.id = 10000
            line_marker.scale.x = 0.1
            line_marker.scale.y = 0.1
            line_marker.scale.z = 0.1
            line_marker.color.r = 1.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 1.0
            line_marker.points = []

            for p in path_list[0::skip]:
                pt = Point()
                pt.x = p[0]
                pt.y = p[1]
                pt.z = 0.0
                line_marker.points.append(pt)

            self.path_markers.markers = [line_marker]

            for indx, point in enumerate(path_list[0::skip]):
                marker = Marker()
                marker.header.frame_id = 'odom'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = line_marker.id + indx + 1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                self.path_markers.markers.append(marker)

        self.path_marker_pub.publish(self.path_markers)

    def _set_reachable_markers(self, points):
        """Draw reachable/explore candidate points in map coordinates."""
        if self.reachable_markers is None:
            self.reachable_markers = MarkerArray()

        with self.mutex:
            self.reachable_markers.markers = []

            for indx, point in enumerate(points):
                marker = Marker()
                marker.header.frame_id = 'odom'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = 20000 + indx + 1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.5
                marker.color.a = 1.0
                self.reachable_markers.markers.append(marker)

        self.reachable_marker_pub.publish(self.reachable_markers)

    def set_marker_points(self):
        self._set_goal_markers()

    def from_map_to_image(self, map_msg: OccupancyGrid, pt_xy=(0.0, 0.0)):
        """Convert map coordinates to image coordinates."""
        info = map_msg.info
        im_u = int((pt_xy[0] - info.origin.position.x) / info.resolution)
        im_v = int((pt_xy[1] - info.origin.position.y) / info.resolution)
        return (im_u, im_v)

    def from_image_to_map(self, map_msg: OccupancyGrid, pt_uv=(0, 0)):
        """Convert image coordinates to map coordinates."""
        info = map_msg.info
        pt_x = pt_uv[0] * info.resolution + info.origin.position.x
        pt_y = pt_uv[1] * info.resolution + info.origin.position.y
        return (pt_x, pt_y)

    def _nearest_free_pixel(self, im_thresh, start_uv, max_radius=8):
        """If start pixel is not free, search nearby for a free one."""
        if is_free(im_thresh, start_uv):
            return start_uv

        width = im_thresh.shape[1]
        height = im_thresh.shape[0]

        for radius in range(1, max_radius + 1):
            for du in range(-radius, radius + 1):
                for dv in range(-radius, radius + 1):
                    test_u = start_uv[0] + du
                    test_v = start_uv[1] + dv
                    if 0 <= test_u < width and 0 <= test_v < height:
                        if is_free(im_thresh, (test_u, test_v)):
                            return (test_u, test_v)

        return start_uv

    def _choose_new_plan(self, map_msg, im_thresh, robot_current_loc_in_image):
        """Build a new path of waypoints in map coordinates."""
        all_unseen_pts = find_all_possible_goals(im_thresh)

        reachable_pts_map = [self.from_image_to_map(map_msg, p) for p in all_unseen_pts]
        self._set_reachable_markers(reachable_pts_map)

        best_goal_image = find_best_point(im_thresh, all_unseen_pts, robot_current_loc_in_image)
        if best_goal_image is None:
            return [], reachable_pts_map

        # If this frontier area was already tried, search for another one
        filtered_candidates = []
        for p in all_unseen_pts:
            map_xy = self.from_image_to_map(map_msg, p)
            if not self._already_tried_goal(map_xy):
                filtered_candidates.append(p)

        if filtered_candidates:
            best_goal_image = find_best_point(im_thresh, filtered_candidates, robot_current_loc_in_image)

        if best_goal_image is None:
            return [], reachable_pts_map

        path = dijkstra(im_thresh, robot_current_loc_in_image, best_goal_image)
        path_waypoints = find_waypoints(im_thresh, path)

        path_pts_map = []
        for p in path_waypoints:
            map_xy = self.from_image_to_map(map_msg, p)
            path_pts_map.append(map_xy)

        self._set_path_markers(path_pts_map, 1)
        return path_pts_map, reachable_pts_map

    def map_callback(self, map_msg: OccupancyGrid):
        """Called when the SLAM map gets updated."""
        self.latest_map = map_msg

        im = np.array(map_msg.data, dtype=np.int8)
        im = im.reshape((map_msg.info.height, map_msg.info.width))

        im_thresh = np.zeros(im.shape, dtype=np.uint8)
        im_thresh[im < 10] = 255
        im_thresh[im >= 100] = 0
        im_thresh[im == -1] = 128

        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().info(f"Transform not ready yet: {e}")
            return

        robot_current_loc_in_map = (
            transform.transform.translation.x,
            transform.transform.translation.y
        )
        robot_current_loc_in_image = self.from_map_to_image(map_msg, robot_current_loc_in_map)
        robot_current_loc_in_image = self._nearest_free_pixel(im_thresh, robot_current_loc_in_image)

        now_sec = self.get_clock().now().nanoseconds / 1e9
        enough_time_passed = (now_sec - self.last_plan_time) > self.min_plan_period

        active_goal_running = (self._result_future is not None)
        no_goals_left = self.completed_all_goals()

        should_replan = self.need_replan or (no_goals_left and not active_goal_running)

        if not should_replan:
            return

        if not enough_time_passed:
            return

        self.last_plan_time = now_sec

        try:
            path_pts, reachable_pts = self._choose_new_plan(
                map_msg,
                im_thresh,
                robot_current_loc_in_image
            )
        except IndexError:
            self.get_logger().info("Robot or goal location not in image map")
            return
        except ValueError:
            self.get_logger().info(f"Planning failed from robot location {robot_current_loc_in_image}")
            return

        if len(path_pts) == 0:
            if len(reachable_pts) == 0:
                self.get_logger().info("No more reachable unexplored goals - exploration complete")
                self.exploration_done = True
            else:
                self.get_logger().info("Could not build a path right now, will try again")
            return

        self.get_logger().info(f"Replacing way points with new ones {path_pts}")

        self.need_replan = False
        self.current_goal_failed = False

        if active_goal_running:
            self.replace_goal_points(path_pts, True)
        else:
            self.replace_goal_points(path_pts, False)


def main(args=None):
    rclpy.init(args=args)

    # Start with a small reasonable seed path; after first map, replanning takes over.
    points = [(-4.5, -3.0), (-4.5, 0.0), (-1.0, 0.0)]
    send_points = SendPoints(points)

    executor = MultiThreadedExecutor()
    executor.add_node(send_points)
    executor.spin()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
