#!/usr/bin/env python3

# Bill Smart, smartw@oregonstate.edu
#
# send_points.py
# Send navigation targets to the robot


# Every Python node in ROS2 should include these lines.  rclpy is the basic Python
# ROS2 stuff, and Node is the class we're going to use to set up the node.
import rclpy
from rclpy.node import Node

import numpy as np
from math import hypot
from scipy import ndimage

from threading import Lock

from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from nav_targets.action import NavTarget
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from nav_msgs.msg import OccupancyGrid

# These are for transforming points/targets in the world into a point in the robot's coordinate space
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer

# Your path planning
from lab3.path_planning import dijkstra, is_free
from lab3.exploring import find_all_possible_goals, find_best_point, find_waypoints


class SendPoints(Node):
	def __init__(self, points):
		""" Initialize way points
		@param - points, an iterable list of x,y tuples"""
		# Initialize the parent class, giving it a name.  The idiom is to use the
		# super() class.
		super().__init__('send_points')

		# A mutex to keep us safe during the list deletions.
		self.mutex = Lock()

		# An action server to send the requests to.
		self.action_client = ActionClient(node=self, action_type=NavTarget, action_name='nav_target')

		# Save the goal points for when we start up the action client/server
		self.next_goal_index = 0
		self.goal_points = [p for p in points]
		self.last_distance = 1e30  # The last distance to goal from the callback
		self.have_map = False
		self.need_new_plan = True
		self.current_frontier_goal_map = None
		self.completed_frontiers = []
		self.failed_frontiers = []
		self.last_feedback_log_distance = None

		# Parameters that hold the current state of the action client
		#   You don't need to mess with these
		self._goal_handle = None
		self._send_goal_future = None
		self._result_future = None
		self._cancel_future = None

		# Subscriber after publisher; this is the map
		self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',  # topic name
            self.map_callback,
            10
        )

		# Create a buffer to put the transform data in
		self.tf_buffer = Buffer()
        
		# This sets up a listener for all of the transform types created
		self.transform_listener = TransformListener(self.tf_buffer, self)
		
		# Timer to make sure we publish the target marker and start the goal sending
		self.start_timer = self.create_timer(0.05, self._start_action_client)

		# Three sets of markers - one for the goal points, one for reachable points, one for path points
		#    The last two are for you to use when getting paths/reachable points from the map
		#    Do not set these directly - use set_xxx methods
		self.goal_markers = None
		self.path_markers = None
		self.reachable_markers = None

		# Publishers for the RViz visualization
		self.goal_marker_pub = self.create_publisher(MarkerArray, 'goal_points', 1)
		self.path_marker_pub = self.create_publisher(MarkerArray, 'path_points', 1)
		self.reachable_marker_pub = self.create_publisher(MarkerArray, 'reachable_points', 1)


	def _start_action_client(self):
		""" This gets called by the timer whenever a new set of goals needs to be kicked off"""

		# Cancel the timer - we're starting
		self.start_timer.cancel()

		if self.next_goal_index == 0:
			# Wait for driver to start
			self.get_logger().info("Start driver.py to get started")
			self.action_client.wait_for_server()
		
		# Run out of goal points
		if self.next_goal_index >= len(self.goal_points):
			self.next_goal_index += 1
			self.get_logger().info("No more points to send")
			self.need_new_plan = True
			return
			
		if self.next_goal_index == 0:
			# First time through - make the marker points and publish them
			#. NOTE: You should call _set_goal_markers() anytime you change points()
			self._set_goal_markers()

		# send the next goal
		pt = self.goal_points[self.next_goal_index]
		self.next_goal_index += 1

		# Create the goal point in the world coordinate frame
		goal = NavTarget.Goal()
		goal.goal.header.frame_id = 'odom'
		goal.goal.header.stamp = self.get_clock().now().to_msg()

		goal.goal.point.x = float(pt[0])
		goal.goal.point.y = float(pt[1])
		goal.goal.point.z = 0.0

		self.get_logger().info(f'Sending goal request... {self.next_goal_index-1} of {len(self.goal_points)} {pt[0], pt[1]}')

		# Send the driver the message that we're ready to send a goal point
		self._send_goal_future: Future = self.action_client.send_goal_async(goal=goal, 
															feedback_callback=self._feedback_callback)
		# This sets the call back for when the driver says it got the goal request 
		self._send_goal_future.add_done_callback(self._goal_sent_callback)

	def _goal_sent_callback(self, future : Future):
		""" This gets called when the server says I got the goal
		@param future - communicate with the server"""

		self._goal_handle: ClientGoalHandle = future.result()
		if not self._goal_handle.accepted:
			self.warn(f"{self.get_name()}: Action server not available; did you kill driver.py?")
		else:
			self.get_logger().info(f"Goal accepted")
			# Add a callback for the actual driver executing the goal
			self._result_future: Future = self._goal_handle.get_result_async()
			self._result_future.add_done_callback(self._goal_done_callback)

	def _goal_done_callback(self, future : Future):
		""" This gets called when the server says I finished the goal"""
		result: NavTarget.Result = future.result().result
		self._send_goal_future = None
		self._result_future = None
		self._cancel_future = None

		if result.success:
			if self.completed_all_goals() and self.current_frontier_goal_map is not None:
				if not any(
					hypot(frontier[0] - self.current_frontier_goal_map[0], frontier[1] - self.current_frontier_goal_map[1]) < 0.15
					for frontier in self.completed_frontiers
				):
					self.completed_frontiers.append(self.current_frontier_goal_map)
					self.get_logger().info(f"Marked frontier as completed {self.current_frontier_goal_map}")
			self.get_logger().info(f"Got to goal {self.next_goal_index}, moving to next")
			if self.next_goal_index < len(self.goal_points):
				self._start_action_client()
			else:
				self.start_timer.reset()
		else:
			self.get_logger().info(f"Did not get to goal, skipping {self.next_goal_index}")
			if self.current_frontier_goal_map is not None:
				if not any(
					hypot(frontier[0] - self.current_frontier_goal_map[0], frontier[1] - self.current_frontier_goal_map[1]) < 0.2
					for frontier in self.failed_frontiers
				):
					self.failed_frontiers.append(self.current_frontier_goal_map)
					self.get_logger().info(f"Marked frontier as failed {self.current_frontier_goal_map}")
			self.get_logger().info(f"Did not get to goal {self.next_goal_index}. Replanning.")
			self.need_new_plan = True # Force a replan on the next map update

	def _feedback_callback(self, feedback):
		"""Every time driver loops in the action callback it send back the distance to the target as feedbackack
		@param feedback - data created by the action server - this has the distance in it (as a float)"""
		
		# Right now not doing anything but publishing the current distance
		
		self.last_distance = feedback.feedback.distance.data
		if (
			self.last_feedback_log_distance is None
			or self.last_feedback_log_distance - self.last_distance >= 0.08
			or self.last_distance < 0.12
		):
			self.get_logger().info(f'Feedback: Distance: {feedback.feedback.distance.data}')
			self.last_feedback_log_distance = self.last_distance

	def _cancel_response_callback(self, future : Future):
		""" This is a call and response to the server to check that it actually canceled the goal"""
		cancel_response = future.result()
		self.start_timer.reset()  # Increment to the next goal (if there is one)
		self.get_logger().info(f'Cancel request accepted by server: {cancel_response.return_code}')
		self._send_goal_future = None
		self._result_future = None
		self._cancel_future = None

	def skip_current_goal(self):
		""" Cancels the current goal and moves to the next (if any)
		GUIDE: Use this to skip over the current goal. Do NOT call repeatedly - it takes a while to process"""
		if not self._goal_handle:
			self.get_logger().info(f"No active goals to skip")
		elif self._cancel_future:
			self.get_logger().info(f"Already skipping goal, wait for this to finish before skipping next")
		else:
			self.get_logger().info(f"Skipping to next goal {self.next_goal_index} of {len(self.goal_points)}")
			self._cancel_future = self._goal_handle.cancel_goal_async()
			self._cancel_future.add_done_callback(self._cancel_response_callback)

	def completed_all_goals(self):
		""" Returns True if all of the goals have been completed
		GUIDE Use this to check if there are any goals left to do y/n"""
		return self.next_goal_index >= len(self.goal_points)
		
	def add_more_goal_points(self, goal_pts: list):
		""" Add more goal points; should be a list of tuples of x,y locations
		GUIDE: Use this if you just want to append more goals to the current list"""
		for pt in goal_pts:
			self.goal_points.append(pt)

		self._set_goal_markers()

		if self._result_future is None:
			self.start_timer.reset()
	
	def replace_goal_points(self, goal_pts: list, skip_current: bool):
		""" Replace the current list of goal points, and, optionally, skip the current
		@param goal_pts: a list of tuples of x,y locations
		@param skip_current: Will call skip-current for you after setting up new goals"""
		self.next_goal_index = 0

		# Just doing this to make sure the points you pass in are in the correct form
		self.goal_points = []
		for p in goal_pts:
			self.goal_points.append((p[0], p[1]))
		
		if skip_current:
			self.skip_current_goal()

		self._set_goal_markers()
		# This will kick start sending more goal points if it's stopped sending
		if self._result_future == None:
			self.start_timer.reset()   # Increment to the next goal

	def _set_goal_markers(self):
		""" Update the goal markers whenever the goals change"""
		if self.goal_markers == None:
			self.goal_markers = MarkerArray()

		# Lock while we make the Marker Array
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
			
			# Make the line(s) between the markers
			self.goal_markers.markers = []
			self.goal_markers.markers.append(line_marker)

			# Make the dots for the markers
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
				marker.pose.orientation.x = 0.0
				marker.pose.orientation.y = 0.0
				marker.pose.orientation.z = 0.0		
				marker.pose.orientation.w = 1.0
				marker.scale.x = 0.2
				marker.scale.y = 0.2
				marker.scale.z = 0.2
				marker.color.r = 0.0
				marker.color.g = 0.0
				marker.color.b = 1.0
				marker.color.a = 1.0

				self.goal_markers.markers.append(marker)

		# Actually publish the list
		self.goal_marker_pub.publish(self.goal_markers)

	def _set_path_markers(self, path_list, skip=5):
		"""Update the path markers. Assumes path_list is a list of tuple x,y locations
		@param path_list - param list of tuples with x,y locations in map coordinate frame
		@param skip draw ever nth one"""
		if self.path_markers == None:
			self.path_markers = MarkerArray()

		# Lock while we make the Marker Array
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
			
			# Make the line(s) between the markers
			self.path_markers.markers = []
			self.path_markers.markers.append(line_marker)

			# Make the dots for the markers
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
				marker.pose.orientation.x = 0.0
				marker.pose.orientation.y = 0.0
				marker.pose.orientation.z = 0.0		
				marker.pose.orientation.w = 1.0
				marker.scale.x = 0.2
				marker.scale.y = 0.2
				marker.scale.z = 0.2
				marker.color.r = 1.0
				marker.color.g = 1.0
				marker.color.b = 0.0
				marker.color.a = 1.0

				self.path_markers.markers.append(marker)
				
		# Actually publish the list
		self.path_marker_pub.publish(self.path_markers)

	def _set_reachable_markers(self, points):
		""" Put markers on the reachable points
		@param points - list of x,y tuples in map space"""

		if self.reachable_markers == None:
			self.reachable_markers = MarkerArray()

		# Lock while we make the Marker Array
		with self.mutex:
			self.reachable_markers.markers = []

			# Make the dots for the markers
			for indx, point in enumerate(points):
				marker = Marker()
				marker.header.frame_id = 'odom'
				marker.header.stamp = self.get_clock().now().to_msg()
				marker.id = 10000 + indx + 1
				marker.type = Marker.SPHERE
				marker.action = Marker.ADD
				marker.pose.position.x = point[0]
				marker.pose.position.y = point[1]
				marker.pose.position.z = 0.0
				marker.pose.orientation.x = 0.0
				marker.pose.orientation.y = 0.0
				marker.pose.orientation.z = 0.0		
				marker.pose.orientation.w = 1.0
				marker.scale.x = 0.05
				marker.scale.y = 0.05
				marker.scale.z = 0.05
				marker.color.r = 0.0
				marker.color.g = 1.0
				marker.color.b = 0.5
				marker.color.a = 1.0

				self.reachable_markers.markers.append(marker)
				
		# Actually publish the list
		self.reachable_marker_pub.publish(self.reachable_markers)

	def set_marker_points(self):
		"""Publishes the points in the list and links them up so they'll show up in RViz"""
		self._set_goal_markers()

	def from_map_to_image(self, map_msg : OccupancyGrid, pt_xy = (0.0, 0.0)):
		""" Convert from a point in the image to a point in the world
		@param map_msg - the map
		@param pt_xy - a tuple with an x,y in it
		@return pt_uv - point in the image"""
		info = map_msg.info

		im_u = 0
		im_v = 0

		# GUIDE: Subtract the origin position of the map and then divide by the resolution
		#   Don't forget to cast to an int
  # YOUR CODE HERE
		im_u = int((pt_xy[0] - info.origin.position.x) / info.resolution)
		im_v = int((pt_xy[1] - info.origin.position.y) / info.resolution)
		# self.get_logger().info(f"before {pt_xy} after {im_u}, {im_v}")
		return (im_u, im_v)
			
	def from_image_to_map(self, map_msg : OccupancyGrid, pt_uv = (0, 0)):
		""" Convert from a point in the world to a point in the image
		@param map_msg - the map
		@param pt_uv - a tuple with a u,v in width/height in it
		@return pt_xy - point in the world"""
		info = map_msg.info

		pt_x = 0.0
		pt_y = 0.0
		# GUIDE: Multiply by the resolution then add the origin position of the map 
  # YOUR CODE HERE
		# self.get_logger().info(f"before {pt_uv} after {pt_x}, {pt_y}")
		pt_x = pt_uv[0] * info.resolution + info.origin.position.x
		pt_y = pt_uv[1] * info.resolution + info.origin.position.y
		return (pt_x, pt_y)

	def find_nearest_free_cell(self, im_thresh, pt_uv, max_radius=3):
		"""Snap a point to a nearby free cell in image coordinates."""
		if is_free(im_thresh, pt_uv):
			return pt_uv

		best_pt = None
		best_dist_sq = None
		for radius in range(1, max_radius + 1):
			for du in range(-radius, radius + 1):
				for dv in range(-radius, radius + 1):
					cand = (pt_uv[0] + du, pt_uv[1] + dv)
					if not (0 <= cand[0] < im_thresh.shape[1] and 0 <= cand[1] < im_thresh.shape[0]):
						continue
					if not is_free(im_thresh, cand):
						continue

					dist_sq = du * du + dv * dv
					if best_dist_sq is None or dist_sq < best_dist_sq:
						best_pt = cand
						best_dist_sq = dist_sq

			if best_pt is not None:
				return best_pt

		return None

	def inflate_obstacles(self, im_thresh, inflation_radius=2):
		"""Create a conservative planning map that keeps the robot away from walls."""
		occupied_like = (im_thresh == 0) | (im_thresh == 128)
		structure = np.ones((2 * inflation_radius + 1, 2 * inflation_radius + 1), dtype=bool)
		inflated = ndimage.binary_dilation(occupied_like, structure=structure)

		planning_im = np.array(im_thresh, copy=True)
		planning_im[inflated] = 0
		planning_im[im_thresh == 128] = 128
		return planning_im

	def rank_frontier_candidates(self, im_thresh, possible_points, robot_loc):
		"""Rank frontier candidates using the same general policy as exploring.py."""
		candidates = []
		for pt in possible_points:
			for du in range(-1, 2):
				for dv in range(-1, 2):
					cand = (pt[0] + du, pt[1] + dv)
					if not is_free(im_thresh, cand):
						continue

					free_neighbors = 0
					unseen_neighbors = 0
					wall_neighbors = 0
					for nu in range(-1, 2):
						for nv in range(-1, 2):
							nbr = (cand[0] + nu, cand[1] + nv)
							if is_free(im_thresh, nbr):
								free_neighbors += 1
							elif 0 <= nbr[0] < im_thresh.shape[1] and 0 <= nbr[1] < im_thresh.shape[0] and im_thresh[nbr[1], nbr[0]] == 128:
								unseen_neighbors += 1
							else:
								wall_neighbors += 1

					if unseen_neighbors == 0 or free_neighbors < 2:
						continue

					dist_to_robot = hypot(cand[0] - robot_loc[0], cand[1] - robot_loc[1])
					score = 4.0 * unseen_neighbors + 0.35 * min(dist_to_robot, 20.0) - 2.5 * wall_neighbors + 0.5 * free_neighbors
					candidates.append((score, cand))

		candidates.sort(key=lambda item: item[0], reverse=True)

		ordered = []
		seen = set()
		for _, cand in candidates:
			if cand not in seen:
				ordered.append(cand)
				seen.add(cand)
		return ordered

	def map_callback(self, map_msg : OccupancyGrid):
		""" Called when the map gets updated. Size etc of the map is in the message"""
		self.get_logger().info(f"Got map size {(map_msg.info.width, map_msg.info.height)}, resolution {map_msg.info.resolution}")
		self.get_logger().info(f" Origin origin {map_msg.info.origin.position}")

	    # msg.data is a flat list of int8 values (-1 for unknown, 0 free, 100 occupied)
		im = np.array(map_msg.data, dtype=np.int8)

		# Reshape to (height, width)
		im = im.reshape((map_msg.info.height, map_msg.info.width))

		im_thresh = np.zeros(im.shape, dtype=np.uint8)

		# Threshold image
		im_thresh[im < 10] = 255    # Free
		im_thresh[im >= 100] = 0    # Wall
		im_thresh[im == -1] = 128   # Unknown
		im_plan = self.inflate_obstacles(im_thresh, inflation_radius=2)
		im_plan_loose = self.inflate_obstacles(im_thresh, inflation_radius=1)

		self.get_logger().info(f"N free {np.count_nonzero(im_thresh == 255)}, N walls {np.count_nonzero(im_thresh == 0)}, N {np.count_nonzero(im_thresh == 128)}")


		# Location of robot
		transform = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
		robot_current_loc_in_map = (transform.transform.translation.x, transform.transform.translation.y)
		robot_current_loc_in_image = self.from_map_to_image(map_msg=map_msg, pt_xy=robot_current_loc_in_map)
		robot_planning_loc_in_image = self.find_nearest_free_cell(im_plan, robot_current_loc_in_image, max_radius=5)
		if robot_planning_loc_in_image is None:
			robot_planning_loc_in_image = self.find_nearest_free_cell(im_plan_loose, robot_current_loc_in_image, max_radius=6)
		robot_planning_loc_raw = self.find_nearest_free_cell(im_thresh, robot_current_loc_in_image, max_radius=8)
		self.get_logger().info(f"Robot current location {robot_current_loc_in_map}")
		if robot_planning_loc_in_image is not None and robot_planning_loc_in_image != robot_current_loc_in_image:
			self.get_logger().info(f"Snapped robot planning cell from {robot_current_loc_in_image} to {robot_planning_loc_in_image}")

		# Condition 1: We finished our current list of goals
		if self.completed_all_goals():
			self.need_new_plan = True

		# Condition 2: Only invalidate a goal if one is actually in flight.
		goal_in_flight = self._send_goal_future is not None or self._result_future is not None
		if goal_in_flight and len(self.goal_points) > 0 and self.next_goal_index > 0:
			active_goal_index = min(self.next_goal_index - 1, len(self.goal_points) - 1)
			current_active_goal = self.goal_points[active_goal_index]
			current_active_goal_uv = self.from_map_to_image(map_msg=map_msg, pt_xy=current_active_goal)

			is_out_of_bounds = not (
				0 <= current_active_goal_uv[0] < im_thresh.shape[1] and
				0 <= current_active_goal_uv[1] < im_thresh.shape[0]
			)

			goal_still_reachable = False
			if not is_out_of_bounds:
				if is_free(im_thresh, current_active_goal_uv):
					goal_still_reachable = True
				elif self.find_nearest_free_cell(im_plan_loose, current_active_goal_uv, max_radius=3) is not None:
					goal_still_reachable = True
				elif self.find_nearest_free_cell(im_thresh, current_active_goal_uv, max_radius=4) is not None:
					goal_still_reachable = True

			if is_out_of_bounds or not goal_still_reachable:
				self.get_logger().info("Current goal is off map or no longer free! Replanning...")
				self.need_new_plan = True

		if not self.need_new_plan:
			return 
			
		# GUIDE: Change this to get just the points you might consider looking at and perhaps don't do it every time a map is made
		all_unseen_pts = find_all_possible_goals(im_thresh)  # Your exploring code

		# Filter out points that are outside the current image bounds
		all_unseen_pts = [
			p for p in all_unseen_pts
			if 0 <= p[0] < im_thresh.shape[1] and 0 <= p[1] < im_thresh.shape[0]
		]

		unvisited_unseen_pts = []
		completed_frontier_radius = 0.2
		failed_frontier_radius = 0.35
		for p in all_unseen_pts:
			map_xy = self.from_image_to_map(map_msg=map_msg, pt_uv=p)
			if (
				not any(
				hypot(map_xy[0] - frontier[0], map_xy[1] - frontier[1]) < completed_frontier_radius
				for frontier in self.completed_frontiers
				)
				and not any(
				hypot(map_xy[0] - frontier[0], map_xy[1] - frontier[1]) < failed_frontier_radius
				for frontier in self.failed_frontiers
				)
			):
				unvisited_unseen_pts.append(p)

		if unvisited_unseen_pts:
			all_unseen_pts = unvisited_unseen_pts

		if len(all_unseen_pts) == 0:
			self.get_logger().info("No more goal points / exploration complete")
			self.need_new_plan = False
			return

		# 2. Flag that we successfully processed our first map
		if not self.have_map:
			self.have_map = True

		# 3. Convert unseen image pixels to map coordinates for RViz
		reachable_pts = []
		for p in all_unseen_pts:
			map_xy = self.from_image_to_map(map_msg=map_msg, pt_uv=p)
			reachable_pts.append(map_xy)

		# This puts markers in RViz for all unseen points
		self._set_reachable_markers(reachable_pts)

		# Always choose a fresh frontier goal when replanning
		if robot_planning_loc_in_image is None:
			if robot_planning_loc_raw is None:
				self.get_logger().info("Could not find a nearby free cell for the robot, waiting for the next map update")
				self.need_new_plan = True
				return
			robot_planning_loc_in_image = robot_planning_loc_raw
			self.get_logger().info("Falling back to raw-map robot start because the inflated map is too tight")

		candidate_frontiers = self.rank_frontier_candidates(im_thresh, all_unseen_pts, robot_current_loc_in_image)
		if not candidate_frontiers:
			self.get_logger().info("No valid frontier goal found, waiting for the next map update")
			self.need_new_plan = True
			return

		selected_goal_loc_in_image = None
		selected_goal_planning_loc_in_image = None
		selected_planning_im = None
		selected_path = None

		for goal_loc_in_image in candidate_frontiers[:35]:
			planning_im = im_plan
			goal_planning_loc_in_image = self.find_nearest_free_cell(im_plan, goal_loc_in_image, max_radius=7)
			if goal_planning_loc_in_image is None:
				goal_planning_loc_in_image = self.find_nearest_free_cell(im_plan_loose, goal_loc_in_image, max_radius=8)
				if goal_planning_loc_in_image is not None:
					planning_im = im_plan_loose
			if goal_planning_loc_in_image is None:
				continue

			try:
				path = dijkstra(planning_im, robot_planning_loc_in_image, goal_planning_loc_in_image)
			except (IndexError, ValueError):
				continue

			if len(path) < 2:
				continue

			selected_goal_loc_in_image = goal_loc_in_image
			selected_goal_planning_loc_in_image = goal_planning_loc_in_image
			selected_planning_im = planning_im
			selected_path = path
			break

		if selected_goal_loc_in_image is None:
			raw_start = robot_planning_loc_raw if robot_planning_loc_raw is not None else robot_current_loc_in_image
			for goal_loc_in_image in candidate_frontiers[:25]:
				goal_planning_loc_in_image = self.find_nearest_free_cell(im_thresh, goal_loc_in_image, max_radius=14)
				if goal_planning_loc_in_image is None:
					continue

				try:
					path = dijkstra(im_thresh, raw_start, goal_planning_loc_in_image)
				except (IndexError, ValueError):
					continue

				if len(path) < 2:
					continue

				selected_goal_loc_in_image = goal_loc_in_image
				selected_goal_planning_loc_in_image = goal_planning_loc_in_image
				selected_planning_im = im_thresh
				selected_path = path
				self.get_logger().info("Falling back to raw map planning for a frontier candidate")
				break

		if selected_goal_loc_in_image is None:
			self.get_logger().info("Could not find a safe planning cell near any frontier, waiting for the next map update")
			self.need_new_plan = True
			return

		goal_loc_in_image = selected_goal_loc_in_image
		goal_planning_loc_in_image = selected_goal_planning_loc_in_image
		planning_im = selected_planning_im
		self.current_frontier_goal_map = self.from_image_to_map(map_msg=map_msg, pt_uv=goal_loc_in_image)
		self.get_logger().info(f"Selected frontier goal {goal_loc_in_image}, planning to {goal_planning_loc_in_image}")

		# GUIDE: This calls dijkstra with the goal location and plots the path that you return in RViz
		#  Note: If you did not fix your code to deal with an unreachable point then this will handle that case
		#   as an exception
		path_pts = []
		#bounds check
		def in_bounds(im, p):
			return 0 <= p[0] < im.shape[1] and 0 <= p[1] < im.shape[0]
		
		if not in_bounds(planning_im, robot_planning_loc_in_image):
			self.get_logger().info("Robot out of bounds, skipping this cycle")
			return
		if not in_bounds(planning_im, goal_planning_loc_in_image):
			self.get_logger().info("Goal out of bounds, replanning")
			self.need_new_plan = True
			return
		try:
			path = selected_path if selected_path is not None else dijkstra(planning_im, robot_planning_loc_in_image, goal_planning_loc_in_image)
			path_waypoints = find_waypoints(planning_im, path)
			self.get_logger().info(f"Planned path with {len(path)} cells and {len(path_waypoints)} waypoints")

			# skip only truly degenerate paths
			if len(path_waypoints) < 2:
				self.get_logger().info("Path too short, replanning")
				self.need_new_plan = True
				return
					
			# Keep only a few well-spaced waypoints so the robot commits to moving
			# through the hallway instead of stopping every few centimeters.
			sparse_stride = max(4, len(path_waypoints) // 3)
			sparse_waypoints = path_waypoints[::sparse_stride]
			if sparse_waypoints[-1] != path_waypoints[-1]:
				sparse_waypoints.append(path_waypoints[-1])

			raw_path_pts = []
			for p in sparse_waypoints:
				map_xy = self.from_image_to_map(map_msg=map_msg, pt_uv=p)
				raw_path_pts.append(map_xy)

			min_goal_spacing = 0.35
			filtered_path_pts = []
			last_anchor = robot_current_loc_in_map
			for pt in raw_path_pts:
				if hypot(pt[0] - last_anchor[0], pt[1] - last_anchor[1]) >= min_goal_spacing:
					filtered_path_pts.append(pt)
					last_anchor = pt

			if filtered_path_pts:
				path_pts = filtered_path_pts
			elif raw_path_pts:
				path_pts = [raw_path_pts[-1]]

			self._set_path_markers(path_pts, 1)
		except IndexError:
			self.get_logger().info("Robot or goal location not in image map, replanning")
			self.need_new_plan = True
			return
		except ValueError:
			if robot_planning_loc_in_image is not None and is_free(planning_im, robot_planning_loc_in_image):
				if is_free(planning_im, goal_planning_loc_in_image):
					self.get_logger().info(f"No valid path {robot_planning_loc_in_image} to {goal_planning_loc_in_image}")
				else:
					self.get_logger().info(f"Goal not free {robot_planning_loc_in_image} to {goal_planning_loc_in_image}")
			else:
				self.get_logger().info(f"Robot starting location not free {robot_current_loc_in_image}")
			self.need_new_plan = True
			return

		# GUIDE: This replaces the last goal if the robot has gone through the first two.
		# THIS IS AN EXAMPLE of how to replace goal points. You can also use skip_current_goal and add_more_goal_points

		if len(path_pts) > 0:
			self.get_logger().info(f"Replacing way points with new ones {path_pts}")
			self.replace_goal_points(path_pts, False)
			self.need_new_plan = False


# Unlike all the previous code, here we'll start up with a list of points to go to
def main(args=None):
	# Initialize rclpy.  We should do this every time.
	rclpy.init(args=args)

	# Create a list of points that will take the robot through the map
	points = []
	send_points = SendPoints(points)

	# Multi-threaded execution
	executor = MultiThreadedExecutor()
	executor.add_node(send_points)
	executor.spin()
	#rclpy.spin(send_points)

	# Make sure we shutdown everything cleanly.  This should happen, even if we don't
	# include this line, but you should do it anyway.
	rclpy.shutdown()
	

# If we run the node as a script, then we're going to start here.
if __name__ == '__main__':
	# The idiom in ROS2 is to set up a main() function and to call it from the entry
	# point of the script.
	main()
