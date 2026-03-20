#!/usr/bin/env python3

import numpy as np
import heapq


def plot_with_path(im_threshhold, zoom=1.0, robot_loc=None, goal_loc=None, path=None):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 1)
    axs.imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs.set_title("threshold image")

    axs.plot(10, 5, 'xy', markersize=5)

    if robot_loc is not None:
        axs.plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
    if goal_loc is not None:
        axs.plot(goal_loc[0], goal_loc[1], '*g', markersize=10)
    if path is not None:
        for p, q in zip(path[0:-1], path[1:]):
            axs.plot([p[0], q[0]], [p[1], q[1]], '-y', markersize=2)
            axs.plot(p[0], p[1], '.y', markersize=2)
    axs.axis('equal')

    width = im_threshhold.shape[1]
    height = im_threshhold.shape[0]

    axs.set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
    axs.set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)


def is_wall(im, pix=(0, 0)):
    if not (0 <= pix[0] < im.shape[1] and 0 <= pix[1] < im.shape[0]):
        return False
    return im[pix[1], pix[0]] == 0


def is_unseen(im, pix=(0, 0)):
    if not (0 <= pix[0] < im.shape[1] and 0 <= pix[1] < im.shape[0]):
        return False
    return im[pix[1], pix[0]] == 128


def is_free(im, pix=(0, 0)):
    if not (0 <= pix[0] < im.shape[1] and 0 <= pix[1] < im.shape[0]):
        return False
    return im[pix[1], pix[0]] == 255


def convert_image(im, wall_threshold, free_threshold):
    im_ret = np.zeros((im.shape[0], im.shape[1]), dtype='uint8') + 128

    im_avg = im
    if len(im.shape) == 3:
        im_avg = np.mean(im, axis=2)

    im_avg = im_avg / np.max(im_avg)
    im_ret[im_avg < wall_threshold] = 0
    im_ret[im_avg > free_threshold] = 255
    return im_ret


def four_connected(pix=(0, 0)):
    for indx in [-1, 1]:
        yield pix[0] + indx, pix[1]
    for indx in [-1, 1]:
        yield pix[0], pix[1] + indx


def eight_connected(pix=(0, 0)):
    for indx in range(-1, 2):
        for j in range(-1, 2):
            if indx == 0 and j == 0:
                continue
            yield pix[0] + indx, pix[1] + j


def dijkstra(im, robot_loc=(0, 0), goal_loc=(0, 0)):
    if not (0 <= robot_loc[0] < im.shape[1] and 0 <= robot_loc[1] < im.shape[0]):
        raise IndexError(f"ERROR: Robot location {robot_loc} is not in map {im.shape}")
    if not (0 <= goal_loc[0] < im.shape[1] and 0 <= goal_loc[1] < im.shape[0]):
        raise IndexError(f"ERROR: Goal location {goal_loc} is not in map {im.shape}")

    if not is_free(im, robot_loc):
        raise ValueError(f"ERROR: Start location {robot_loc} is not in the free space of the map")
    if not is_free(im, goal_loc):
        raise ValueError(f"ERROR: Goal location {goal_loc} is not in the free space of the map")

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    priority_queue = []
    heapq.heappush(priority_queue, (heuristic(robot_loc, goal_loc), robot_loc))

    visited = {}
    visited[robot_loc] = (0.0, None, False)

    closest_node = robot_loc
    closest_distance_to_goal = heuristic(robot_loc, goal_loc)

    while priority_queue:
        current_score, current_node = heapq.heappop(priority_queue)
        current_dist, current_parent, current_closed = visited[current_node]

        if current_closed:
            continue

        if current_node == goal_loc:
            break

        visited[current_node] = (current_dist, current_parent, True)

        dist_to_goal = heuristic(current_node, goal_loc)
        if dist_to_goal < closest_distance_to_goal:
            closest_distance_to_goal = dist_to_goal
            closest_node = current_node

        for neighbor in eight_connected(current_node):
            if not (0 <= neighbor[0] < im.shape[1] and 0 <= neighbor[1] < im.shape[0]):
                continue
            if not is_free(im, neighbor):
                continue

            step_dist = np.hypot(neighbor[0] - current_node[0], neighbor[1] - current_node[1])
            new_dist = current_dist + step_dist

            if neighbor not in visited:
                visited[neighbor] = (new_dist, current_node, False)
                heapq.heappush(priority_queue, (new_dist + heuristic(neighbor, goal_loc), neighbor))
            else:
                old_dist, old_parent, old_closed = visited[neighbor]
                if not old_closed and new_dist < old_dist:
                    visited[neighbor] = (new_dist, current_node, False)
                    heapq.heappush(priority_queue, (new_dist + heuristic(neighbor, goal_loc), neighbor))

    if goal_loc not in visited:
        goal_loc = closest_node

    path = []
    current_trace = goal_loc
    while current_trace is not None:
        path.append(current_trace)
        current_trace = visited[current_trace][1]

    path.reverse()
    return path


def open_image(im_name):
    import imageio.v2 as imageio
    import yaml as yaml
    import os

    fnames = [
        "Data/" + im_name,
        "Assignments/Data/" + im_name,
        "Skills/Data/" + im_name,
        "../../../../Skills/Data/" + im_name,
        "../../../../Assignments/Data" + im_name,
    ]

    im = None
    print(f"{os.getcwd()}")
    for fname in fnames:
        if os.path.exists(fname):
            im = imageio.imread(fname)

    wall_threshold = 0.7
    free_threshold = 0.9
    try:
        yaml_name = "Data/" + im_name[0:-3] + "yaml"
        with open(yaml_name, "r") as f:
            dict = yaml.load_all(f)
            wall_threshold = dict["occupied_thresh"]
            free_threshold = dict["free_thresh"]
    except:
        pass

    im_thresh = convert_image(im, wall_threshold, free_threshold)
    return im, im_thresh


def check_path_continuous(im, path, expected_len_four, expected_len_eight):
    b_is_eight = False
    pass_connected_test = True
    for p1, p2 in zip(path[0:-1], path[1:]):
        if abs(p1[0] - p2[0]) > 1:
            pass_connected_test = False
        if abs(p1[1] - p2[1]) > 1:
            pass_connected_test = False
        if abs(p1[0] - p2[0]) > 0 and abs(p1[1] - p2[1]):
            b_is_eight = True

    pass_len_test = False
    expected_len = expected_len_eight if b_is_eight else expected_len_four
    if abs(len(path) - expected_len) < 3:
        pass_len_test = True

    pass_free_test = True
    for pt in path:
        if not is_free(im, pt):
            pass_free_test = False

    if not pass_connected_test:
        print("Failed connected test")
        return False
    if not pass_free_test:
        print("Failed all path points must be free test")
        return False
    if not pass_len_test:
        print("Failed length test")
        if b_is_eight:
            print(" Assumed 8 connected")
        else:
            print(" Assumed 4 connected")
        return False
    return True


if __name__ == '__main__':
    robot_start_loc = (40, 60)
    robot_goal_loc_close = (60, 80)
    robot_goal_hallway = (80, 175)
    robot_goal_next_room = (130, 50)
    loc_not_reachable = (115, 145)

    _, im_thresh = open_image("map.pgm")
    zoom = 1.0

    robot_goal_loc = robot_goal_loc_close
    path = dijkstra(im_thresh, robot_start_loc, robot_goal_loc)
    plot_with_path(im_thresh, zoom=zoom, robot_loc=robot_start_loc, goal_loc=robot_goal_loc, path=path)

    path_straight = dijkstra(im_thresh, robot_loc=(40, 60), goal_loc=robot_goal_loc_close)
    assert check_path_continuous(im_thresh, path_straight, 41, 21)

    path_hallway = dijkstra(im_thresh, robot_loc=(40, 60), goal_loc=robot_goal_hallway)
    assert check_path_continuous(im_thresh, path_hallway, 156, 116)

    path_next_room = dijkstra(im_thresh, robot_loc=(40, 60), goal_loc=robot_goal_next_room)
    assert check_path_continuous(im_thresh, path_next_room, 315, 241)

    path_not_reachable = dijkstra(im_thresh, robot_loc=(40, 60), goal_loc=loc_not_reachable)
    assert len(path_not_reachable) > 20

    import matplotlib.pyplot as plt
    plt.show()

    print("Done")
