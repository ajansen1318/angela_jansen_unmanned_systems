# import things
import time

from pathfinding_library.rrt import RRT
from pathfinding_library.Grid import Grid
from pathfinding_library.node import Node
from pathfinding_library.obstacle import Obstacle
from pathfinding_library.dijkstra_astar import DijkstraAstar

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # parameters
    start_time = time.time()
    obstacle_radius = 0.25
    start_x = 1
    start_y = 1
    goal_x = 7
    goal_y = 13
    obstacle_x = [
        2,
        2,
        2,
        2,
        0,
        1,
        2,
        3,
        4,
        5,
        5,
        5,
        5,
        5,
        8,
        9,
        10,
        11,
        12,
        13,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        2,
        2,
        2,
        2,
        2,
        2,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        12,
        12,
        12,
        12,
    ]
    obstacle_y = [
        2,
        3,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        2,
        3,
        4,
        5,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        7,
        7,
        7,
        7,
        7,
        7,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        12,
        12,
        12,
        12,
        12,
        12,
        8,
        9,
        10,
        11,
        12,
    ]
    max_x = 15
    max_y = 15

    obstacle_list = [
        Obstacle(x, y, obstacle_radius) for x, y in zip(obstacle_x, obstacle_y)
    ]
    grid = Grid(
        min_x=0,
        min_y=0,
        max_x=max_x,
        max_y=max_y,
        grid_space=0.5,
        obstacles=obstacle_list,
        robot_radius=0.5,
    )
    start_node = Node(start_x, start_y)
    end_node = Node(goal_x, goal_y)
    dijkstra = DijkstraAstar(
        grid=grid, start_node=start_node, end_node=end_node, use_dijkstra=True
    )
    path = dijkstra.find_path()
    # rrt = RRT(grid, start_node, end_node, step_length=0.5)
    # path, cost = rrt.find_path()
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(elapsed_time, cost)
    # grid.plot(path)
