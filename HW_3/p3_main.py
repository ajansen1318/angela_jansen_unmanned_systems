from pathfinding_library.rrt import RRT
from pathfinding_library.Grid import Grid
from pathfinding_library.node import Node
from pathfinding_library.obstacle import Obstacle

import matplotlib.pyplot as plt


if __name__ == "__main__":
    obstacle_radius = 0.25
    obstacles = [
        Obstacle(2, 2, obstacle_radius),
        Obstacle(2, 3, obstacle_radius),
        Obstacle(2, 4, obstacle_radius),
        Obstacle(5, 5, obstacle_radius),
        Obstacle(5, 6, obstacle_radius),
        Obstacle(6, 6, obstacle_radius),
        Obstacle(7, 3, obstacle_radius),
        Obstacle(7, 4, obstacle_radius),
        Obstacle(7, 5, obstacle_radius),
        Obstacle(7, 6, obstacle_radius),
        Obstacle(8, 6, obstacle_radius),
    ]

    grid = Grid(
        min_x=0,
        min_y=0,
        max_x=10,
        max_y=10,
        grid_space=0.5,
        obstacles=obstacles,
        robot_radius=0.5,
    )
    start_node = Node(0, 0)
    end_node = Node(9, 8)
    step_length = 0.5

    rrt = RRT(
        grid=grid, start_node=start_node, end_node=end_node, step_length=step_length
    )
    path = rrt.find_path()
    grid.plot(path)
