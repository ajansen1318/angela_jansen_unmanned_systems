from pathfinding_library.dijkstra_astar import DijkstraAstar
from pathfinding_library.Grid import Grid
from pathfinding_library.node import Node
from pathfinding_library.obstacle import Obstacle

import matplotlib.pyplot as plt

if __name__ == "__main__":
    obstacle_radius = 0.25
    obstacles = [
        Obstacle(1, 1, obstacle_radius),
        Obstacle(4, 4, obstacle_radius),
        Obstacle(3, 4, obstacle_radius),
        Obstacle(5, 0, obstacle_radius),
        Obstacle(5, 1, obstacle_radius),
        Obstacle(0, 7, obstacle_radius),
        Obstacle(1, 7, obstacle_radius),
        Obstacle(2, 7, obstacle_radius),
        Obstacle(3, 7, obstacle_radius),
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

    astar = DijkstraAstar(
        grid=grid, start_node=start_node, end_node=end_node, use_dijkstra=False
    )
    path = astar.find_path()
    grid.plot(path)
