from itertools import permutations
import math as m
import matplotlib.pyplot as plt
from time import perf_counter
from pathfinding_library.Grid import Grid
from pathfinding_library.obstacle import Obstacle
from pathfinding_library.dijkstra_astar import DijkstraAstar
from pathfinding_library.node import Node


# parameters
start_x = 0
start_y = 0
start_node = Node(start_x, start_y)
waypoint_list = (
    (1, 1),
    (9, 7),
    (1, 9),
    (4, 4),
    (9, 4),
    (6, 14),
    (3, 11),
    (14, 1),
    (1, 14),
    (14, 14),
    (7, 10),
)
waypoint_nodes = [Node(x, y) for x, y in waypoint_list]
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
    9,
    10,
    11,
    12,
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


obstacle_radius = 0.25
obstacle_list = [
    Obstacle(x, y, obstacle_radius) for x, y in zip(obstacle_x, obstacle_y)
]
grid = Grid(
    min_x=0,
    min_y=0,
    max_x=15,
    max_y=15,
    grid_space=0.5,
    obstacles=obstacle_list,
    robot_radius=0.5,
)
permutation_list = list(permutations(waypoint_nodes))
min_cost = float("inf")
path = None

compute_time = 0
cost_dictionary = {}
total_cost = []

# Compute costs between waypoints before A*
start_time = perf_counter()
for waypoint_1 in waypoint_nodes:
    for waypoint_2 in waypoint_nodes:
        if not (
            waypoint_1 != waypoint_2
            and (waypoint_1, waypoint_2) not in cost_dictionary
            and (waypoint_2, waypoint_1) not in cost_dictionary
        ):
            continue

        compute_time += 1
        astar = DijkstraAstar(grid, waypoint_1, waypoint_2, False)
        astar_path = astar.find_path()
        total_distance = astar_path[0].cost
        cost_dictionary[waypoint_1, waypoint_2] = {
            "cost": total_distance,
            "path": astar_path,
        }
        cost_dictionary[waypoint_2, waypoint_1] = {
            "cost": total_distance,
            "path": astar_path,
        }

for points in permutation_list:
    if points[0] != waypoint_nodes[0]:
        total_cost.append(float("inf"))
        continue

    cost = 0
    for i in range(len(points) - 1):
        waypoint_1 = points[i]
        waypoint_2 = points[i + 1]
        cost += cost_dictionary[(waypoint_1, waypoint_2)]["cost"]
    total_cost.append(cost)


min_total_cost = min(total_cost)
min_total_cost_index = total_cost.index(min_total_cost)
best_path = permutation_list[min_total_cost_index]
print(min_total_cost, (perf_counter() - start_time))

# Plot the path
wp_list: list[Node] = []
for i in range(len(best_path) - 1):
    path = cost_dictionary[(best_path[i], best_path[i + 1])]["path"]
    x_array = [wp.x for wp in path]
    y_array = [wp.y for wp in path]
    plt.plot(x_array, y_array, "g-")
    wp_list.extend(path)

    plt.plot(best_path[i].x, best_path[i].y, "bo")
    plt.text(best_path[i].x, best_path[i].y, str(i + 1))
plt.plot(best_path[-1].x, best_path[-1].y, "bo")
plt.text(best_path[-1].x, best_path[-1].y, str(len(best_path)))

grid.plot([(wp.x, wp.y) for wp in wp_list])
plt.show()
