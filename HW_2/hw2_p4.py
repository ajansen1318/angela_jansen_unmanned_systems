# Hw 2 P3

import math as m
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x: float, y: float, cost: float, parent_index: int) -> None:
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index


class Obstacle:
    def __init__(self, x_pos: float, y_pos: float, radius: float) -> None:
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = radius

    def is_inside(
        self, curr_x: float, curr_y: float, radius: float = 0, robot_radius: float = 0
    ) -> bool:
        dist_from = np.sqrt((curr_x - self.x_pos) ** 2 + (curr_y - self.y_pos) ** 2)
        if dist_from >= self.radius + robot_radius:
            return False

        return True


def compute_index(
    min_x: int, min_y: int, max_x: int, max_y: int, gs: float, curr_x: int, curr_y: int
) -> int:
    index = int(((max_x / gs) + 1) * (curr_y / gs) + (curr_x / gs))  # row len * y + x

    return index


def get_all_moves(current_x: float, current_y: float, gs: float) -> list[tuple]:
    gs_x_bounds = np.arange(-gs, gs + gs, gs)
    gs_y_bounds = np.arange(-gs, gs + gs, gs)
    move_list = []

    for dx in gs_x_bounds:
        for dy in gs_y_bounds:
            x_next = current_x + dx
            y_next = current_y + dy

            if [x_next, y_next] == [current_x, current_y]:
                continue

            move = (x_next, y_next)
            move_list.append(move)

    return move_list


def is_not_valid(
    x_curr: float,
    y_curr: float,
    obs_list: list[Obstacle],
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    robot_radius: float = 0.0,
):
    # Check if near or inside obstacle
    for obs in obs_list:
        if obs.is_inside(x_curr, y_curr, obstacle.radius, robot_radius):
            return True

        # Check if outside boundary
        if x_min >= x_curr:
            return True
        if x_max <= x_curr:
            return True
        if y_min >= y_curr:
            return True
        if y_max <= y_curr:
            return True

    return False


def add_to_unvisited(
    unvisited_nodes: dict[int, Node],
    current_node: Node,
    robot_radius: float,
    obs_list: list[Obstacle],
):
    # check if move is valid
    all_moves = get_all_moves(current_node.x, current_node.y, gs)
    filtered_moves = []

    for move in all_moves:
        if is_not_valid(
            move[0], move[1], obs_list, min_x, min_y, max_x, max_y, robot_radius
        ):
            continue
        filtered_moves.append(move)

    for move in filtered_moves:
        new_index = compute_index(min_x, min_y, max_x, max_y, gs, move[0], move[1])
        # cost = parent cost + next distance
        new_cost = current_node.cost + m.dist(move, [current_node.x, current_node.y])

        if (move[0], move[1]) == (goal_x, goal_y):
            pass

        if new_index in unvisited_nodes:
            if new_cost < unvisited_nodes[new_index].cost:
                # update the cost value
                unvisited_nodes[new_index].cost = new_cost
                unvisited_nodes[new_index].parent_index = current_index
            continue

        elif new_index not in visited_nodes:
            new_node = Node(move[0], move[1], new_cost, current_index)
            unvisited_nodes[new_index] = new_node

    return unvisited_nodes


# initialize some params
start_x = 0.5
start_y = 2
min_x = 0
max_x = 10
min_y = 0
max_y = 10
gs = 0.5
goal_x = 8
goal_y = 9

obstacle_positions = [
    (1, 1),
    (4, 4),
    (3, 4),
    (5, 0),
    (5, 1),
    (0, 7),
    (1, 7),
    (2, 7),
    (3, 7),
]
obstacle_list: list[Obstacle] = []  # store obstacle classes
obstacle_radius = 0.25
robot_radius = 0.5

for obs_pos in obstacle_positions:
    # store obstacle info in obstacle list
    obstacle = Obstacle(obs_pos[0], obs_pos[1], obstacle_radius)
    obstacle_list.append(obstacle)


# start dictionaries
unvisited_nodes: dict[int, Node] = {}
visited_nodes: dict[int, Node] = {}

# initialize current_node
current_node = Node(start_x, start_y, 0, int(-1))

# initialize current_index using compute_index()
current_index = compute_index(min_x, min_y, max_x, max_y, gs, start_x, start_y)

# put current node in dictionary - use current_index as the key
unvisited_nodes[current_index] = current_node

# set current index to what the minimum in unvisted
# while [current_node.x, current_node.y] != [goal_x, goal_y]:
while unvisited_nodes:
    current_index = min(unvisited_nodes, key=lambda x: unvisited_nodes[x].cost)
    current_node = unvisited_nodes[current_index]
    visited_nodes[current_index] = current_node

    del unvisited_nodes[current_index]

    unvisited_nodes = add_to_unvisited(
        unvisited_nodes, current_node, robot_radius, obstacle_list
    )

print("The search is over")

goal_node_index = compute_index(min_x, min_y, max_x, max_y, gs, goal_x, goal_y)
wp_node = visited_nodes[goal_node_index]
wp_list = []
wp_list.append([wp_node.x, wp_node.y])

while wp_node.parent_index != -1:
    next_index = wp_node.parent_index
    wp_node = visited_nodes[next_index]
    wp_list.append([wp_node.x, wp_node.y])

# plot things
plt.figure(figsize=(7, 7))
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)

plt.plot(start_x, start_y, "go")
plt.plot(goal_x, goal_y, "ro")

# plot obstacles
for obstacle in obstacle_list:
    circle = plt.Circle(
        (obstacle.x_pos, obstacle.y_pos), obstacle.radius, color="r", fill=True
    )
    plt.gcf().gca().add_artist(circle)

# plot path
x_array = [wp[0] for wp in wp_list]
y_array = [wp[1] for wp in wp_list]
plt.plot(x_array, y_array)


plt.show()
