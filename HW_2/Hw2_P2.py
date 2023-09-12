# Hw 2 p2

import numpy as np
from matplotlib import pyplot as plt


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
            # print("You're dead at", obs.x_pos, obs.y_pos)
            return True

        # Check if outside boundary
        if x_min > x_curr:
            return True
        if x_max < x_curr:
            return True
        if y_min > y_curr:
            return True
        if y_max < y_curr:
            return True

    return False


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
# obstacle_radius = 0.5  ### for testing

robot_radius = 0.0

for obs_pos in obstacle_positions:
    # store obstacle info in obstacle list
    obstacle = Obstacle(obs_pos[0], obs_pos[1], obstacle_radius)
    obstacle_list.append(obstacle)

# test
x_curr = 2
y_curr = 2
x_min = 0
y_min = 0
x_max = 10
y_max = 10

is_node_valid = not is_not_valid(
    x_curr, y_curr, obstacle_list, x_min, y_min, x_max, y_max, robot_radius
)

plt.figure(figsize=(7, 7))
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# plot obstacles
for obstacle in obstacle_list:
    circle = plt.Circle(
        (obstacle.x_pos, obstacle.y_pos), obstacle.radius, color="r", fill=True
    )
    plt.gcf().gca().add_artist(circle)

plt.text(x_curr, y_curr, s=f"{x_curr},{x_curr}\nvalid", ha="center", va="center")

plt.show()
