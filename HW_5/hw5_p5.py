# problem 5 - TSP

from itertools import permutations
import math as m
import matplotlib.pyplot as plt

# parameters
start_location = (0, 0)
waypoint_list = [(2, 2), (5, 3), (3, 4), (6, 4)]
permutation_list = permutations(waypoint_list)
min_cost = float("inf")
path = None


for points in permutation_list:
    cost = 0
    for i in range(len(points) - 1):
        wp_curr = points[i]
        wp_next = points[i + 1]
        cost += m.dist(wp_curr, wp_next)

    cost += m.dist(points[-1], start_location)
    if cost < min_cost:
        min_cost = cost
        path = points

    print(path, min_cost)
    x_coords = [point[0] for point in path]
    y_coords = [point[1] for point in path]

# Plot the path
plt.figure()
plt.plot(x_coords, y_coords, marker="o", linestyle="-", color="b")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Ideal Path for TSP")
plt.grid(True)
plt.show()
