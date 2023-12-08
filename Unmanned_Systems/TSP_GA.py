import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import math as m
import pylab as pl  # Needed for plotting numbers on plots
from pathfinding_library.dijkstra_astar import DijkstraAstar
from pathfinding_library.node import Node
from pathfinding_library.Grid import Grid
from pathfinding_library.obstacle import Obstacle


np.random.seed(42)


class Path_store:
    def __init__(self, x, y, cost):
        self.x = x
        self.y = y
        self.cost = cost


goal_wp_x = [0, 9, 4, 1, 9, 6]  # 5 desired waypoints - now 6
goal_wp_y = [0, 4, 4, 9, 7, 14]
goal_wp_nodes = [Node(x, y) for x, y in zip(goal_wp_x, goal_wp_y)]
cities = [1, 2, 3, 4, 5]  # added one

start_x = 0  # Starting point -> changed
start_y = 0
start_node = Node(start_x, start_y)

# use simple list to start
# obstacle_x = [1, 2, 3, 4, 6, 7, 7]
# obstacle_y = [7, 7, 7, 7, 2, 2, 3]

# ox = [2, 3, 4, 4, 4, 4, 10, 9, 8, 7, 6, 6]
# oy = [7, 7, 7, 8, 9, 10, 5, 5, 5, 5, 5, 4]

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


grid_size = 0.5
max_x = 15
max_y = 15
min_x = 0
min_y = 0
obstacle_radius = 0.25
robot_radius = 0.5

# Plotting
plt.plot(obstacle_x, obstacle_y, ".b")
plt.plot(start_x, start_y, "xg")
plt.plot(goal_wp_x, goal_wp_y, "xr")
plt.axis([min_x, max_x, min_y, max_y])
# plt.show()

# Initialize a cost matrix (distance cost to/from each waypoint)
cost_matrix = np.zeros([len(goal_wp_x), len(goal_wp_x)])
path_matrix = dict()

obstacle_list = [
    Obstacle(x, y, obstacle_radius) for x, y in zip(obstacle_x, obstacle_y)
]

grid = Grid(min_x, min_y, max_x, max_y, grid_size, obstacle_list, robot_radius)


t = perf_counter()
for i, start in enumerate(goal_wp_nodes):
    for j, goal in enumerate(goal_wp_nodes):
        if i != j:
            astar = DijkstraAstar(grid, start, goal, False)
            astar_path = astar.find_path()
            temp_cost = astar_path[0].cost
            cost_matrix[i, j] = temp_cost
            path_matrix[i, j] = Path_store(
                [n.x for n in astar_path], [n.y for n in astar_path], temp_cost
            )


adjacency_mat = cost_matrix


# %%
class Population:
    def __init__(self, bag, adjacency_mat):
        self.bag = bag
        self.parents = []
        self.score = 0
        self.best = None
        self.adjacency_mat = adjacency_mat


def init_population(cities, adjacency_mat, n_population):
    return Population(
        np.asarray([np.random.permutation(cities) for _ in range(n_population)]),
        adjacency_mat,
    )


pop = init_population(cities, adjacency_mat, 50)


def fitness(self, chromosome):
    return (
        sum(
            [
                self.adjacency_mat[chromosome[i], chromosome[i + 1]]
                for i in range(len(chromosome) - 1)
            ]
        )
        + self.adjacency_mat[0, chromosome[0]]
    )  # because we have to start at first node every time


Population.fitness = fitness


def evaluate(self):
    distances = np.asarray([self.fitness(chromosome) for chromosome in self.bag])
    self.score = np.min(distances)
    self.best = self.bag[distances.tolist().index(self.score)]
    self.parents.append(self.best)
    if False in (distances[0] == distances):
        distances = np.max(distances) - distances
    return distances / np.sum(distances)


Population.evaluate = evaluate


def select(self, k=4):
    fit = self.evaluate()
    while len(self.parents) < k:
        idx = np.random.randint(0, len(fit))
        if fit[idx] > np.random.rand():
            self.parents.append(self.bag[idx])
    self.parents = np.asarray(self.parents)


Population.select = select


def swap(chromosome):
    a, b = np.random.choice(len(chromosome), 2)
    chromosome[a], chromosome[b] = (
        chromosome[b],
        chromosome[a],
    )
    return chromosome


def crossover(self, p_cross=0.1):
    children = []
    count, size = self.parents.shape
    for _ in range(len(self.bag)):
        if np.random.rand() > p_cross:
            children.append(list(self.parents[np.random.randint(count, size=1)[0]]))
        else:
            parent1, parent2 = self.parents[np.random.randint(count, size=2), :]
            idx = np.random.choice(range(size), size=2, replace=False)
            start, end = min(idx), max(idx)
            child = [None] * size
            for i in range(start, end + 1, 1):
                child[i] = parent1[i]
            pointer = 0
            for i in range(size):
                if child[i] is None:
                    while parent2[pointer] in child:
                        pointer += 1
                    child[i] = parent2[pointer]
            children.append(child)
    return children


Population.crossover = crossover


def mutate(self, p_cross=0.1, p_mut=0.1):
    next_bag = []
    children = self.crossover(p_cross)
    for child in children:
        if np.random.rand() < p_mut:
            next_bag.append(swap(child))
        else:
            next_bag.append(child)
    return next_bag


Population.mutate = mutate


def genetic_algorithm(
    cities,
    adjacency_mat,
    n_population=500,
    n_iter=2000,
    selectivity=0.75,
    p_cross=0.5,
    p_mut=0.3,
    print_interval=100,
    return_history=False,
    verbose=False,
):
    pop = init_population(cities, adjacency_mat, n_population)
    best = pop.best
    score = float("inf")
    history = []
    for i in range(n_iter):
        print(i)
        pop.select(n_population * selectivity)
        history.append(pop.score)
        if verbose:
            print(f"Generation {i}: {pop.score}")
        elif i % print_interval == 0:
            print(f"Generation {i}: {pop.score}")
        if pop.score < score:
            best = pop.best
            score = pop.score
        children = pop.mutate(p_cross, p_mut)
        pop = Population(children, pop.adjacency_mat)
    if return_history:
        return best, history
    return best


# returns the best order of waypoints to visit
best = genetic_algorithm(cities, adjacency_mat, verbose=False)
# add the starting point to the list of waypoints
best = np.insert(best, 0, 0)

# best order of waypoints to visit
wp_order = []
for i in range(0, len(best)):
    wp_order.append([goal_wp_x[best[i]], goal_wp_y[best[i]]])

print("best order of waypoints to visit", wp_order)
print(perf_counter() - t)

# Plotting
plt.plot(obstacle_x, obstacle_y, ".b")
plt.plot(start_x, start_y, "xg")
plt.plot(goal_wp_x, goal_wp_y, "xr")
plt.axis([min_x - grid_size, max_x + grid_size, min_y - grid_size, max_y + grid_size])

for i in range(1, len(best)):
    plt.plot(
        path_matrix[best[i - 1], best[i]].x, path_matrix[best[i - 1], best[i]].y, "r-"
    )
# plt.plot(pathx, pathy, "-r")
plt.xlabel("X Distance")
plt.ylabel("Y Distance")
plt.show()
