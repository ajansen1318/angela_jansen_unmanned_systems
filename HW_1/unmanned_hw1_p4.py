# Unmanned HW 1 Problem 4

import matplotlib.pyplot as plt
import numpy as np


# get the numbers to put on the graph
def compute_index(min_x, min_y, max_x, max_y, gs, x_curr, y_curr):
    index = ((x_curr - min_x) / gs) + (
        ((y_curr - min_y) / gs) * (((max_x + gs) - min_x) / gs)
    )
    return index


# values
gs = 0.5
min_x = 0
min_y = 0
max_x = 10
max_y = 10
x_curr = 0
y_curr = 0

index = compute_index(min_x, min_y, max_x, max_y, gs, x_curr, y_curr)

x = []
y = []

x_array = np.arange(min_x, max_x + gs, gs)
y_array = np.arange(min_y, max_y + gs, gs)

plt.figure(figsize=(10, 10))
plt.xlim(min_x, max_x + gs)
plt.ylim(min_y, max_y + gs)

# put the values on the graph
for y_val in y_array:
    for x_val in x_array:
        something = compute_index(min_x, min_y, max_x, max_y, gs, x_val, y_val)
        plt.text(x_val, y_val, str(int(something)), color="red", fontsize=8)

plt.show()
