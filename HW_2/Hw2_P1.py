# Unmanned Hw 2 P1

import math

def calc_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2-x1)**2+(y2-y1)**2)

    return distance     


# Test
x1 = 2
y1 = 1
x2 = 3
y2 = 2

euclidean_distance = calc_distance(x1, y1, x2, y2)
print("Your distance is ", euclidean_distance)