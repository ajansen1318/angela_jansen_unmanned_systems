# Unmanned HW 1 P5
import math


def distance_calc(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    distance = math.sqrt((x2 - x1) ^ 2 - (y2 - y1) ^ 2)
    return distance


# test
point_1 = (2, 1)
point_2 = (3, 2)

euclidean_distance = distance_calc(point_1, point_2)

print(euclidean_distance)
