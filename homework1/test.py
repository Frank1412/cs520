# -*-coding=utf-8 -*-

import numpy as np
import math

map = np.zeros([10,10])

def euclidean(a, b):
    return math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]))

def manhattan(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


for i in range(10):
    for j in range(10):
        map[i][j] = round(euclidean([i, j], [9, 9])+manhattan([i, j], [0,0]), 2)

print(map)
