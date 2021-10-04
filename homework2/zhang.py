# encoding= utf=8

import numpy as np
from utils import *


def empty_update(x, N, Map, Maze, C, B, E, H, m, n):
    Maze[x[0]][x[1]] = 0
    for i, j in getAllNeighbors(x, m, n):
        E[i][j] += 1
        H[i][j] -= 1
    return
