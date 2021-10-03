# encoding= utf=8

import numpy as np
from utils import *


def block_update(x, N, Map, visited, Maze, C, B, E, H, m, n):
    """
    when we meet block, update by latest information
    :param N: the number of neighbors cell x has.
    :param visited: Whether or not cell x has been visited.
    :param Maze: Whether or not cell x has been confirmed as empty or blocked, or is currently unconfirmed.
    :param C: the number of neighbors of x that are sensed to be blocked.
    :param B: the number of neighbors of x that have been confirmed to be blocked.
    :param E: the number of neighbors of x that have been confirmed to be empty.
    :param H: the number of neighbors of x that are still hidden or unconfirmed either way.
    :return:
    """
    Maze[x[0]][x[1]] = 1
    C[x[0]][x[1]] = sense(x, Map, C)
    E[x[0]][x[1]] = N[x[0]][x[1]] - C[x[0]][x[1]]
    # H[x[0]][x[1]] = updateH(x, Maze, C[x[0]][x[1]], H, m, n)
    for i, j in getAllNeighbors(x, m, n):
        B[i][j] += 1
    return
