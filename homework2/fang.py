# encoding= utf=8

import numpy as np
from utils import *


def block_update(x, N, visited, Maze, C, B, E, H, m, n):
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
    C[x[0]][x[1]] = sense(x, Maze, C)
    for nei in directions:
        i, j = (x[0] + nei[0], x[1] + nei[1])
        if isValid((i, j), m, n) and Maze[i][j] == 2:
            B[i][j] += 1
            E[i][j] -= 1
        if C[i][j] == B[i][j]:
            for
    return
