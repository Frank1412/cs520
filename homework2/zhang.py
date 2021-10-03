# encoding= utf=8

import numpy as np
from utils import *


def empty_update(x, N, Map, Maze, C, B, E, H, m, n):
    Maze[x[0]][x[1]] = 0
    for i, j in getAllNeighbors(x, m, n):
        E[i][j] += 1
        H[i][j] -= 1 
    C[x[0]][x[1]] = sense(x, Map, C)
    if C[x[0]][x[1]] == B[x[0]][x[1]]:#if all blocks are confirmed
        E[x[0]][x[1]] = N[x[0]][x[1]] - C[x[0]][x[1]]#then all empty cells can be confirmed
        H[x[0]][x[1]] = 0#no more hidden cells
    elif E[x[0]][x[1]] == N[x[0]][x[1]] - C[x[0]][x[1]]:#if all empty cells are confirmed
        B[x[0]][x[1]] = C[x[0]][x[1]]#then all the blocks can be confirmed
        H[x[0]][x[1]] = 0

    return