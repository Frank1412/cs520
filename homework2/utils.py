# encoding=utf-8

import numpy as np
import copy

directions = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]


def sense(x, Map, m, n):
    blocks = 0
    for (i, j) in getAllNeighbors(x, m, n):
        if Map[i][j] == 1:
            blocks += 1
    return blocks


def countB(x, Maze, m, n, target):
    nums = 0
    for (i, j) in getAllNeighbors(x, m, n):
        if Maze[i][j] == target:
            nums += 1
    return nums


def isVertex(x, m, n):
    return (x[0] == 0 and x[1] == 0) or (x[0] == 0 and x[1] == n - 1) or (x[0] == m - 1 and x[1] == 0) or (x[0] == m - 1 and x[1] == n - 1)


def isBorder(x, m, n):
    return x[0] == 0 or x[0] == m - 1 or x[1] == 0 or x[1] == n - 1


def isValid(x, m, n):
    return 0 <= x[0] < m and 0 <= x[1] < n


def initialize(map):
    m, n = len(map), len(map[0])
    Maze = np.full([m, n], 2)
    N = np.full((m, n), 8)
    C = np.zeros([m, n])
    B = np.zeros([m, n])
    E = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            C[i][j] = sense((i, j), map, m, n)
            if isVertex((i, j), m, n):
                N[i][j] = 3
            elif isBorder((i, j), m, n):
                N[i][j] = 5
    H = copy.deepcopy(N)
    # H = np.full([m, n], -1)
    return Maze, N, C, B, E, H


def getAllNeighbors(x, m, n):
    neighbors = []
    for dir in directions:
        x1 = x[0] + dir[0]
        y1 = x[1] + dir[1]
        if 0 <= x1 < m and 0 <= y1 < n:
            neighbors.append((x1, y1))
    return neighbors


def updateMaze(x, Maze, pivot, m, n):
    if pivot == 1:
        for i, j in getAllNeighbors(x, m, n):
            if Maze[i][j] != 1:
                Maze[i][j] = 0
    elif pivot == 0:
        for i, j in getAllNeighbors(x, m, n):
            if Maze[i][j] != 0:
                Maze[i][j] = 1


def updateAllVisited(visited, m, n, Maze, C, B, N, E, H):
    for i, j in visited:
        if C[i][j] == B[i][j]:
            updateMaze((i, j), Maze, 1, m, n)
            continue
        if N[i][j] - C[i][j] == E[i][j]:
            updateMaze((i, j), Maze, 0, m, n)


# shen
# grid is the random generated 2d array with probability p
# This function is to calulate the value of C of the original gridworld
def calculateC(grid, C, m, n):
    for i in range(m):
        for j in range(n):
            for nei in getAllNeighbors((i, j), m, n):
                C[i][j] += grid[nei[0]][nei[1]]
    return C
