# encoding=utf-8

import numpy as np

directions = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]


def sense(x, Maze, C):
    blocks = 0
    for i, j in directions:
        if Maze[x[0] + i][x[1] + j] == 1:
            blocks += 1
    return blocks


def updateMaze(x, Maze, pivot, m, n):
    if pivot == 1:
        for i, j in getAllNeighbors(x, m, n):
            if Maze[i][j] != 1:
                Maze[i][j] = 0
    elif pivot == 0:
        for i, j in getAllNeighbors(x, m, n):
            if Maze[i][j] != 0:
                Maze[i][j] = 1


def isVertex(x, m, n):
    return (x[0] == 0 and x[1] == 0) or (x[0] == 0 and x[1] == n - 1) or (x[0] == m - 1 and x[1] == 0) or (x[0] == m - 1 and x[1] == n - 1)


def isBorder(x, m, n):
    return x[0] == 0 or x[0] == m - 1 or x[1] == 0 or x[1] == n - 1


def isValid(x, m, n):
    return 0 <= x[0] < m and 0 <= x[1] < n


def initializeN(Maze):
    m, n = len(Maze), len(Maze[0])
    N = np.full((m, n), 8)
    B = np.zeros([m, n])
    E = np.zeros([m, n])
    H = np.full([m, n], 8)
    for i in range(m):
        for j in range(n):
            if isVertex((i, j), m, n):
                N[i][j], H[i][j] = 3, 3
            elif isBorder((i, j), m, n):
                N[i][j], H[i][j] = 5, 5
    return N


def getAllNeighbors(x, m, n):  # return valid neighbors
    neighbors = []
    for dir in directions:
        x1 = x[0] + dir[0]
        y1 = x[1] + dir[1]
        if 0 <= x1 < m and 0 <= y1 < n:
            neighbors.append((x1, y1))
    return neighbors


def updateAllVisited(visited, m, n, Maze, C, B, N, E, H):
    for i, j in visited:
        if C[i][j] == B[i][j]:
            updateMaze((i, j), Maze, 1, m, n)
            continue
        if N[i][j] - C[i][j] == E[i][j]:
            updateMaze((i, j), Maze, 0, m, n)
