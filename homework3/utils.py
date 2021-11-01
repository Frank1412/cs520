# encoding = utf-8

import numpy as np
import os
import json
import random
import sys
sys.setrecursionlimit(10 ** 6)

def loadMaze(path, filename):
    file = open(os.path.join(path, filename), "r")
    mazes = json.loads(file.read())
    allMaps = []
    for k in mazes:
        maze = np.asarray(mazes[k])
        allMaps.append(maze)
    file.close()
    return allMaps


def randomInitialize(m, n, maze, start):
    x = random.randint(0, m - 1)
    y = random.randint(0, n - 1)
    if start:
        while maze[x][y] == 1:
            x = random.randint(0, m - 1)
            y = random.randint(0, n - 1)
    return x, y


def generateTerrain(m, n):
    """
    randomly generate terrain 0:flat, 1:hilly, 2:forest
    :param m:
    :param n:
    :return:
    """
    terrain = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            if random.random() < 1 / 3:
                terrain[i][j] = 1  # hilly
            elif random.random() < 2 / 3:
                terrain[i][j] = 2  # forest
    return terrain


def maxProbChoices(P, maxProb):
    (m, n), res = P.shape, []
    for i in range(m):
        for j in range(n):
            if P[i][j] == maxProb:
                res.append((i, j))
    return res


def genMaze(m, n, p):
    maze = np.zeros([m, n])
    for x in range(m):
        for y in range(n):
            if random.random() < p:
                maze[x][y] = 1
    while True:
        cells = m * n - sum(sum(maze))
        start = randomInitialize(m, n, maze, True)
        seen = np.full([m, n], False)
        dfs(start, maze, (m, n), seen)
        if sum(sum(seen)) == cells:
            print(cells, True)
            break
        else:
            print(cells, sum(sum(seen)))
            maze = np.zeros([m, n])
            for x in range(m):
                for y in range(n):
                    if random.random() < p:
                        maze[x][y] = 1
    return maze


direction = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def getAllNeighbors(x, m, n):
    neighbors = []
    for dir in direction:
        x1 = x[0] + dir[0]
        y1 = x[1] + dir[1]
        if 0 <= x1 < m and 0 <= y1 < n:
            neighbors.append((x1, y1))
    return neighbors


def dfs(grid, maze, shape, seen):
    i, j = grid
    m, n = shape
    if maze[i][j] == 1 or seen[i][j]:
        return
    seen[i][j] = True
    for point in getAllNeighbors((i, j), m, n):
        dfs(point, maze, shape, seen)


if __name__ == '__main__':
    for _ in range(1):
        maze = genMaze(101, 101, 0.3)
