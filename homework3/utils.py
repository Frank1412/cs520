# encoding = utf-8

import numpy as np
import os
import json
import random


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
