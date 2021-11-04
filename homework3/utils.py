# encoding = utf-8

import numpy as np
import os
import json
import random
import sys
from generateMaze import NumpyArrayEncoder

sys.setrecursionlimit(10 ** 8)


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


def maxProbChoices(P, maxProb, origin):
    (m, n), res = P.shape, []
    for i in range(m):
        for j in range(n):
            if P[i][j] == maxProb and origin[i][j] != 1:
                res.append((i, j))
    return res


def genByNum(m, n, p):
    maze = np.zeros([m, n])
    gridList = []
    for i in range(m):
        for j in range(n):
            gridList.append((i, j))
    count, blocks = 0, round(m * n * p)
    used = set()
    while count < blocks:
        index = random.randint(0, m * n - 1)
        while True:
            # print(index)
            if index not in used:
                count += 1
                used.add(index)
                i, j = gridList[index]
                seen = np.full([m, n], False)
                maze[i][j] = 1
                start = randomInitialize(m, n, maze, True)
                dfs(start, maze, (m, n), seen)
                if sum(sum(seen)) == m * n - count:
                    # print(count)
                    break
                else:
                    count -= 1
                    used.remove(index)
                    maze[i][j] = 0
                    index = random.randint(0, m * n - 1)
            else:
                index = random.randint(0, m * n - 1)
    return maze


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
    m, n, p = 101, 101, 0.3
    map = np.zeros([m, n])
    num = 0
    for _ in range(50):
        maze1 = genByNum(50, 50, 0.3)
        maze2 = genByNum(50, 51, 0.3)
        maze3 = genByNum(51, 50, 0.3)
        maze4 = genByNum(51, 51, 0.3)
        map[0:50, 0:50] = maze1
        map[0:50, 50:] = maze2
        map[50:, 0:50] = maze3
        map[50:, 50:] = maze4
        seen = np.full([m, n], False)
        start = randomInitialize(m, n, map, True)
        dfs(start, map, (m, n), seen)
        if sum(sum(seen)) == m * n - round(m * n * p):
            num += 1
            print(True, num)
            d = dict()
            d["dim101"] = map
            js = json.dumps(d, cls=NumpyArrayEncoder)
            filenames = "dim101_" + str(num) + ".json"
            f = open(os.path.join("./full_connected_maps", filenames), "w")
            f.write(js)
            f.close()
