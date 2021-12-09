# coding=utf-8

# from homework3 import Astar
from Astar import *
import os
import numpy as np


def genSave(total_num, m, n):
    path = "./maps"
    if not os.path.isdir(path):
        os.mkdir(path)
    maps = []
    for p in np.linspace(0.3, 0.3, 1):
        for i in range(total_num):
            maze = genMaze(m, n, p, (0, 0), (m - 1, n - 1))
            alg = AStar(maze, 1)
            alg.start = (0, 0)
            alg.goal = (m - 1, n - 1)
            while not alg.run():
                maze = genMaze(m, n, p, (0, 0), (m - 1, n - 1))
                alg = AStar(maze, 1)
                alg.start = (0, 0)
                alg.goal = (m - 1, n - 1)
            maps.append(maze)
    maps = np.array(maps)
    np.save(os.path.join(path, "test_30x30dim"), maps)
    loadmaps = np.load(os.path.join(path, "test_30x30dim.npy"))
    print(loadmaps.shape)


if __name__ == '__main__':
    genSave(200, 30, 30)
