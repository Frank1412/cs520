# encoding = utf-8

import numpy as np
import os
import sys
from Astar import AStar, genMaze
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def genSave(total_num, m, n):
    path = "../homework4/maps"
    if not os.path.isdir(path):
        os.mkdir(path)
    for p in np.linspace(0.3, 0.3, 1):
        name = "density" + str(p) + ".json"
        maps = dict()
        for i in range(total_num):
            maze = genMaze(m, n, p, (0, 0), (m-1, n-1))
            alg = AStar(maze, 1)
            alg.start = (0, 0)
            alg.goal = (m - 1, n - 1)
            while not alg.run():
                maze = genMaze(m, n, p, (0, 0), (m - 1, n - 1))
                alg = AStar(maze, 1)
                alg.start = (0, 0)
                alg.goal = (m - 1, n - 1)
            # print(alg.trajectory, alg.goal)
            maps[i] = maze
        js = json.dumps(maps, cls=NumpyArrayEncoder)
        f = open(os.path.join("../homework4/maps", name), "w")
        f.write(js)
        f.close()


def load(m, n):
    path = "../homework4/maps"
    dir = os.listdir(path)
    for filename in dir:
        file = open(os.path.join(path, filename), "r")
        mazes = json.loads(file.read())
        for k in mazes:
            maze = np.asarray(mazes[k])
            print(maze.shape)
            alg = AStar(maze, 1)
            alg.start = (0, 0)
            alg.goal = (m-1, n-1)
        file.close()
    return


if __name__ == '__main__':
    total_num, m, n = 50, 30, 30
    # genSave(total_num, m, n)
    load(m, n)
