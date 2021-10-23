# encoding = utf-8

import numpy as np
import os
import sys
from homework3.Astar import AStar, genMaze
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def genSave(total_num, m, n):
    path = "../maps"
    if not os.path.isdir(path):
        os.mkdir(path)
    for p in np.linspace(0, 0.33, 34):
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
        f = open(os.path.join("../maps", name), "w")
        f.write(js)
        f.close()


def load(m, n):
    path = "../maps"
    dir = os.listdir(path)
    for filename in dir:
        file = open(os.path.join(path, filename), "r")
        mazes = json.loads(file.read())
        for k in mazes:
            maze = np.asarray(mazes[k])
            alg = AStar(maze, 1)
            alg.start = (0, 0)
            alg.goal = (m-1, n-1)
            # if alg.run(maze, 1):
            #     print(True)
            # else:
            #     print(maze)
            #     print("aowergowiefm")
        file.close()


if __name__ == '__main__':
    total_num, m, n = 50, 101, 101
    genSave(total_num, m, n)
    load(m, n)
