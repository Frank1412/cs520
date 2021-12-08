# coding = utf-8

import numpy as np
from Astar import *
import pandas as pd
# from keras.utils.np_utils import to_categorical

directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
cls = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


class RepeatedAStar(object):
    def __init__(self, map):
        self.map = map  # map definition
        self.m, self.n = map.shape
        self.start = (0, 0)
        self.goal = (self.m - 1, self.n - 1)
        self.gridWorld = np.full(map.shape, 2)  # maintain a unblocked map with same size
        self.gridWorld[0][0] = 0
        self.visited = set()  # closedSet
        self.cost = {}  # g(n) steps from start to current node
        self.path = {}  # record (child, parent) of each node
        self.trajectory = []  # record every node pop up from the fringe
        self.visit = np.full(map.shape, 0)
        self.visitCount = np.zeros(map.shape)
        self.cells = 0
        self.cur = None
        self.dataX = []
        self.dataY = []

    def gen_data(self, next, prev):
        x = np.eye(3)[self.gridWorld]
        place = np.zeros(self.gridWorld.shape)
        # print(prev)
        place[prev[0]][prev[1]] = 1
        place = np.expand_dims(place, 2)
        x = np.concatenate([x, place], axis=-1)
        # x = np.concatenate([x, np.expand_dims(self.visit, 2)], axis=-1)
        # x = np.concatenate([x, np.expand_dims(self.visitCount, 2)], axis=-1)
        label = cls.get((next[0] - prev[0], next[1] - prev[1]))
        return np.array(x), label

    def followStep(self, trajectory):
        # count = min(len(trajectory), 5)
        block, index = (), len(trajectory)
        for idx, (i, j) in enumerate(trajectory):
            # self.visit[i][j] = 1
            if idx != 0:
                self.trajectory.append((i, j))
                # print(i - self.cur[0], j - self.cur[1])
                # if idx < 4:
                sample = self.gen_data((i, j), self.cur)
                self.dataX.append(sample[0])
                self.dataY.append(sample[1])
                # self.visitCount[i][j] += 1
            if self.map[i][j] == 1:
                block = (i, j)
                index = idx - 1
                break
            for nei in directions:
                x, y = nei[0] + i, nei[1] + j
                if x < 0 or x >= self.m or y < 0 or y >= self.n:
                    continue
                self.visited.add((x, y))
                if self.map[x][y] == 1:
                    self.gridWorld[x][y] = 1
                else:
                    self.gridWorld[x][y] = 0
            self.cur = (i, j)
        return index, block

    def run(self):
        self.visit[0][0] = 1
        As = AStar(self.gridWorld, 1)
        As.start = self.start
        As.goal = self.goal
        self.cur = (0, 0)
        while True:
            if not As.run():
                return False
            # sample = self.gen_data(As.trajectory[1], self.cur)
            # self.dataX.append(sample[0])
            # self.dataY.append(sample[1])

            index, block = self.followStep(As.trajectory)
            if index == len(As.trajectory):
                return True
            self.gridWorld[block[0]][block[1]] = 1
            start = As.trajectory[index]
            As = AStar(self.gridWorld, 1)
            As.start = start
            self.cur = start
            As.goal = self.goal


if __name__ == '__main__':
    m, n = 30, 30
    mazes = np.load("maps/30x30dim.npy")
    dataset = []
    print(mazes.shape)
    for i in range(len(mazes)):
        maze = mazes[i]
        alg = RepeatedAStar(maze)
        res = alg.run()
        print(res)
        # print(alg.trajectory)
        x = np.array(alg.dataX)
        y = np.array(alg.dataY)
        np.save("./data/map_{i}".format(i=i+1), x)
        np.save("./data/label_{i}".format(i=i+1), y)

    # data = np.load("./data/map1.npy")
    # x, y = data[:, 0], data[:, 1]
    # print(x[0].shape)


