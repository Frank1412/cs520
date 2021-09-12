# -*-coding=utf-8 -*-

import numpy as np
import random
from queue import PriorityQueue
import math
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm


class Map(object):

    def __init__(self, m=19, n=19):

        self.map = np.zeros([m, n])
        self.m = m
        self.n = n
        self.start = (0, 0)
        self.end = (m - 1, n - 1)

    def setStartPoint(self, x, y):
        self.start = (x, y)

    def getStartPoint(self):
        return self.start

    def setEndPoint(self, x, y):
        self.end = (x, y)

    def getEndPoint(self):
        return self.end

    def reset(self):
        self.map = np.zeros([self.m, self.n])

    def setObstacles(self, isRandom=True, rate=0.2, obstacles=None):
        # rate = min(0.2, rate)
        if isRandom:
            for x in range(self.m):
                for y in range(self.n):
                    if ([x, y] == self.start) or ([x, y] == self.end):
                        continue
                    if rate > random.random():
                        self.map[x][y] = 1
        else:
            for (i, j) in obstacles:
                self.map[i][j] = 1


class AStar(object):

    def __init__(self, map, distanceType):
        self.map = map
        self.distanceType = distanceType
        self.cost = 0
        self.directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.visited = set()
        self.fringe = PriorityQueue(self.map.m * self.map.n / 2)
        self.cost = {}
        self.path = {}
        self.trajectory = set()
        # self.blocks = set()

    def distance(self, a, b):
        """
        distanceType_dict = {
            1: 'Manhattan',
            2: 'Euclidean'
        }
        :param a: point a
        :param b: point b
        :return: distance between a and b
        """

        if self.distanceType == 1:
            return abs(b[0] - a[0]) + abs(b[1] - a[1])
        elif self.distanceType == 2:
            return math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]))

    def calculate_distance(self, point):
        return self.cost.get(point) + self.distance(point, self.map.end)

    def run(self):

        start = self.map.getStartPoint()
        self.visited.add(start)
        self.cost[start] = 0
        self.fringe.put((0, start))

        while not self.fringe.empty():
            _, cur = self.fringe.get()
            self.trajectory.add(cur)

            if cur == self.map.end:
                return True
            for (x, y) in self.directions:
                i, j = cur[0] + x, cur[1] + y
                if i < 0 or i >= self.map.m or j < 0 or j >= self.map.n:
                    continue
                # if (i, j) in self.blocks:
                #     continue
                if self.map.map[i][j] == 1:
                    # self.blocks.add((i, j))
                    continue
                if (i, j) in self.visited:
                    continue
                else:
                    self.cost[(i, j)] = self.cost.get(cur) + 1
                    self.visited.add((i, j))
                    priority = self.calculate_distance((i, j))
                    self.fringe.put((priority, (i, j)))
                    self.path[(i, j)] = cur

        return False


def findShortestPath():
    map = Map(101, 101)
    map.setStartPoint(4, 4)
    obstacles = [(0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8),
                 (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1)]
    # map.setObstacles(False, 0.1, obstacles)
    map.setObstacles(True, 0.2)
    # print(map.map)
    algo = AStar(map, 1)
    sum = 0
    for i in range(100):
        map = Map(101, 101)
        map.setObstacles(True, 0.1)
        algo = AStar(map, 1)
        result = algo.run()
        print(result)
        if result:
            sum += 1
    print(sum / 100)
    path = algo.path
    last = map.end
    while last in path:
        # print(last, path[last])
        last = path[last]

    img = Image.fromarray(np.uint8(cm.gist_earth(map.map) * 255))

    # mymap = np.array(img)  # 图像转化为二维数组
    # 绘制路径
    img = np.array(img.convert('RGB'))
    print(img.shape)
    last = map.end
    while last in path:
        img[last[0]][last[1]] = [0, 255, 255]
        last = path[last]
    start = map.getStartPoint()
    end = map.getEndPoint()
    img[start[0]][start[1]] = [255, 0, 0]
    img[end[0]][end[1]] = [255, 0, 0]
    ax = plt.gca()
    # ax.set_xticks(range(101))
    # ax.set_yticks(range(101))
    plt.imshow(img)
    plt.grid(linewidth=1)
    # plt.axis('off')
    plt.show()


if __name__ == "__main__":
    findShortestPath()


