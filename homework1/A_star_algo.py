# -*-coding=utf-8 -*-

import numpy as np
import random
from queue import PriorityQueue
import math
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import copy


def distance(a, b, distanceType):
    """

    :param distanceType:
        distanceType_dict = {
            1: 'Manhattan',
            2: 'Euclidean',
            3: 'Chebyshev'
        }
    :param a: point a
    :param b: point b
    :return: distance between a and b
    """

    if distanceType == 1:
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    elif distanceType == 2:
        return math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]))
    elif distanceType == 3:
        return max(abs(b[0] - a[0]), abs(b[1] - a[1]))


"""define a map"""


class Map(object):

    def __init__(self, m=19, n=19):

        self.map = np.zeros([m, n])
        self.m = m
        self.n = n
        self.start = (0, 0)
        self.end = (m - 1, n - 1)

    def setStartPoint(self, point):
        self.start = point

    def getStartPoint(self):
        return self.start

    def setEndPoint(self, point):
        self.end = point

    def getEndPoint(self):
        return self.end

    def reset(self):
        self.map = np.zeros([self.m, self.n])

    def setObstacles(self, isRandom=True, rate=0.2, obstacles=None):
        """

        :param isRandom: wither generate blocks randomly
        :param rate: blocks rate
        :param obstacles: if Random==False, use given obstacles
        :return:
        """
        # rate = min(0.2, rate)
        if isRandom:
            for x in range(self.m):
                for y in range(self.n):
                    if ((x, y) == self.start) or ((x, y) == self.end):
                        continue
                    if rate > random.random():
                        self.map[x][y] = 1
        else:
            for (i, j) in obstacles:
                self.map[i][j] = 1


class AStar(object):
    """
    basic A* algorithm (find the shortest path if h(n) is admissible or optimistic)
    """

    def __init__(self, map, distanceType):
        self.map = map  # Map definition
        self.distanceType = distanceType  # heuristic function type
        self.directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # 4 neighbors
        self.visited = set()  # closedSet
        self.fringe = PriorityQueue(self.map.m * self.map.n / 2)  # PriorityQueue lowest f(n) first out
        self.cost = {}  # g(n) steps from start to current node
        self.path = {}  # record (child, parent) of each node
        self.cells = []  # record every node pop up from the fringe
        self.trajectory = []

    def calculate_distance(self, point):
        """
        calculate f(n)
        :param point:  (x, y)
        :return: f(n) = g(n) + h(n)
        """
        return self.cost.get(point) + distance(point, self.map.end, self.distanceType)

    def clear(self):
        """
        clear A* information of last executing
        """
        self.visited = set()
        self.fringe = PriorityQueue(self.map.m * self.map.n / 2)
        self.path = {}
        self.cells = []
        self.trajectory = []
        self.cost = {}

    def run(self):
        """
        run A*
        :return: is the maze is solvable
        """
        start = self.map.getStartPoint()
        self.visited.add(start)
        self.cost[start] = 0
        self.fringe.put((0, start))

        while not self.fringe.empty():
            _, cur = self.fringe.get()
            self.cells.append(cur)
            # print(cur)
            # print(self.map.map[:4, :4])

            if cur == self.map.end:
                self.trajectory.append(cur)
                while cur in self.path:
                    cur = self.path[cur]
                    self.trajectory.append(cur)
                self.trajectory.reverse()
                return True
            for (x, y) in self.directions:
                i, j = cur[0] + x, cur[1] + y
                if i < 0 or i >= self.map.m or j < 0 or j >= self.map.n:
                    continue
                if self.map.map[i][j] == 1:
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


class RepeatedAStar(object):
    """
    repeated A* algorithm
    """

    def __init__(self, map, distanceType):
        self.map = map  # map definition
        self.start = map.getStartPoint()
        self.goal = map.getEndPoint()
        self.gridWorld = Map(map.m, map.n)  # maintain a unblocked map with same size
        self.gridWorld.setStartPoint(self.start)
        self.gridWorld.setEndPoint(self.goal)
        self.distanceType = distanceType
        self.directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # 4 neighbors
        self.visited = set()  # closedSet
        self.cost = {}  # g(n) steps from start to current node
        self.path = {}  # record (child, parent) of each node
        self.trajectory = []  # record every node pop up from the fringe
        self.cells = 0

    def calculate_distance(self, point, distanceType):
        return self.cost.get(point) + distance(point, self.map.end, distanceType)

    def run(self, bumpInto=False, improvement=False):
        As = AStar(self.gridWorld, 1)

        while True:
            if not As.run():
                return False
            self.cells += len(As.cells)

            if bumpInto:
                res = self.bumpInto(As)
            else:
                res = self.Repeated_Astar(As)
            # print(res)
            if res == True:
                return True
            index, start = res[0], res[1]

            # question 8 re-start from the best place
            if improvement:
                while index > 0:
                    count = 4
                    for (i, j) in self.directions:
                        x, y = start[0] + i, start[1] + j
                        if x < 0 or x >= self.map.m or y < 0 or y >= self.map.n or As.map.map[x][y] == 1 or As.path[start] == (x, y):
                            count -= 1
                    if count > 0:
                        break
                    else:
                        As.map.map[start[0]][start[1]] = 1
                    index -= 1
                    start = As.trajectory[index]

            As.map.setStartPoint(start)
            As.clear()

    def Repeated_Astar(self, As):
        block, index = (), len(As.trajectory)
        for idx, (i, j) in enumerate(As.trajectory):
            self.trajectory.append((i, j))
            if self.map.map[i][j] == 1:
                block = (i, j)
                index = idx - 1
                break
            for nei in self.directions:
                x, y = nei[0] + i, nei[1] + j
                if x < 0 or x >= self.map.m or y < 0 or y >= self.map.n:
                    continue
                self.visited.add((x, y))
                if self.map.map[x][y] == 1:
                    As.map.map[x][y] = 1
        # print(index)
        # print(block, As.trajectory[index])
        if index == len(As.trajectory):
            return True
        As.map.map[block[0]][block[1]] = 1
        start = As.trajectory[index]
        return index, start

    def bumpInto(self, As):
        block, index = (), len(As.trajectory)
        for idx, (i, j) in enumerate(As.trajectory):
            self.trajectory.append((i, j))
            if self.map.map[i][j] == 1:
                block = (i, j)
                index = idx - 1
                break
        if index == len(As.trajectory):
            return True
        As.map.map[block[0]][block[1]] = 1
        start = As.trajectory[index]
        return index, start


def findShortestPath():
    map = Map(101, 101)
    map.setStartPoint((4, 4))
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
