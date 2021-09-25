# -*-coding=utf-8 -*-

import numpy as np
import random
from queue import PriorityQueue, Queue
import math
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]


def distance(a, b, distanceType, ):
    """
    :param distanceType:
        distanceType_dict = {
            1: 'Manhattan',
            2: 'Euclidean',
            3: 'Chebyshev',
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


def search_around(algo, algoName, fringe, cur, map, visited, cost, path, admissible=None, coef=None):
    for (x, y) in directions:
        i, j = cur[0] + x, cur[1] + y
        if i < 0 or i >= map.m or j < 0 or j >= map.n or map.map[i][j] == 1 or (i, j) in visited:
            continue
        else:
            cost[(i, j)] = cost.get(cur) + 1
            visited.add((i, j))
            if algoName == "A*":
                priority = algo.calculate_distance((i, j), admissible, coef)
                fringe.put((priority, (i, j)))
            else:
                fringe.put((i, j))
            path[(i, j)] = cur


def repeated_step(self, algo):
    block, index = (), len(algo.trajectory)
    for idx, (i, j) in enumerate(algo.trajectory):
        self.trajectory.append((i, j))
        if self.map.map[i][j] == 1:
            block = (i, j)
            index = idx - 1
            break
        for nei in directions:
            x, y = nei[0] + i, nei[1] + j
            if x < 0 or x >= self.map.m or y < 0 or y >= self.map.n:
                continue
            self.visited.add((x, y))
            if self.map.map[x][y] == 1:
                algo.map.map[x][y] = 1
    return index, block


class Map(object):

    def __init__(self, m=19, n=19):
        self.map = np.zeros([m, n])
        self.m = m
        self.n = n
        self.start = (0, 0)
        self.end = (m - 1, n - 1)

    def reset(self):
        self.map = np.zeros([self.m, self.n])

    def setObstacles(self, isRandom=True, rate=0.2, obstacles=None):
        """
        :param isRandom: wither generate blocks randomly
        :param rate: blocks rate
        :param obstacles: if Random==False, use given obstacles
        :return:
        """
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

    def calculate_distance(self, point, admissible=True, weighted=0.8):
        """
        calculate f(n)
        :param weighted:
        :param admissible:
        :param point:  (x, y)
        :return: f(n) = g(n) + h(n)
        """
        h = distance(point, self.map.end, self.distanceType)
        g = self.cost.get(point)
        if admissible == True:
            return g + h
        else:
            if admissible == "weighted":
                return g + weighted * h
            else:
                h2 = distance(point, self.map.end, 2)
                return g + 0.5 * (h + 2 * h2)

    def clear(self):
        self.visited = set()
        self.fringe = PriorityQueue(self.map.m * self.map.n / 2)
        self.path = {}
        self.cells = []
        self.trajectory = []
        self.cost = {}

    def run(self, admissible=True, coef=0.5):
        """
        :return: is the maze is solvable
        """
        start = self.map.start  # getStartPoint()
        self.visited.add(start)
        self.cost[start] = 0
        self.fringe.put((0, start))
        while not self.fringe.empty():
            _, cur = self.fringe.get()
            self.cells.append(cur)
            if cur == self.map.end:
                self.trajectory.append(cur)
                while cur in self.path:
                    cur = self.path[cur]
                    self.trajectory.append(cur)
                self.trajectory.reverse()
                return True
            search_around(self, "A*", self.fringe, cur, self.map, self.visited, self.cost, self.path, admissible, coef)
        return False


class RepeatedAStar(object):
    def __init__(self, map, distanceType):
        self.map = map  # map definition
        self.start = map.start
        self.goal = map.end
        self.gridWorld = Map(map.m, map.n)  # maintain a unblocked map with same size
        self.gridWorld.start = self.start
        self.gridWorld.end = self.goal
        self.distanceType = distanceType
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
                index, block = self.bumpInto(As)
            else:
                index, block = repeated_step(self, As)
            if index == len(As.trajectory):
                return True
            As.map.map[block[0]][block[1]] = 1
            start = As.trajectory[index]

            # question 8 re-start from the best place
            if improvement:
                while index > 0:
                    count = 4
                    for (i, j) in directions:
                        x, y = start[0] + i, start[1] + j
                        if x < 0 or x >= self.map.m or y < 0 or y >= self.map.n or As.map.map[x][y] == 1 or As.path[start] == (x, y):
                            count -= 1
                    if count > 0:
                        break
                    else:
                        As.map.map[start[0]][start[1]] = 1
                    index -= 1
                    start = As.trajectory[index]
            As.map.start = start
            As.clear()

    def bumpInto(self, As):
        block, index = (), len(As.trajectory)
        for idx, (i, j) in enumerate(As.trajectory):
            self.trajectory.append((i, j))
            if self.map.map[i][j] == 1:
                block = (i, j)
                index = idx - 1
                break
        return index, block


class BFS(object):
    def __init__(self, map):
        self.map = map  # map definition
        self.start = map.start  # getStartPoint()
        self.goal = map.end
        self.directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # 4 neighbors
        self.queue = Queue(self.map.m + self.map.n)  # queue first in first out
        self.visited = set()  # closedSet
        self.cost = {}  # g(n) steps from start to current node
        self.path = {}  # record (child, parent) of each node
        self.trajectory = []  # record every node pop up from the fringe
        self.cells = []

    def clear(self):
        self.visited = set()
        self.queue = Queue(self.map.m + self.map.n)
        self.path = {}
        self.trajectory = []
        self.cost = {}
        self.cells = []

    def run(self):
        self.queue.put(self.start)
        self.visited.add(self.start)
        self.cost[self.start] = 0
        while not self.queue.empty():
            cur = self.queue.get()
            self.cells.append(cur)
            if cur == self.map.end:
                self.trajectory.append(cur)
                while cur in self.path:
                    cur = self.path[cur]
                    self.trajectory.append(cur)
                self.trajectory.reverse()
                return True
            search_around(self, "bfs", self.queue, cur, self.map, self.visited, self.cost, self.path)
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
                    self.queue.put((i, j))
                    self.path[(i, j)] = cur
        return False


class RepeatedBFS(object):
    def __init__(self, map):
        self.map = map  # map definition
        self.start = map.start
        self.goal = map.end
        self.gridWorld = Map(map.m, map.n)  # maintain a unblocked map with same size
        self.gridWorld.start = self.start
        self.gridWorld.goal = self.goal
        self.directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # 4 neighbors
        self.queue = Queue()  # queue first in first out
        self.visited = set()  # closedSet
        self.cost = {}  # g(n) steps from start to current node
        self.path = {}  # record (child, parent) of each node
        self.trajectory = []  # record every node pop up from the fringe
        self.cells = 0

    def run(self):
        bfs = BFS(self.gridWorld)
        while True:
            if not bfs.run():
                return False
            self.cells += len(bfs.cells)
            index, block = repeated_step(self, bfs)
            if index == len(bfs.trajectory):
                return True
            bfs.map.map[block[0]][block[1]] = 1
            bfs.map.start = bfs.trajectory[index]
            bfs.clear()

    def bump_into(self):
        bfs = BFS(self.gridWorld)
        while True:
            if not bfs.run():
                return False
            self.cells += len(bfs.cells)
            block, index = (), len(bfs.trajectory)
            for idx, (i, j) in enumerate(bfs.trajectory):
                self.trajectory.append((i, j))
                if self.map.map[i][j] == 1:
                    block = (i, j)
                    index = idx - 1
                    break
            if index == len(bfs.trajectory):
                return True
            bfs.map.map[block[0]][block[1]] = 1
            start = bfs.trajectory[index]
            bfs.map.start = start
            bfs.clear()
