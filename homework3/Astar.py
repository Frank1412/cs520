# encoding = utf-8

import numpy as np
import copy
import random
from queue import PriorityQueue
import math

directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def distance(a, b, distanceType):
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


def genMaze(m, n, p, start, goal):
    maze = np.zeros([m, n])
    for x in range(m):
        for y in range(n):
            if (x, y) == start or ((x, y) == goal):
                continue
            if p > random.random():
                maze[x][y] = 1
    return maze


class AStar(object):
    """
    basic A* algorithm (find the shortest path if h(n) is admissible or optimistic)
    """

    def __init__(self, map, distanceType):
        self.map = map  # Map definition
        self.m = self.map.shape[0]
        self.n = self.map.shape[1]
        self.start = (0, 0)
        self.goal = (0, 0)
        self.distanceType = distanceType  # heuristic function type
        self.directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # 4 neighbors
        self.visited = set()  # closedSet
        self.fringe = PriorityQueue(self.m * self.n / 2)  # PriorityQueue lowest f(n) first out
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
        h = distance(point, self.goal, self.distanceType)
        g = self.cost.get(point)
        return g+h

    def clear(self):
        self.visited.clear()
        self.fringe = PriorityQueue(self.m * self.n / 2)
        self.path = {}
        self.cells = []
        self.trajectory = []
        self.cost = {}

    def run(self, admissible=True, coef=0.5):
        """
        :return: is the maze is solvable
        """
        start = self.start
        self.visited.add(start)
        self.cost[start] = 0
        self.fringe.put((0, start))
        while not self.fringe.empty():
            _, cur = self.fringe.get()
            self.cells.append(cur)
            if cur == self.goal:
                self.trajectory.append(cur)
                while cur in self.path:
                    cur = self.path[cur]
                    self.trajectory.append(cur)
                self.trajectory.reverse()
                return True
            for (x, y) in directions:
                i, j = cur[0] + x, cur[1] + y
                if i < 0 or i >= self.m or j < 0 or j >= self.n or self.map[i][j] == 1 or (i, j) in self.visited:
                    continue
                else:
                    self.cost[(i, j)] = self.cost.get(cur) + 1
                    self.visited.add((i, j))
                    priority = self.calculate_distance((i, j), admissible, coef)
                    self.fringe.put((priority, (i, j)))
                    self.path[(i, j)] = cur
        return False


if __name__ == '__main__':
    total_num, p = 20, 0.2
    for _ in range(total_num):
        maze = genMaze(101, 101, p, (0, 0), (100, 100))
        alg = AStar(maze, 1)
        alg.start = (0, 0)
        alg.goal = (100, 100)
        while True:
            res = alg.run()
            print(res)
            if not res:
                maze = genMaze(101, 101, p, (0, 0), (100, 100))
                alg = AStar(maze, 1)
                alg.start = (0, 0)
                alg.goal = (100, 100)
            else:
                break
