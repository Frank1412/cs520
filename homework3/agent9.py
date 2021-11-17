# encoding=utf-8

import copy
import numpy as np
from random import random
from utils import *
from Astar import AStar

direction = [(1, 0), (-1, 0), (0, 1), (0, -1)]
outer = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]  # 3/8
inner = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]  # 1/2


class Agent9(object):

    def __init__(self, maze):
        self.origin = maze
        self.m, self.n = maze.shape
        self.gridWorld = np.full([self.m, self.n], 2)
        self.sense = False
        self.preSense = False
        self.trajectory = []
        self.cur = ()
        self.start = ()
        self.target = ()
        self.distance = 1

    def randomNextStep(self, cur):
        choices = []
        for i, j in direction:
            x1 = cur[0] + i
            y1 = cur[1] + j
            if 0 <= x1 < m and 0 <= y1 < n and self.origin[i][j] != 1:
                choices.append((x1, y1))
        return choices[random.randint(0, len(choices) - 1)]

    def randomInRadius(self):
        choices = []
        for i in range(self.m):
            for j in range(self.n):
                if abs(self.cur[0]-i) + abs(self.cur[1]-j) <= self.distance:
                    choices.append((i, j))
        return choices

    def replan(self):
        As = AStar(self.gridWorld, 1)
        As.start = self.cur
        if not self.sense:
            if not self.preSense or self.distance > 20:
                self.preSense = False
                As.goal = randomInitialize(self.m, self.n, self.gridWorld, True)
                while not As.run():
                    i, j = As.goal
                    self.gridWorld[i][j] = 1
                    As = AStar(self.gridWorld, 1)
                    As.start = self.cur
                    As.goal = randomInitialize(self.m, self.n, self.gridWorld, True)
            else:
                choices = self.randomInRadius()
                As.goal = choices[random.randint(0, len(choices)-1)]
                while not As.run():
                    i, j = As.goal
                    self.gridWorld[i][j] = 1
                    As = AStar(self.gridWorld, 1)
                    As.start = self.cur
                    As.goal = choices[random.randint(0, len(choices) - 1)]
        else:
            As.goal = self.predGoal(start)
            while not As.run():
                i, j = As.goal
                self.gridWorld[i][j] = 1
                As = AStar(self.gridWorld, 1)
                As.start = self.cur
                As.goal = self.predGoal(start)
        return As.trajectory

    def predGoal(self, cur):
        outer, inner = [], []
        rand = random.random()
        if rand < 1 / 8:  # itself
            return cur
        elif rand < 1 / 2:  # outer
            choices = []
            for i, j in outer:
                x, y = cur[0] + i, cur[1] + j
                if self.gridWorld[i][j] != 1:
                    choices.append((i, j))
            return choices[random.randint(0, len(choices) - 1)]
        else:   # inner
            choices = []
            for i, j in inner:
                x, y = cur[0] + i, cur[1] + j
                if self.gridWorld[i][j] != 1:
                    choices.append((i, j))
            return choices[random.randint(0, len(choices) - 1)]

    def senseTarget(self, cur):
        for i, j in inner:
            x, y = cur[0] + i, cur[1] + j
            if (x, y) == self.target:
                self.sense = True
                self.preSense = True
                return

    def followPath(self, trajectory):
        ret, last = 0, trajectory[-1]

        for (i, j) in trajectory:
            if self.cur != (i, j):
                self.trajectory.append((i, j))
            if self.origin[i][j] == 1:  # find block, no sense, replan
                self.gridWorld[i][j] = 1
                break
            self.cur = (i, j)
            self.senseTarget((i, j))
            if self.sense:
                return 1  # not find but sense, replan
            else:
                if self.preSense:
                    self.target = self.randomNextStep(self.target)
                    self.distance += 1
                    break
            self.target = self.randomNextStep(self.target)
            if (i, j) == last:   # find
                if (i, j) == self.target:
                    return 2  # find the target
        return 0  # no sense, replan

    def run(self):
        goal = randomInitialize(self.m, self.n, self.gridWorld, True)
        self.cur = self.start
        while True:
            trajectory = self.replan()
            ret = self.followPath(trajectory)
            if ret == 2:
                break
        return True
