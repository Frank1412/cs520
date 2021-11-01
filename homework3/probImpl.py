# encoding=utf-8

import copy
import numpy as np
from random import random
from utils import *
from Astar import AStar
import time


class ProbAgent(object):
    def __init__(self, maze, target):
        self.origin = maze
        self.m, self.n = maze.shape
        self.start = ()
        self.goal = ()
        self.target = target
        self.containP = np.full([self.m, self.n], 1.0 / (self.m * self.n))
        self.maxFindingProb = 0
        self.maxContainProb = 0
        self.findingP = copy.deepcopy(self.containP)
        self.gridWorld = np.full([self.m, self.n], 2)
        self.terrain = generateTerrain(self.m, self.n)
        self.blocks = 0
        self.unknownGrid = self.m * self.n
        self.cur = ()
        self.trajectory = []
        self.agentType = 6

    def followPlan(self, trajectory):
        for (i, j) in trajectory:
            self.trajectory.append((i, j))
            ret = self.updateContainP((i, j))
            if ret == 0:  # find the target
                return 0
            if ret == 1:  # find block
                return 1
            if self.agentType == 7:
                self.updateFindingP((i, j))
            self.cur = (i, j)
        return 2

    def updateUnknownProb(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.gridWorld[i][j] == 2:
                    self.containP[i][j] = 1.0 / self.unknownGrid

    def updateUnvisitedProb(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.gridWorld[i][j] == 2:
                    self.findingP[i][j] = 1.0 / self.unknownGrid * 0.5

    def updateContainP(self, grid):
        i, j = grid
        if self.origin[i][j] == 1:
            self.containP[i][j] = 0
            self.gridWorld[i][j] = 1
            self.unknownGrid -= 1
            self.updateUnknownProb()
            if self.agentType == 7:
                self.updateUnvisitedProb()
            return 1
        else:
            self.gridWorld[i][j] = 0
            if (i, j) == self.target:
                self.containP[i][j] = 1
                self.maxContainProb = 1
                return 0
            p = self.containP[i][j]
            if self.terrain[i][j] == 0:  # flat
                self.containP[i][j] = 0.2 * p / (1 - 0.8 * p)
            elif self.terrain[i][j] == 1:  # hilly
                self.containP[i][j] = 0.5 * p / (1 - 0.5 * p)
            else:  # forest
                self.containP[i][j] = 0.8 * p / (1 - 0.2 * p)
            return 2

    def updateFindingP(self, grid):
        (i, j) = grid
        p = self.containP[i][j]
        if self.terrain[i][j] == 0:  # flat
            self.findingP[i][j] = 0.8 * p
        elif self.terrain[i][j] == 1:  # hilly
            self.findingP[i][j] = 0.5 * p
        else:  # forest
            self.findingP[i][j] = 0.2 * p

    def agent6(self):
        # self.start = randomInitialize(self.m, self.n, self.origin, True)
        # self.goal = randomInitialize(self.m, self.n, self.origin, False)
        self.cur = self.start
        while True:
            As = AStar(self.gridWorld, 1)
            As.start = self.start
            As.goal = self.goal
            As.run()
            # print(As.trajectory)
            ret = self.followPlan(As.trajectory)
            if ret == 0:
                break
            maxProb = np.amax(self.containP)
            choiceList = maxProbChoices(self.containP, maxProb)
            index = 0
            if len(choiceList) > 1:
                index = random.randint(0, len(choiceList) - 1)
            self.goal = choiceList[index]  # tuple deep copy
            self.start = self.cur
            # print(self.origin[self.start[0]][self.start[1]], self.start, self.goal, self.containP[self.goal[0]][self.goal[1]])
        return

    def agent7(self):
        self.updateUnvisitedProb()
        self.cur = self.start
        while True:
            As = AStar(self.gridWorld, 1)
            As.start = self.start
            As.goal = self.goal
            As.run()
            # print(As.trajectory)
            ret = self.followPlan(As.trajectory)
            if ret == 0:
                break
            maxProb = np.amax(self.findingP)
            choiceList = maxProbChoices(self.findingP, maxProb)
            index = 0
            if len(choiceList) > 1:
                index = random.randint(0, len(choiceList) - 1)
            self.goal = choiceList[index]  # tuple deep copy
            self.start = self.cur
            # print(self.origin[self.start[0]][self.start[1]], self.start, self.goal, self.findingP[self.goal[0]][self.goal[1]])
        return


if __name__ == '__main__':
    allMaze = loadMaze("../maps", "density0.3.json")
    for map in allMaze:
        agent6 = ProbAgent(map, (100, 100))
        agent6.start = randomInitialize(agent6.m, agent6.n, map, True)
        agent6.goal = randomInitialize(agent6.m, agent6.n, map, False)

        agent7 = ProbAgent(map, (100, 100))
        agent7.start = agent6.start
        agent7.goal = agent6.goal

        agent6.agentType = 6
        time1 = time.time()
        agent6.agent6()
        time2 = time.time()
        print("true")
        print(len(agent6.trajectory), time2-time1)

        agent7.agentType = 7
        time3 = time.time()
        agent7.agent7()
        time4 = time.time()
        print("true")

        print(len(agent7.trajectory), time4 - time3)
        break
    print(len(allMaze))
