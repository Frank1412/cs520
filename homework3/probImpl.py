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
        self.examined = np.full([self.m, self.n], False)
        self.terrain = generateTerrain(self.m, self.n)
        self.blocks = 0
        self.unknown = self.m * self.n
        self.cur = ()
        self.trajectory = []
        self.agentType = 6
        self.As = None

    def followPlan(self, trajectory):
        # print(len(trajectory))
        ret, last = 2, trajectory[-1]

        for (i, j) in trajectory:
            if self.cur != (i, j):
                self.trajectory.append((i, j))
            ret = self.passNoExamine((i, j))  # 2 unblock, 1 find the block
            if ret == 1:  # find block
                self.normalize()
                return 1
            self.cur = (i, j)
            if (i, j) == last:
                # print(True)
                ret = self.examine((i, j))
                if ret == 0:
                    break
            self.normalize()
            curMaxProb = np.amax(self.containP)
            if curMaxProb > self.containP[self.goal[0]][self.goal[1]]:
                ret = 2
                break
        return ret  # 2 unblock, 1 find the block, 0 target

    def examine(self, grid):
        i, j = grid
        if grid == self.target:
            self.containP[i][j] = 1
            self.gridWorld[i][j] = 0
            return 0
        else:
            p = self.containP[i][j]
            if self.terrain[i][j] == 0:  # flat
                self.containP[i][j] = 0.2 * p / (1 - 0.8 * p)
            elif self.terrain[i][j] == 1:  # hilly
                self.containP[i][j] = 0.5 * p / (1 - 0.5 * p)
            else:  # forest
                self.containP[i][j] = 0.8 * p / (1 - 0.2 * p)
            # self.updateExamine(grid, p)
            return 2

    def passNoExamine(self, grid):
        i, j = grid
        if self.origin[i][j] == 1:  # block
            self.containP[i][j] = 0
            self.gridWorld[i][j] = 1
            self.unknown -= 1
            self.blockUpdate(grid, 0.3)
            return 1
        else:  # unblock
            if self.gridWorld[i][j] == 2:
                self.gridWorld[i][j] = 0
                self.containP[i][j] = 1 / self.unknown / 0.7
                # self.updateNoExamineIJ(grid, 0.7)
        return 2

    def blockUpdate(self, grid, p):
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) != grid:
                    self.containP[i][j] = self.containP[i][j] / p

    def updateNoExamineIJ(self, grid, p):
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) != grid:
                    self.containP[i][j] = self.containP[i][j] / p

    def updateExamine(self, grid, p):
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) != grid:
                    if self.terrain[i][j] == 0:  # flat
                        self.containP[i][j] = 0.2 * self.containP[i][j] / (1 - 0.8 * p)
                    elif self.terrain[i][j] == 1:  # hilly
                        self.containP[i][j] = 0.5 * self.containP[i][j] / (1 - 0.5 * p)
                    else:  # forest
                        self.containP[i][j] = 0.8 * self.containP[i][j] / (1 - 0.2 * p)
                else:
                    if self.terrain[i][j] == 0:  # flat
                        self.containP[i][j] = 0.2 * p / (1 - 0.8 * p)
                    elif self.terrain[i][j] == 1:  # hilly
                        self.containP[i][j] = 0.5 * p / (1 - 0.5 * p)
                    else:  # forest
                        self.containP[i][j] = 0.8 * p / (1 - 0.2 * p)

    def normalize(self):
        total = sum(sum(self.containP))
        self.containP = self.containP / total

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
        print(self.start, self.goal, self.target)
        self.cur = self.start
        while True:
            self.As = AStar(self.gridWorld, 1)
            self.As.start = self.start
            self.As.goal = self.goal
            self.As.run()
            ret = self.followPlan(self.As.trajectory)
            if ret == 0:
                break
            maxProb = np.amax(self.containP)
            choiceList = maxProbChoices(self.containP, maxProb)
            index = 0
            if len(choiceList) > 1:
                index = random.randint(0, len(choiceList) - 1)
                x, y = choiceList[index]
                while self.origin[x][y] == 1:
                    index = random.randint(0, len(choiceList) - 1)
                    x, y = choiceList[index]
            self.goal = choiceList[index]
            self.start = self.cur
            # print(self.cur, self.containP[self.cur[0]][self.cur[1]], self.start, self.goal, self.containP[self.goal[0]][self.goal[1]])
        return

    def agent7(self):
        # self.updateUnvisitedProb()
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
            print(self.origin[self.start[0]][self.start[1]], self.start, self.goal, self.findingP[self.goal[0]][self.goal[1]])
        return


def genMaze(m, n, p, start, goal):
    maze = np.zeros([m, n])
    for x in range(m):
        for y in range(n):
            if p > random.random():
                maze[x][y] = 1
    return maze


if __name__ == '__main__':
    # allMaze = loadMaze("../maps", "density0.3.json")
    n = 20
    allMaze = loadMaze("./", "dim101.json")
    print(allMaze[0].shape)
    map = allMaze[0]
    timeAgent6, timeAgent7 = 0, 0
    tjtAgent6, tjtAgent7 = 0, 0
    for _ in range(n):
        target = randomInitialize(map.shape[0], map.shape[1], map, True)
        start = randomInitialize(map.shape[0], map.shape[1], map, True)
        goal = randomInitialize(map.shape[0], map.shape[1], map, False)

        agent6 = ProbAgent(map, target)
        agent6.start = start
        agent6.goal = goal

        # agent7 = ProbAgent(map, target)
        # agent7.start = start
        # agent7.goal = goal

        agent6.agentType = 6
        time1 = time.time()
        agent6.agent6()
        time2 = time.time()
        print("agent6 true, time={time}, trajectory length={tjt}".format(time=time2 - time1, tjt=len(agent6.trajectory)))
        timeAgent6 += time2 - time1
        tjtAgent6 += len(agent6.trajectory)
        # print(len(agent6.trajectory), time2 - time1)

        # agent7.agentType = 7
        # time3 = time.time()
        # agent7.agent7()
        # time4 = time.time()
        # print("agent7 true, time={time}, trajectory length={tjt}".format(time=time4-time3, tjt=len(agent7.trajectory)))
        # timeAgent7 += time4 - time3
        # tjtAgent7 += len(agent7.trajectory)
        # print(len(agent7.trajectory), time4 - time3)
        # break
    print("agent6 time={timeAgent6}, trajectory length={len}".format(timeAgent6=timeAgent6 / n, len=tjtAgent6/n))
    # print("agent7 time={timeAgent7}, trajectory length={len}".format(timeAgent7=timeAgent7/n, len=tjtAgent7))
