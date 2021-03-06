# encoding=utf-8

import copy
import numpy as np
from random import random
from utils import *
from Astar import AStar
import time
import gc


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
        self.terrain = []
        self.blocks = 0
        self.unknown = self.m * self.n
        self.cur = ()
        self.trajectory = []
        self.agentType = 6
        self.examination = 0
        self.choiceList = []
        self.As = None

    def followPlan(self, trajectory):
        # print(len(trajectory))
        ret, last = 2, trajectory[-1]

        for (i, j) in trajectory:
            if self.cur != (i, j):
                self.trajectory.append((i, j))
            ret = self.passNoExamine((i, j))  # 2 unblock, 1 find the block
            if ret == 1:  # find block
                self.normalizeContainP()
                if self.agentType == 7:
                    self.updateFindingP((i, j))
                    # self.normalizeFindingP()
                if self.agentType == 6:
                    self.maxContainProb = np.amax(self.containP)
                if self.agentType == 7:
                    self.maxFindingProb = np.amax(self.findingP)
                return 1
            self.cur = (i, j)
            if (i, j) == last:
                ret = self.examine((i, j))
                if ret == 0:
                    break
            self.normalizeContainP()
            if self.agentType == 7:
                self.updateFindingP((i, j))
                # self.normalizeFindingP()
            if self.judgeMaxChange():
                ret = 2
                break
        return ret  # 2 unblock, 1 find the block, 0 target

    def judgeMaxChange(self):
        if self.agentType == 6:
            self.maxContainProb = np.amax(self.containP)
            if self.maxContainProb > self.containP[self.goal[0]][self.goal[1]]:
                return True
        else:
            self.maxFindingProb = np.amax(self.findingP)
            if self.maxFindingProb > self.findingP[self.goal[0]][self.goal[1]]:
                return True
        return False

    def examine(self, grid):
        self.examination += 1
        i, j = grid
        p = self.containP[i][j]
        if grid == self.target:
            rand = random.random()
            if self.terrain[i][j] == 0:  # flat
                if rand > 0.2:
                    self.containP[i][j] = 1
                    self.findingP[i][j] = 1
                    self.gridWorld[i][j] = 0
                    return 0
                else:
                    self.containP[i][j] = 0.2 * p / (1 - 0.8 * p)
                    for x in range(self.m):
                        for y in range(self.n):
                            if (x, y) != grid:
                                self.containP[x][y] = self.containP[x][y] / (1 - 0.8 * p)
            if self.terrain[i][j] == 1:  # hilly
                if rand > 0.5:
                    self.containP[i][j] = 1
                    self.findingP[i][j] = 1
                    self.gridWorld[i][j] = 0
                    return 0
                else:
                    self.containP[i][j] = 0.5 * p / (1 - 0.5 * p)
                    for x in range(self.m):
                        for y in range(self.n):
                            if (x, y) != grid:
                                self.containP[x][y] = self.containP[x][y] / (1 - 0.5 * p)
            if self.terrain[i][j] == 2:  # forest
                if rand > 0.8:
                    self.containP[i][j] = 1
                    self.findingP[i][j] = 1
                    self.gridWorld[i][j] = 0
                    return 0
                else:
                    self.containP[i][j] = 0.8 * p / (1 - 0.2 * p)
                    for x in range(self.m):
                        for y in range(self.n):
                            if (x, y) != grid:
                                self.containP[x][y] = self.containP[x][y] / (1 - 0.2 * p)
            return 2
        else:
            if self.terrain[i][j] == 0:  # flat
                self.containP[i][j] = 0.2 * p / (1 - 0.8 * p)
                for x in range(self.m):
                    for y in range(self.n):
                        if (x, y) != grid:
                            self.containP[x][y] = self.containP[x][y] / (1 - 0.8 * p)
            if self.terrain[i][j] == 1:  # hilly
                self.containP[i][j] = 0.5 * p / (1 - 0.5 * p)
                for x in range(self.m):
                    for y in range(self.n):
                        if (x, y) != grid:
                            self.containP[x][y] = self.containP[x][y] / (1 - 0.5 * p)
            if self.terrain[i][j] == 2:  # forest
                self.containP[i][j] = 0.8 * p / (1 - 0.2 * p)
                for x in range(self.m):
                    for y in range(self.n):
                        if (x, y) != grid:
                            self.containP[x][y] = self.containP[x][y] / (1 - 0.2 * p)
            return 2

    def passNoExamine(self, grid):
        i, j = grid
        if self.origin[i][j] == 1:  # block
            self.containP[i][j] = 0
            self.gridWorld[i][j] = 1
            # self.unknown -= 1
            # self.blockUpdate(grid, 0.3)
            return 1
        # else:  # unblock
        #     if self.gridWorld[i][j] == 2:
        #         self.gridWorld[i][j] = 0
        #         self.containP[i][j] = 1 / self.unknown / 0.7
        return 2

    def normalizeContainP(self):
        total = sum(sum(self.containP))
        self.containP = self.containP / total

    def normalizeFindingP(self):
        total2 = sum(sum(self.findingP))
        self.findingP = self.findingP / total2

    def updateFindingP(self, grid):
        for x in range(self.m):
            for y in range(self.n):
                p = self.containP[x][y]
                if self.gridWorld[x][y] == 2:
                    self.findingP[x][y] = p * 0.5
                else:
                    if self.terrain[x][y] == 0:  # flat
                        self.findingP[x][y] = 0.8 * p
                    elif self.terrain[x][y] == 1:  # hilly
                        self.findingP[x][y] = 0.5 * p
                    else:  # forest
                        self.findingP[x][y] = 0.2 * p

    def agent6(self):
        self.goal = randomInitialize(self.origin.shape[0], self.origin.shape[1], self.gridWorld, False)
        print(self.terrain[self.target[0]][self.target[1]])
        print(self.start, self.goal, self.target)
        self.cur = self.start
        while True:
            self.As = AStar(self.gridWorld, 1)
            self.As.start = self.start
            self.As.goal = self.goal
            while not self.As.run():
                i, j = self.As.goal
                self.gridWorld[i][j] = 1
                self.containP[i][j] = 0
                self.maxContainProb = np.amax(self.containP)
                # self.choiceList = maxProbChoices(self.containP, self.maxContainProb, self.gridWorld)
                # index = random.randint(0, len(self.choiceList) - 1)
                # x, y = self.choiceList[index]
                self.choiceList = maxProbSortByDis(self.containP, self.maxContainProb, self.gridWorld, self.cur)
                x, y = self.choiceList[random.randint(0, len(self.choiceList) - 1)]
                self.As = AStar(self.gridWorld, 1)
                self.As.start = self.start
                self.As.goal = (x, y)
            self.normalizeContainP()
            ret = self.followPlan(self.As.trajectory)
            if ret == 0:
                break
            # self.choiceList = maxProbChoices(self.containP, self.maxContainProb, self.gridWorld)
            # index = random.randint(0, len(self.choiceList) - 1)
            # x, y = self.choiceList[index]
            self.choiceList = maxProbSortByDis(self.containP, self.maxContainProb, self.gridWorld, self.cur)
            x, y = self.choiceList[random.randint(0, len(self.choiceList) - 1)]
            self.goal = (x, y)
            self.start = self.cur
            # print(self.start, self.goal, self.containP[self.goal[0]][self.goal[1]], self.containP[self.target[0]][self.target[1]])
        return True

    def agent7(self):
        print(self.terrain[self.target[0]][self.target[1]])
        self.cur = self.start
        self.goal = randomInitialize(self.origin.shape[0], self.origin.shape[1], self.gridWorld, False)
        print(self.start, self.goal, self.target)
        while True:
            self.As = AStar(self.gridWorld, 1)
            self.As.start = self.start
            self.As.goal = self.goal
            while not self.As.run():
                i, j = self.As.goal
                self.gridWorld[i][j] = 1
                self.containP[i][j] = 0
                self.findingP[i][j] = 0
                self.maxFindingProb = np.amax(self.findingP)
                # self.choiceList = maxProbChoices(self.findingP, self.maxFindingProb, self.gridWorld)
                # index = random.randint(0, len(self.choiceList) - 1)
                # x, y = self.choiceList[index]
                self.choiceList = maxProbSortByDis(self.findingP, self.maxFindingProb, self.gridWorld, self.cur)
                x, y = self.choiceList[random.randint(0, len(self.choiceList) - 1)]
                self.As = AStar(self.gridWorld, 1)
                self.As.start = self.start
                self.As.goal = (x, y)
            self.normalizeContainP()
            self.normalizeFindingP()
            ret = self.followPlan(self.As.trajectory)
            if ret == 0:
                break
            # self.choiceList = maxProbChoices(self.findingP, self.maxFindingProb, self.gridWorld)
            # index = random.randint(0, len(self.choiceList) - 1)
            # x, y = self.choiceList[index]
            self.choiceList = maxProbSortByDis(self.findingP, self.maxFindingProb, self.gridWorld, self.cur)
            x, y = self.choiceList[random.randint(0, len(self.choiceList) - 1)]
            self.goal = (x, y)
            self.start = self.cur
            # print(self.start, self.goal, self.findingP[self.goal[0]][self.goal[1]], self.findingP[self.target[0]][self.target[1]])
        return True

    def agent8(self):
        print(self.terrain[self.target[0]][self.target[1]])
        self.cur = self.start
        self.goal = randomInitialize(self.origin.shape[0], self.origin.shape[1], self.gridWorld, False)
        print(self.start, self.goal, self.target)
        while True:
            self.As = AStar(self.gridWorld, 1)
            self.As.start = self.start
            self.As.goal = self.goal
            while not self.As.run():
                i, j = self.As.goal
                self.gridWorld[i][j] = 1
                self.containP[i][j] = 0
                self.findingP[i][j] = 0
                self.maxFindingProb = np.amax(self.findingP)
                # self.choiceList = maxProbSortByDis(self.findingP, self.maxFindingProb, self.gridWorld, self.cur)
                choices = chooseByCluster(self.findingP, self.choiceList)
                choices = judgeByDisAs(self.findingP, np.amax(self.findingP), self.gridWorld, self.cur)
                # choices = sortByUtility(self.findingP, self.cur)
                x, y = choices[random.randint(0, len(choices) - 1)]
                self.As = AStar(self.gridWorld, 1)
                self.As.start = self.start
                self.As.goal = (x, y)
            self.normalizeContainP()
            ret = self.followPlan(self.As.trajectory)
            if ret == 0:
                break
            # self.choiceList = maxProbSortByDis(self.findingP, self.maxFindingProb, self.gridWorld, self.cur)
            # choices = chooseByCluster(self.findingP, self.choiceList)
            choices = judgeByDisAs(self.findingP, np.amax(self.findingP), self.gridWorld, self.cur)
            # choices = sortByUtility(self.findingP, self.cur)
            x, y = choices[random.randint(0, len(choices) - 1)]
            self.goal = (x, y)
            self.start = self.cur
            print(np.amax(self.findingP), self.start, self.goal, self.findingP[self.goal[0]][self.goal[1]], self.findingP[self.target[0]][self.target[1]])
        return True

    def agent8Try(self):
        self.cur = self.start
        self.goal = randomInitialize(self.origin.shape[0], self.origin.shape[1], self.gridWorld, False)
        print(self.start, self.goal, self.target)
        while True:
            self.As = AStar(self.gridWorld, 1)
            self.As.start = self.start
            self.As.goal = self.goal
            while not self.As.run():
                i, j = self.As.goal
                self.gridWorld[i][j] = 1
                self.containP[i][j] = 0
                self.findingP[i][j] = 0
                self.maxFindingProb = np.amax(self.findingP)

                self.choiceList = maxProbSortByDis(self.findingP, self.maxFindingProb, self.gridWorld, self.cur)
                choices = self.moveForward(self.choiceList, self.cur)
                x, y = choices[random.randint(0, len(choices) - 1)]
                self.As = AStar(self.gridWorld, 1)
                self.As.start = self.start
                self.As.goal = (x, y)
            self.normalizeContainP()
            ret = self.followPlan(self.As.trajectory)
            if ret == 0:
                break
            self.choiceList = maxProbSortByDis(self.findingP, self.maxFindingProb, self.gridWorld, self.cur)
            choices = self.moveForward(self.choiceList, self.cur)
            x, y = choices[random.randint(0, len(choices) - 1)]
            self.goal = (x, y)
            self.start = self.cur
            # print(np.amax(self.findingP), self.start, self.goal, self.findingP[self.goal[0]][self.goal[1]], self.findingP[self.target[0]][self.target[1]])
        return True

    def tmpExamine(self, tmpContainP, tmpFindingP, point):
        i, j = point
        p = tmpContainP[i][j]
        tmpContainP[i][j] = 0.2 * p / (1 - 0.8 * p)
        for x in range(self.m):
            for y in range(self.n):
                if (x, y) != point:
                    tmpContainP[x][y] = tmpContainP[x][y] / (1 - 0.8 * p)
        for x in range(self.m):
            for y in range(self.n):
                p = tmpContainP[x][y]
                if self.gridWorld[x][y] == 2:
                    tmpFindingP[x][y] = p * 0.5
                else:
                    if self.terrain[x][y] == 0:  # flat
                        tmpFindingP[x][y] = 0.8 * p
                    elif self.terrain[x][y] == 1:  # hilly
                        tmpFindingP[x][y] = 0.5 * p
                    else:  # forest
                        tmpFindingP[x][y] = 0.2 * p

    def moveForward(self, choiceList, start):
        move_list = PriorityQueue()
        for goal in choiceList:
            As = AStar(self.gridWorld, 1)
            As.start = start
            As.goal = goal
            As.run()
            # move1 = len(As.trajectory)
            # move1 = abs(start[0]-goal[0]) + abs(start[1] - goal[1])
            tmpContainP, tmpFindingP = copy.deepcopy(self.containP), copy.deepcopy(self.findingP)
            for (i, j) in As.trajectory:
                if (i, j) == goal:
                    self.tmpExamine(tmpContainP, tmpFindingP, (i, j))
                else:
                    p = tmpContainP[i][j]
                    for x in range(self.m):
                        for y in range(self.n):
                            if self.gridWorld[x][y] == 2:
                                tmpFindingP[x][y] = p * 0.5
                            else:
                                if self.terrain[x][y] == 0:  # flat
                                    tmpFindingP[x][y] = 0.8 * p
                                elif self.terrain[x][y] == 1:  # hilly
                                    tmpFindingP[x][y] = 0.5 * p
                                else:  # forest
                                    tmpFindingP[x][y] = 0.2 * p
            tmpMaxFindProb = np.amax(tmpFindingP)
            choices = maxProbSortByDis(tmpFindingP, tmpMaxFindProb, self.gridWorld, goal)
            x, y = choices[random.randint(0, len(choices) - 1)]
            # move2 = abs(goal[0]-x) + abs(goal[1]-y)
            move_list.put((tmpMaxFindProb, goal))
        first = move_list.get()
        res = [first[1]]
        while not move_list.empty():
            priority, point = move_list.get()
            if first[0] == priority:
                res.append(point)
            else:
                break
        return res


if __name__ == '__main__':
    # allMaze = loadMaze("../maps", "density0.3.json")
    n = 10
    allMaze = loadMaze("./full_connected_maps", "dim50_13.json")
    print(allMaze[0].shape)
    map, terrain = allMaze[0], allMaze[1]
    print(sum(sum(terrain == 0)), sum(sum(terrain == 1)), sum(sum(terrain == 2)))
    timeAgent6, timeAgent7 = 0, 0
    tjtAgent6, tjtAgent7 = 0, 0

    for _ in range(n):
        target = randomInitialize(map.shape[0], map.shape[1], map, True)
        start = randomInitialize(map.shape[0], map.shape[1], map, True)
        # start = (21, 20)
        # target = (43, 21)
        print(terrain[target[0]][target[1]], map[start[0]][start[1]], map[target[0]][target[1]])
        # goal =  (27, 14)

        # terrain = generateTerrain(map.shape[0], map.shape[1])
        # terrain[target[0]][target[1]] = 0

        # agent6 = ProbAgent(map, target)
        # agent6.start = start
        # agent6.terrain = terrain
        #
        # agent7 = ProbAgent(map, target)
        # agent7.start = start
        # agent7.terrain = terrain

        agent8 = ProbAgent(map, target)
        agent8.start = start
        agent8.terrain = terrain

        # agent6.agentType = 6
        # time1 = time.time()
        # agent6.agent6()
        # time2 = time.time()
        # print("agent6 true, time={time}, movement={movement}, examination={examination}, ratio={ratio}".format(time=time2 - time1, movement=len(agent6.trajectory), examination=agent6.examination, ratio=len(agent6.trajectory)/agent6.examination))
        # timeAgent6 += time2 - time1
        # tjtAgent6 += len(agent6.trajectory)
        #
        # agent7.agentType = 7
        # time3 = time.time()
        # agent7.agent7()
        # time4 = time.time()
        # print("agent7 true, time={time}, movement={movement}, examination={examination}, ratio={ratio}".format(time=time4 - time3, movement=len(agent7.trajectory), examination=agent7.examination, ratio=len(agent7.trajectory)/agent7.examination))
        # timeAgent7 += time4 - time3
        # tjtAgent7 += len(agent7.trajectory)

        agent8.agentType = 7
        time5 = time.time()
        agent8.agent8Try()
        time6 = time.time()
        print("agent8 true, time={time}, movement={movement}, examination={examination}, ratio={ratio}".format(time=time6 - time5, movement=len(agent8.trajectory), examination=agent8.examination, ratio=len(agent8.trajectory) / agent8.examination))

        # break
    # print("agent6 time={timeAgent6}, trajectory length={len}".format(timeAgent6=timeAgent6 / n, len=tjtAgent6 / n))
    # print("agent7 time={timeAgent7}, trajectory length={len}".format(timeAgent7=timeAgent7 / n, len=tjtAgent7 / n))
