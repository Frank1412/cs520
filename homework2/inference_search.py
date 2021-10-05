# encoding=utf-8

from utils import *
from cs520.homework1.A_star_algo import *
from shen import *


# from fang import *


class InferenceSearch(object):
    def __init__(self, map):
        self.map = map
        self.maze = copy.deepcopy(self.map)
        self.m = map.m
        self.n = map.n
        self.start, self.goal = map.start, map.end
        self.Maze, self.N, self.C, self.B, self.E, self.H = initialize(self.map.map)
        self.maze.map = self.Maze
        self.visited = set()
        self.trajectory = []

    def block_update(self, x):
        for i, j in getAllNeighbors(x, self.m, self.n):
            self.B[i][j] += 1
            self.H[i][j] -= 1

    def update_NoBlock_Sense(self, x):
        if self.C[x[0]][x[1]] == 0:
            for i, j in getAllNeighbors(x, self.m, self.n):
                self.E[i][j] += 1
                self.H[i][j] -= 1
                if self.Maze[i][j] == 2:
                    self.Maze[i][j] = 0
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        self.E[nei[0]][nei[1]] += 1
                        self.H[nei[0]][nei[1]] -= 1
            return
        for i, j in getAllNeighbors(x, self.m, self.n):
            self.E[i][j] += 1
            self.H[i][j] -= 1

    def update_NoBlock_NoSense(self, x):
        if self.C[x[0]][x[1]] == 0:
            neighborsOfX = getAllNeighbors(x, self.m, self.n)
            for i, j in neighborsOfX:
                if self.Maze[i][j] == 2:
                    self.Maze[i][j] = 0
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        self.E[nei[0]][nei[1]] += 1
                        self.H[nei[0]][nei[1]] -= 1

    def updateMaze(self, x, pivot):
        if pivot == 1:  # C=B
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] == 2:
                    # self.visited.add((i, j))
                    self.Maze[i][j] = 0
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        # if nei != x and self.Maze[nei[0]][nei[1]] == 2:
                        self.E[nei[0]][nei[1]] += 1
                        self.H[nei[0]][nei[1]] -= 1

        elif pivot == 0:  # N-C=E
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] == 2:
                    # self.visited.add((i, j))
                    self.Maze[i][j] = 1
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        # if nei != x and self.Maze[nei[0]][nei[1]] == 2:
                        self.B[nei[0]][nei[1]] += 1
                        self.H[nei[0]][nei[1]] -= 1

    def updateByH_0(self, x):
        for i, j in getAllNeighbors(x, self.m, self.n):
            for nei in getAllNeighbors((i, j), self.m, self.n):
                # if nei != x and self.Maze[nei[0]][nei[1]] == 2:
                self.E[nei[0]][nei[1]] += 1
                self.H[nei[0]][nei[1]] -= 1

    def updateAllVisited(self, As):
        # for i, j in self.visited:
        tmp = set()
        for t in range(len(self.trajectory)):
            i, j = self.trajectory[len(self.trajectory) - t - 1]
            if (i, j) not in tmp:
                tmp.add((i, j))
            else:
                continue
            if self.H[i][j] == 0:
                print("H=0", self.N[i][j], self.C[i][j], self.B[i][j], self.E[i][j], self.H[i][j], (i, j))
                continue
            if self.C[i][j] == self.B[i][j]:
                print('C=B', self.N[i][j], self.C[i][j], self.B[i][j], self.E[i][j], self.H[i][j], (i, j))
                self.updateMaze((i, j), 1)
                continue
            if self.N[i][j] - self.C[i][j] == self.E[i][j]:
                print('N-C=E', self.N[i][j], self.C[i][j], self.B[i][j], self.E[i][j], self.H[i][j], (i, j))
                self.updateMaze((i, j), 0)
                continue

    def inference(self, As):
        print(As.trajectory)
        block, index = (), len(As.trajectory)
        for idx, x in enumerate(As.trajectory):
            self.C[x[0]][x[1]] = sense(x, self.map.map, self.m, self.n)
            self.trajectory.append(x)

            if self.map.map[x[0]][x[1]] == 1:
                self.visited.add(x)
                if self.Maze[x[0]][x[1]] == 2:
                    self.block_update(x)
                self.Maze[x[0]][x[1]] = 1
                block, index = x, idx - 1
                # print(x, index)
                break
            else:
                if x in self.visited:
                    continue
                if self.Maze[x[0]][x[1]] == 2:
                    self.Maze[x[0]][x[1]] = 0
                    self.update_NoBlock_Sense(x)
                else:
                    self.update_NoBlock_NoSense(x)
                self.visited.add(x)
                # if x in self.visited:
                #     continue
                # self.visited.add(x)
                # if self.C[x[0]][x[1]] == 0:
                #     if self.Maze[x[0]][x[1]] == 2:
                #         self.update_NoBlock_NoSense(x)  # shenchong
                #     self.E[x[0]][x[1]] = self.N[x[0]][x[1]]
                #     self.B[x[0]][x[1]] = 0
                #     self.H[x[0]][x[1]] = 0
                # else:
                #     if self.Maze[x[0]][x[1]] == 2:
                #         self.update_NoBlock_Sense(x)  # zhangxiangnan
                # self.Maze[x[0]][x[1]] = 0
            self.updateAllVisited(As)
        # self.updateAllVisited(As)
        return block, index

    def run(self):
        As = AStar(self.maze, 1)
        while True:
            print(As.map.map)
            print(self.map.map)
            # print(123)
            if not As.run():
                return False
            block, index = self.inference(As)
            # print(block)
            if index == len(As.trajectory):
                print(self.map.map)
                print(As.map.map)
                return True
            start = As.trajectory[index]
            As.map.start = start
            As.clear()


if __name__ == '__main__':
    p = 0.3
    for i in range(1):
        map = Map(10, 10)
        map.setObstacles(True, p)
        As = AStar(map, 1)
        while True:
            if not As.run():
                map.reset()
                map.setObstacles(True, p)
                As = AStar(map, 1)
            else:
                break
        print(As.trajectory)
        algo = InferenceSearch(map)
        print(algo.run())
