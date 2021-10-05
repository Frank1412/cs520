# encoding=utf-8
import sys

sys.path.append("..")
from utils import *
from homework1.A_star_algo import *
from shen import *


# from fang import *


class InferenceSearch(object):
    def __init__(self, map, init):
        self.map = map
        self.maze = copy.deepcopy(self.map)
        self.m = map.m
        self.n = map.n
        self.start, self.goal = map.start, map.end
        self.Maze, self.N, self.C, self.B, self.E, self.H = init
        self.maze.map = self.Maze
        self.visited = set()
        self.trajectory = []

    def block_update(self, x):
        for i, j in getAllNeighbors(x, self.m, self.n):
            self.B[i][j] += 1
            self.H[i][j] -= 1
            self.updateCurrent((i, j))
        # self.B[i][j] += 1
        # self.H[i][j] -= 1
        # self.updateCurrent((i, j))

    def empty_update(self, x):
        for i, j in getAllNeighbors(x, self.m, self.n):
            self.E[i][j] += 1
            self.H[i][j] -= 1
            self.updateCurrent((i, j))

    def updateMaze(self, x, pivot):
        if pivot == 1:  # C=B
            nums = countB(x, self.Maze, self.m, self.n, pivot)
            if nums != self.B[x[0]][x[1]]:
                print("error", nums, self.B[x[0]][x[1]])
                return
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] == 2:
                    self.Maze[i][j] = 0
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        self.E[nei[0]][nei[1]] += 1
                        self.H[nei[0]][nei[1]] -= 1
                        self.updateCurrent(nei)

        elif pivot == 0:  # N-C=E
            nums = countB(x, self.Maze, self.m, self.n, pivot)
            if nums != self.E[x[0]][x[1]]:
                print("error", nums)
                return
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] == 2:
                    self.Maze[i][j] = 1
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        self.B[nei[0]][nei[1]] += 1
                        self.H[nei[0]][nei[1]] -= 1
                        self.updateCurrent(nei)

    def updateAllVisited(self, As):
        # for i, j in self.visited:
        tmp = set()
        for t in range(len(self.trajectory)):
            i, j = self.trajectory[len(self.trajectory) - t - 1]
            # if (i, j) not in tmp:
            #     tmp.add((i, j))
            # else:
            #     continue
            self.updateCurrent((i, j))

    def updateCurrent(self, x):
        i, j = x
        if self.H[i][j] == 0:
            # print("H=0", self.N[i][j], self.C[i][j], self.B[i][j], self.E[i][j], self.H[i][j], (i, j))
            num1 = countB((i, j), self.Maze, self.m, self.n, 0)
            num2 = countB((i, j), self.Maze, self.m, self.n, 1)
            # print(num1 + num2, self.N[i][j] - num1 - num2)
            return
        if self.C[i][j] == self.B[i][j]:
            # print('C=B', self.N[i][j], self.C[i][j], self.B[i][j], self.E[i][j], self.H[i][j], (i, j))
            self.updateMaze((i, j), 1)
            # print('after C=B', self.N[i][j], self.C[i][j], self.B[i][j], self.E[i][j], self.H[i][j], (i, j))
            return
        if self.N[i][j] - self.C[i][j] == self.E[i][j]:
            # print('N-C=E', self.N[i][j], self.C[i][j], self.B[i][j], self.E[i][j], self.H[i][j], (i, j))
            self.updateMaze((i, j), 0)
            # print('after N-C=E', self.N[i][j], self.C[i][j], self.B[i][j], self.E[i][j], self.H[i][j], (i, j))
            return

    def inference(self, As):
        # print(As.trajectory)
        block, index = (), len(As.trajectory)
        for idx, x in enumerate(As.trajectory):
            self.C[x[0]][x[1]] = sense(x, self.map.map, self.m, self.n)
            self.trajectory.append(x)

            if self.map.map[x[0]][x[1]] == 1:
                self.visited.add(x)
                if self.Maze[x[0]][x[1]] == 2:
                    self.Maze[x[0]][x[1]] = 1
                    self.block_update(x)
                self.Maze[x[0]][x[1]] = 1
                block, index = x, idx - 1
                # print(x, index)
                break
            else:
                if self.Maze[x[0]][x[1]] == 2:
                    self.Maze[x[0]][x[1]] = 0
                    self.empty_update(x)
                self.Maze[x[0]][x[1]] = 0
                self.visited.add(x)
        # self.updateAllVisited(As)
        return block, index

    def run(self):
        As = AStar(self.maze, 1)
        while True:
            # print(As.map.map)
            # print(self.map.map)
            # print(self.C)
            # print(123)
            if not As.run():
                return False
            block, index = self.inference(As)
            # print(block)
            if index == len(As.trajectory):
                # print(self.map.map)
                # print(As.map.map)
                return True
            start = As.trajectory[index]
            As.map.start = start
            As.clear()


if __name__ == '__main__':
    p = 0.3
    for i in range(10):
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
        # print(As.trajectory)
        init = initialize(map.map)
        # print(init[2], map)
        algo = InferenceSearch(map, init)
        print(algo.run())
