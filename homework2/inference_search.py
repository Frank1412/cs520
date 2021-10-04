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
        """
        when we meet block, update by latest information
        :param N: the number of neighbors cell x has.
        :param visited: Whether or not cell x has been visited.
        :param Maze: Whether or not cell x has been confirmed as empty or blocked, or is currently unconfirmed.
        :param C: the number of neighbors of x that are sensed to be blocked.
        :param B: the number of neighbors of x that have been confirmed to be blocked.
        :param E: the number of neighbors of x that have been confirmed to be empty.
        :param H: the number of neighbors of x that are still hidden or unconfirmed either way.
        :return:
        """
        # self.Maze[x[0]][x[1]] = 1
        for i, j in getAllNeighbors(x, self.m, self.n):
            self.H[i][j] -= 1
            self.B[i][j] += 1

    def update_NoBlock_Sense(self, x):
        # self.Maze[x[0]][x[1]] = 0
        for i, j in getAllNeighbors(x, self.m, self.n):
            self.E[i][j] += 1
            self.H[i][j] -= 1

    def update_NoBlock_NoSense(self, x):
        # self.Maze[x[0]][x[1]] = 0
        self.H[x[0]][x[1]] = 0
        for nei in getNeighborsOnVertex(x, self.m, self.n):
            self.E[nei[0]][nei[1]] = max(self.E[nei[0]][nei[1]], 3)
            self.H[nei[0]][nei[1]] = min(5, self.H[nei[0]][nei[1]])
        for nei in getNeighborsOnEdge(x, self.m, self.n):
            self.E[nei[0]][nei[1]] = max(self.E[nei[0]][nei[1]], 5)
            self.H[nei[0]][nei[1]] = min(3, self.H[nei[0]][nei[1]])
        for nei in getAllNeighbors(x, self.m, self.n):
            self.Maze[nei[0]][nei[1]] = 0

    def updateMaze(self, x, pivot):
        if pivot == 1:
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] == 2:
                    self.Maze[i][j] = 0
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        if self.H[nei[0]][nei[1]] != 0:
                            self.E[nei[0]][nei[1]] += 1
                            self.H[nei[0]][nei[1]] -= 1

        elif pivot == 0:
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] == 2:
                    self.Maze[i][j] = 1
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        if self.H[nei[0]][nei[1]] != 0:
                            self.B[nei[0]][nei[1]] += 1
                            self.H[nei[0]][nei[1]] -= 1

    def updateAllVisited(self):
        # for i, j in self.visited:
        for t in range(len(self.trajectory) - 1, -1, -1):
            i, j = self.trajectory[t]
            if self.H[i][j] == 0:
                continue
            if self.C[i][j] == self.B[i][j]:
                self.updateMaze((i, j), 1)
                continue
            if self.N[i][j] - self.C[i][j] == self.E[i][j]:
                self.updateMaze((i, j), 0)

    def inference(self, As):
        block, index = (), len(As.trajectory)
        for idx, x in enumerate(As.trajectory):
            print(x)
            self.trajectory.append(x)
            self.C[x[0]][x[1]] = sense(x, self.map.map, self.m, self.n)
            self.visited.add(x)

            if self.map.map[x[0]][x[1]] == 1:
                self.Maze[x[0]][x[1]] = 1
                self.block_update(x)
                block, index = x, idx - 1
                print(x, index)
                break
            else:
                self.Maze[x[0]][x[1]] = 0
                if self.C[x[0]][x[1]] == 0:
                    self.update_NoBlock_NoSense(x)  # shenchong
                else:
                    self.update_NoBlock_Sense(x)  # zhangxiangnan
        self.updateAllVisited()
        return block, index

    def run(self):
        As = AStar(self.maze, 1)
        while True:
            # print(123)
            if not As.run():
                return False
            block, index = self.inference(As)
            # print(block)
            if index == len(As.trajectory):
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
