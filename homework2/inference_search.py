# encoding=utf-8

from utils import *
from cs520.homework1.A_star_algo import *


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
        self.Maze[x[0]][x[1]] = 1
        self.E[x[0]][x[1]] = self.N[x[0]][x[1]] - self.C[x[0]][x[1]]
        # H[x[0]][x[1]] = updateH(x, Maze, C[x[0]][x[1]], H, m, n)
        for i, j in getAllNeighbors(x, self.m, self.n):
            self.H[i][j] -= 1
            self.B[i][j] += 1
        return

    def updateMaze(self, x, pivot):
        if pivot == 1:
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] != 1:
                    self.Maze[i][j] = 0
        elif pivot == 0:
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] != 0:
                    self.Maze[i][j] = 1

    def updateAllVisited(self):
        for i, j in self.visited:
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
            self.trajectory.append(x)
            self.C[x[0]][x[1]] = sense(x, self.map.map, self.C)
            self.visited.add(x)

            if self.map.map[x[0]][x[1]] == 1:
                self.block_update(x)
                block, index = x, idx - 1
                break
            else:
                if self.C[x[0]][x[1]] == 0:
                    print()  # shenchong
                else:
                    print()  # zhangxiangnan
        self.updateAllVisited()
        return block, index

    def run(self):
        As = AStar(self.maze, 1)
        while True:
            # print(123)
            if not As.run():
                return False
            block, index = self.inference(As)
            print(block)
            if index == len(As.trajectory):
                return True
            start = As.trajectory[index]
            As.map.start = start


if __name__ == '__main__':
    map = Map(101, 101)
    infer_algo = InferenceSearch(map)
    As = AStar(map, 1)
    p = 0.2
    for i in range(1):
        map.setObstacles(True, p)
        while True:
            if not infer_algo.run():
                map.reset()
                map.setObstacles(True, p)
                algo = AStar(map, 1)
            else:
                break
        print(infer_algo.run())