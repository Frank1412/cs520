# encoding=utf-8

from utils import *
from cs520.homework1.A_star_algo import *


# from fang import *


class InferenceSearch(object):
    def __init__(self, map):
        self.map = map.map
        self.m = map.m
        self.n = map.n
        self.start, self.goal = map.start, map.end
        self.Maze, self.N, self.C, self.B, self.E, self.H = initialize(self.map)
        self.visited = set()

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
            self.B[i][j] += 1
        return

    def run(self):

        As = AStar(self.Maze, 1)
        while True:
            As.run()
            path = As.trajectory
            for x in path:
                if x==self.map.end
                self.C[x[0]][x[1]] = sense(x, self.map, self.C)
                self.visited.add(x)
                if self.map[x[0]][x[1]] == 1:
                    self.block_update(x)
                    break
                else:
                    if self.C[x[0]][x[1]]==0:
                        print() # shenchong
                    else:
                        print() #zhangxiangnan
            updateAllVisited()