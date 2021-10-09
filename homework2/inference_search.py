# encoding=utf-8
import sys

sys.path.append("..")
from utils import *
from homework1.A_star_algo import *
from shen import *
import gc
import time

# from fang import *
sys.setrecursionlimit(10 ** 6)


class InferenceSearch(object):
    def __init__(self, map):
        self.map = map
        self.maze = copy.deepcopy(self.map)
        self.m = map.m
        self.n = map.n
        self.start, self.goal = map.start, map.end
        self.Maze, self.N, self.C, self.B, self.E, self.H = initialize(map.map)
        self.maze.map = self.Maze
        self.visited = set()
        self.trajectory = []

    def block_update(self, x):
        self.updateByBFS(x)

    def empty_update(self, x):
        self.updateByBFS(x)

    def updateCurrent(self, x):
        i, j = x
        if self.H[i][j] == 0:
            return
        if self.C[i][j] == self.B[i][j]:
            self.updateMaze((i, j), 1)
            return
        if self.N[i][j] - self.C[i][j] == self.E[i][j]:
            self.updateMaze((i, j), 0)
            return

    def updateByBFS(self, x):
        Enodes = [x]
        while len(Enodes) > 0:
            cur = Enodes.pop()
            type = self.Maze[cur[0]][cur[1]]
            for i, j in getAllNeighbors(cur, self.m, self.n):
                if type == 1:
                    self.B[i][j] += 1
                    self.H[i][j] -= 1
                if type == 0:
                    self.E[i][j] += 1
                    self.H[i][j] -= 1
            if self.H[cur[0]][cur[1]] == 0:
                continue
            if self.C[cur[0]][cur[1]] == self.B[cur[0]][cur[1]]:
                for i, j in getAllNeighbors(cur, self.m, self.n):
                    if self.Maze[i][j] == 2:
                        self.Maze[i][j] = 0
                        Enodes.append((i, j))
                continue
            if self.N[cur[0]][cur[1]] - self.C[cur[0]][cur[1]] == self.E[cur[0]][cur[1]]:
                for i, j in getAllNeighbors(cur, self.m, self.n):
                    if self.Maze[i][j] == 2:
                        self.Maze[i][j] = 1
                        Enodes.append((i, j))
                continue
        return

    def inference(self, As):
        # print(As.trajectory)
        block, index = (), len(As.trajectory)
        for idx, x in enumerate(As.trajectory):
            self.trajectory.append(x)

            if self.map.map[x[0]][x[1]] == 1:
                self.visited.add(x)
                if self.Maze[x[0]][x[1]] == 2:
                    self.Maze[x[0]][x[1]] = 1
                    self.block_update(x)
                block, index = x, idx - 1
                # print(x, index)
                break
            else:
                self.C[x[0]][x[1]] = sense(x, self.map.map, self.m, self.n)
                if self.Maze[x[0]][x[1]] == 2:
                    self.Maze[x[0]][x[1]] = 0
                    self.empty_update(x)
                self.visited.add(x)
        # self.updateAllVisited(As)
        return block, index

    def run(self):
        As = AStar(self.maze, 1)
        while True:
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
        algo = InferenceSearch(map)
        time1 = time.time()
        print(algo.run())
        time2 = time.time()
        print(time2 - time1)
        del algo
        gc.collect()
