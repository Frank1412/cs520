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
    def __init__(self, map, p):
        self.map = map
        self.maze = copy.deepcopy(self.map)
        self.m = map.m
        self.n = map.n
        self.start, self.goal = map.start, map.end
        self.Maze, self.N, self.C, self.B, self.E, self.H = initialize(map.map)
        self.maze.map = self.Maze
        self.P = np.full([self.m, self.n], p)
        self.p = p
        self.visited = set()
        self.trajectory = []

    def setBlock(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.Maze[i][j]==2 and self.P[i][j] > 0.7:
                    self.Maze[i][j] = 1
                    self.updateByStack((i, j))

    def block_update(self, x):
        for i, j in getAllNeighbors(x, self.m, self.n):
            self.B[i][j] += 1
            self.H[i][j] -= 1
            self.updateByStack((i, j))
        for i, j in getAllNeighbors(x, self.m, self.n):
            if (i,j) not in visited:
                self.P[i][j] = (self.C[i][j]-self.B[i][j])*1.0/self.H[i][j]

    def empty_update(self, x):
        for i, j in getAllNeighbors(x, self.m, self.n):
            self.E[i][j] += 1
            self.H[i][j] -= 1
            self.updateByStack((i, j))
        for i, j in getAllNeighbors(x, self.m, self.n):
            if (i,j) not in visited:
                self.P[i][j] = (self.C[i][j]-self.B[i][j])*1.0/self.H[i][j]

    def updateMaze(self, x, pivot):
        if pivot == 1:  # C=B
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] == 2:
                    self.Maze[i][j] = 0
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        self.E[nei[0]][nei[1]] += 1
                        self.H[nei[0]][nei[1]] -= 1
                        self.updateByStack(nei)

        elif pivot == 0:  # N-C=E
            for i, j in getAllNeighbors(x, self.m, self.n):
                if self.Maze[i][j] == 2:
                    self.Maze[i][j] = 1
                    for nei in getAllNeighbors((i, j), self.m, self.n):
                        self.B[nei[0]][nei[1]] += 1
                        self.H[nei[0]][nei[1]] -= 1
                        self.updateByStack(nei)

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

    def updateByStack(self, x):
        Enodes = []
        if self.H[x[0]][x[1]] == 0:
            return
        node = x
        sublist = []
        while len(Enodes) > 0:
            for i, j in getAllNeighbors(node, self.m, self.n):
                if self.Maze[i][j] == 2:
                    self.Maze[i][j] = 0
                    neilist = getAllNeighbors((i, j), self.m, self.n)
                    for t in range(len(neilist) - 1, -1, -1):
                        if self.H[node[0]][node[1]] == 0:
                            continue
                        Enodes.append(neilist[t])
                        if self.C[node[0]][node[1]] == self.B[node[0]][node[1]]:
                            sublist.append(0)
                            continue
                        if self.N[node[0]][node[1]] - self.C[node[0]][node[1]] == self.E[node[0]][node[1]]:
                            sublist.append(1)
            node = Enodes.pop()
            types = sublist.pop()
            if types == 0:
                self.E[node[0]][node[1]] += 1
            else:
                self.B[node[0]][node[1]] += 1
            self.H[node[0]][node[1]] -= 1

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
                # if self.Maze[x[0]][x[1]] == 2:

                self.Maze[x[0]][x[1]] = 0
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
                # print(self.map.map)
                # print(As.map.map)
                return True
            start = As.trajectory[index]
            As.map.start = start
            As.clear()


if __name__ == '__main__':
    p = 0.3
    for i in range(10):
        map = Map(101, 101)
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
        algo = InferenceSearch(map, p)
        time1 = time.time()
        print(algo.run())
        time2 = time.time()
        print(time2 - time1)
        del algo
        gc.collect()
