# encoding=utf-8
import sys
from util import *
import gc
import time
from Astar import *

sys.path.append("..")
sys.setrecursionlimit(10 ** 6)

directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
cls = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


class InferenceSearch(object):
    def __init__(self, map):
        self.trick = False  # False---agent3  True---agent4
        self.map = map
        self.m, self.n = map.shape
        self.start = (0, 0)
        self.goal = (self.m-1, self.n-1)
        self.Maze, self.N, self.C, self.B, self.E, self.H = initialize(map)
        self.subMaze, self.subN, self.subC, self.subB, self.subE, self.subH = initialize(map)
        self.visited = set()
        self.trajectory = []
        self.scanArea = self.m + self.n
        self.prev = None
        self.dataX = []
        self.dataY = []

    def block_update(self, x):
        self.updateByBFS(x)

    def empty_update(self, x):
        self.updateByBFS(x)

    def updateByBFS(self, x):
        Enodes = [x]
        while len(Enodes) > 0:
            cur = Enodes.pop()
            if abs(x[0] - cur[0]) + abs(x[1] - cur[1]) > self.scanArea:
                continue
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

    def trick121(self):
        length = len(self.trajectory)

        """121"""
        for i in range(1, length - 1):
            lst = [self.trajectory[i - 1], self.trajectory[i], self.trajectory[i + 1]]
            if self.Maze[lst[0][0]][lst[0][1]] == 1 or self.Maze[lst[1][0]][lst[1][1]] == 1 or self.Maze[lst[2][0]][lst[2][1]] == 1:
                continue
            if self.H[lst[0][0]][lst[0][1]] != 0 and self.H[lst[1][0]][lst[1][1]] != 0 and self.H[lst[2][0]][lst[2][1]] != 0:
                if len(set(lst)) == 3:
                    pre, cur, next = lst[0], lst[1], lst[2]
                    if self.C[pre[0]][pre[1]] - self.B[pre[0]][pre[1]] == 1 and self.C[cur[0]][cur[1]] - self.B[cur[0]][cur[1]] == 2 and self.C[next[0]][next[1]] - self.B[next[0]][next[1]] == 1:
                        if pre[0] == next[0]:
                            if cur[0] - 1 >= 0 and self.Maze[cur[0] - 1][cur[1]] == 2:
                                self.Maze[cur[0] - 1][cur[1]] = 0
                                self.updateByBFS((cur[0] - 1, cur[1]))
                            if cur[0] + 1 < self.m and self.Maze[cur[0] + 1][cur[1]] == 2:
                                self.Maze[cur[0] + 1][cur[1]] = 0
                                self.updateByBFS((cur[0] + 1, cur[1]))
                        if pre[1] == next[1]:
                            if cur[1] - 1 >= 0 and self.Maze[cur[0]][cur[1] - 1] == 2:
                                self.Maze[cur[0]][cur[1] - 1] = 0
                                self.updateByBFS((cur[0], cur[1] - 1))
                            if cur[1] + 1 < self.n and self.Maze[cur[0]][cur[1] + 1] == 2:
                                self.Maze[cur[0]][cur[1] + 1] = 0
                                self.updateByBFS((cur[0], cur[1] + 1))

        """1221"""
        for i in range(length - 3):
            lst = [self.trajectory[i], self.trajectory[i + 1], self.trajectory[i + 2], self.trajectory[i + 3]]
            if self.Maze[lst[0][0]][lst[0][1]] == 1 or self.Maze[lst[1][0]][lst[1][1]] == 1 or self.Maze[lst[2][0]][lst[2][1]] == 1 or self.Maze[lst[3][0]][lst[3][1]] == 1:
                continue
            if self.H[lst[0][0]][lst[0][1]] != 0 and self.H[lst[1][0]][lst[1][1]] != 0 and self.H[lst[2][0]][lst[2][1]] != 0 and self.H[lst[3][0]][lst[3][1]] != 0:
                if len(set(lst)) == 4:
                    node1, node2, node3, node4 = lst[0], lst[1], lst[2], lst[3]
                    if self.C[node1[0]][node1[1]] - self.B[node1[0]][node1[1]] == 1 and self.C[node2[0]][node2[1]] - self.B[node2[0]][node2[1]] == 2 and self.C[node3[0]][node3[1]] - self.B[node3[0]][node3[1]] == 2 and \
                            self.C[node4[0]][node4[1]] - self.B[node4[0]][node4[1]] == 1:
                        # if self.C[pre[0]][pre[1]] and self.C[cur[0]][cur[1]] == 2 and self.C[next[0]][next[1]] == 1:
                        if node1[0] == node2[0] == node3[0] == node4[0]:
                            for i, j in directions:
                                if j <= 0:
                                    if 0 <= node1[0] + i < self.m and node1[1] + j >= 0 and self.Maze[node1[0] + i][node1[1] + j] == 2:
                                        self.Maze[node1[0] + i][node1[1] + j] = 0
                                        self.updateByBFS((node1[0] + i, node1[1] + j))
                            for i, j in directions:
                                if j >= 0:
                                    if 0 <= node4[0] + i < self.m and node4[1] + j < self.n and self.Maze[node4[0] + i][node4[1] + j] == 2:
                                        self.Maze[node4[0] + i][node4[1] + j] = 0
                                        self.updateByBFS((node4[0] + i, node4[1] + j))
                        if node1[1] == node2[1] == node3[1] == node4[1]:
                            for i, j in directions:
                                if i <= 0:
                                    if node1[0] + i >= 0 and 0 <= node1[1] + j < self.n and self.Maze[node1[0] + i][node1[1] + j] == 2:
                                        self.Maze[node1[0] + i][node1[1] + j] = 0
                                        self.updateByBFS((node1[0] + i, node1[1] + j))
                            for i, j in directions:
                                if i >= 0:
                                    if node4[0] + i < self.m and 0 <= node4[1] + j < self.n and self.Maze[node4[0] + i][node4[1] + j] == 2:
                                        self.Maze[node4[0] + i][node4[1] + j] = 0
                                        self.updateByBFS((node4[0] + i, node4[1] + j))

    def gen_data(self, next):
        x = np.eye(3)[self.subMaze]
        place = np.zeros(self.subMaze.shape)
        # print(prev)
        place[self.prev[0]][self.prev[1]] = 1
        place = np.expand_dims(place, 2)
        # N = np.expand_dims(self.subN, 2)
        C = np.expand_dims(self.subC, 2)
        B = np.expand_dims(self.subB, 2)
        E = np.expand_dims(self.subE, 2)
        H = np.expand_dims(self.subH, 2)
        # x = np.concatenate([x, place, N, C, B, E, H], axis=-1)
        x = np.concatenate([x, place, C, B, E, H], axis=-1)
        label = cls.get((next[0] - self.prev[0], next[1] - self.prev[1]))
        return np.array(x), label

    def inference(self, As):
        # print(As.trajectory)
        block, index = (), len(As.trajectory)
        m, n = self.Maze.shape
        for idx, x in enumerate(As.trajectory):
            self.trajectory.append(x)
            if idx != 0:
                sample = self.gen_data(x)
                self.dataX.append(sample[0])
                self.dataY.append(sample[1])
            if self.map[x[0]][x[1]] == 1:
                self.visited.add(x)
                if self.Maze[x[0]][x[1]] == 2:
                    self.Maze[x[0]][x[1]] = 1
                    self.subMaze[x[0]][x[1]] = 1
                    for (i, j) in get8Neighbors(x, m, n):
                        self.subB[i][j] += 1
                        self.subH[i][j] -= 1
                    self.block_update(x)
                block, index = x, idx - 1
                # print(x, index)
                break
            else:
                self.C[x[0]][x[1]] = sense(x, self.map)
                self.subC[x[0]][x[1]] = self.C[x[0]][x[1]]
                if self.Maze[x[0]][x[1]] == 2:
                    self.Maze[x[0]][x[1]] = 0
                    for (i, j) in get8Neighbors(x, m, n):
                        self.subE[i][j] += 1
                        self.subH[i][j] -= 1
                    self.empty_update(x)
                self.visited.add(x)
                self.prev = x
        if self.trick:
            self.trick121()
        return block, index

    def run(self, trick=False):
        self.trick = trick
        As = AStar(self.Maze, 1)
        As.start = self.start
        As.goal = self.goal
        while True:
            if not As.run():
                print(As.trajectory)
                return False
            block, index = self.inference(As)
            # print(block, index, len(As.trajectory))
            # print(block)
            if index == len(As.trajectory):
                # print(self.map.map)
                # print(self.Maze)
                return True
            start = As.trajectory[index]
            As = AStar(self.Maze, 1)
            As.start = start
            As.goal = self.goal


if __name__ == '__main__':
    mazes = np.load("maps/test_30x30dim.npy")
    dataX, dataY = np.zeros([1, 30, 30, 8]), np.array([10])
    lenList = []
    print(mazes.shape)
    for i in range(len(mazes[:])):
        map = mazes[i]
        algo = InferenceSearch(map)
        res = algo.run()
        print("{i}th len=".format(i=i+1), len(algo.trajectory))
        lenList.append(len(algo.trajectory))
        x = np.array(algo.dataX)
        y = np.array(algo.dataY)
        dataX = np.concatenate([dataX, x], axis=0)
        dataY = np.concatenate([dataY, y], axis=0)

    # np.save("./data/proj2/test_dataX", dataX[1:])
    # np.save("./data/proj2/test_dataY", dataY[1:])
    # np.save("./pics/proj2/agentLen", np.array(lenList))
