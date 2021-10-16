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
        self.trick = False
        self.map = map
        self.maze = copy.deepcopy(self.map)
        self.m = map.m
        self.n = map.n
        self.start, self.goal = map.start, map.end
        self.Maze, self.N, self.C, self.B, self.E, self.H = initialize(map.map)
        self.maze.map = self.Maze
        self.visited = set()
        self.trajectory = []
        self.scanArea = max(self.m, self.n)

    def block_update(self, x):
        self.updateByBFS(x)

    def empty_update(self, x):
        self.updateByBFS(x)

    def updateByBFS(self, x):
        Enodes = [x]
        while len(Enodes) > 0:
            cur = Enodes.pop()
            if abs(x[0] - cur[0]) >= self.scanArea and abs(x[1] - cur[1]) >= self.scanArea:
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
        if self.trick:
            self.trick121()
        return block, index

    def run(self, trick=False):
        self.trick = trick
        As = AStar(self.maze, 1)
        while True:
            if not As.run():
                return False
            block, index = self.inference(As)
            # print(block)
            if index == len(As.trajectory):
                # print(self.map.map)
                # print(self.Maze)
                return True
            start = As.trajectory[index]
            As.map.start = start
            As.clear()


if __name__ == '__main__':
    p = 0.2
    for i in range(50):
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
        algo = InferenceSearch(map)
        hasTrick = InferenceSearch(copy.deepcopy(map))
        time1 = time.time()
        # print(algo.run())
        algo.run()
        time2 = time.time()
        print(time2 - time1)
        hasTrick.run(True)
        print(sum(sum(algo.Maze == 2)), sum(sum(hasTrick.Maze == 2)))
        print(len(algo.trajectory), len(hasTrick.trajectory))
        # del algo
        # gc.collect()
