# encoding=utf-8

import numpy as np

directions = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]


def sense(x, Maze, C):
    blocks = 0
    for i, j in directions:
        if Maze[x[0] + i][x[1] + j] == 1:
            blocks += 1
    return blocks


def isVertex(x, m, n):
    return (x[0] == 0 and x[1] == 0) or (x[0] == 0 and x[1] == n - 1) or (x[0] == m - 1 and x[1] == 0) or (x[0] == m - 1 and x[1] == n - 1)


def isBorder(x, m, n):
    return x[0] == 0 or x[0] == m - 1 or x[1] == 0 or x[1] == n - 1


def isValid(x, m, n):
    return 0 <= x[0] < m and 0 <= x[1] < n


def initializeN(Maze):
    m, n = len(Maze), len(Maze[0])
    N = np.full((m, n), 8)
    B = np.zeros([m, n])
    E = np.zeros([m, n])
    H = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            if isVertex((i, j), m, n):
                N[i][j] = 3
            elif isBorder((i, j), m, n):
                N[i][j] = 5
    return N

def getAllNeighbors(x,m,n):
    neighbors = []
    for dir in directions:
        x1 = x[0]+dir[0]
        y1 = x[1]+dir[1]
        if x1>=0 and x1<m and y1>=0 and y1<n:
            neighbors.append((x1,y1))
    return neighbors
#shen
#grid is the random generated 2d array with probability p
#This function is to calulate the value of C of the original gridworld
def calculateC(grid,C,m,n):
    for i in range(m):
        for j in range(n):
            for nei in getAllNeighbors((i,j),m,n):
                C[i][j]+=grid[nei[0]][nei[1]]
    return C