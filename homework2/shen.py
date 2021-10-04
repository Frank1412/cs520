import numpy as np
from utils import *


def getAllNeighbors(x, m, n):
    neighbors = []
    for dir in directions:
        x1 = x[0] + dir[0]
        y1 = x[1] + dir[1]
        if 0 <= x1 < m and 0 <= y1 < n:
            neighbors.append((x1, y1))
    return neighbors


# def getNeighborsOnVertex(x, m, n):
#     neighborsOnVertex = []
#     vertexDirections = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
#     for dir in vertexDirections:
#         x1 = x[0] + dir[0]
#         y1 = x[1] + dir[1]
#         if 0 <= x1 < m and 0 <= y1 < n:
#             neighborsOnVertex.append((x1, y1))
#     return neighborsOnVertex
#
#
# def getNeighborsOnEdge(x, m, n):
#     neighborsOnEdge = []
#     edgeDirections = [[1, 0], [0, 1], [-1, 0], [0, -1]]
#     for dir in edgeDirections:
#         x1 = x[0] + dir[0]
#         y1 = x[1] + dir[1]
#         if 0 <= x1 < m and 0 <= y1 < n:
#             neighborsOnEdge.append((x1, y1))
#     return neighborsOnEdge


# Maze = 0 empty, Maze = 1 blocked, Maze = 2 unconfirmed
def update_NoBlock(x, N, visited, Maze, C, B, E, H, m, n):
    visited[x[0]][x[1]] = True
    Maze[x[0]][x[1]] = 0
    H[x[0]][x[1]] = 0

    neighborsOfX = getAllNeighbors(x, m, n)
    XAndNeighborsOfX = neighborsOfX.append(x)
    for nei in neighborsOfX:
        Maze[nei[0]][nei[1]] = 0
        E[nei[0]][nei[1]]+=1
        H[nei[0]][nei[1]]-=1
        #NOfN means neighbor of neighbor
        for NOfN in getAllNeighbors((nei[0],nei[1]),m,n):
            if NOfN not in XAndNeighborsOfX:
                E[NOfN[0]][NOfN[1]]+=1
                H[NOfN[0]][NOfN[1]]-=1
