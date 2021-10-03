import numpy as np
from utils import *

def getAllNeighbors(x,m,n):
    neighbors = []
    for dir in directions:
        x1 = x[0]+dir[0]
        y1 = x[1]+dir[1]
        if x1>=0 and x1<m and y1>=0 and y1<n:
            neighbors.append((x1,y1))
    return neighbors
def getNeighborsOnVertex(x,m,n):
    neighborsOnVertex = []
    vertexDirections = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    for dir in vertexDirections:
        x1 = x[0] + dir[0]
        y1 = x[1] + dir[1]
        if x1 >= 0 and x1 < m and y1 >= 0 and y1 < n:
            neighborsOnVertex.append((x1,y1))
    return neighborsOnVertex

def getNeighborsOnEdge(x,m,n):
    neighborsOnEdge = []
    edgeDirections = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    for dir in edgeDirections:
        x1 = x[0] + dir[0]
        y1 = x[1] + dir[1]
        if x1 >= 0 and x1 < m and y1 >= 0 and y1 < n:
            neighborsOnEdge.append((x1,y1))
    return neighborsOnEdge

#Maze = 0 empty, Maze = 1 blocked, Maze = 2 unconfirmed
def update_NoBlock(x,N,visited,Maze,C,B,E,H,m,n):
    visited[x[0]][x[1]] = True
    Maze[x[0]][x[1]] = 0
    C[x[0]][x[1]] = 0
    B[x[0]][x[1]] = 0
    E[x[0]][x[1]] = N[x[0]][x[1]]
    H[x[0]][x[1]] = 0
    #update information of each neighbor of x

    for nei in getNeighborsOnVertex(x,m,n):
        E[nei[0]][nei[1]]  = max(E[nei[0]][nei[1]],3)
        H[nei[0]][nei[1]] = min(5,H[nei[0]][nei[1]])
    for nei in getNeighborsOnEdge(x,m,n):
        E[nei[0]][nei[1]] = max(E[nei[0]][nei[1]], 5)
        H[nei[0]][nei[1]] = min(3, H[nei[0]][nei[1]])
    for nei in getAllNeighbors(x,m,n):
        Maze[nei[0]][nei[1]] = 0







