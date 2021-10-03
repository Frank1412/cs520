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

def update_NoBlock(x,N,visited,Maze,C,B,E,H):


