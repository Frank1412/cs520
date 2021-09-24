import matplotlib.pyplot as plt
import numpy as np
import queue
import math
import time
import timeit
from math import sqrt, fabs



class Maze():

    def __init__(self, dim, prob,heuristic):
        self.grid = np.random.choice(
                        a=[True, False],
                        size=(dim, dim),
                        p=[prob, 1-prob])
        self.grid[0][0] = False
        self.grid[dim-1][dim-1] = False
        self.dim = dim
        self.prob = prob
        self.start = (0,0)
        self.heuristic = heuristic
        self.goal = ((dim-1),(dim-1))



class MazeNode():
    def __init__(self, loc, g, h):
        self.g = g
        self.h = h
        self.f = g + h
        self.loc = loc
        self.valid = True
        self.blocked = 0
        self.parent = None

    def set_f_value(self):
        self.f = self.g+ self.h

    def __lt__(self, other):
        return self.f < other.f

    def set_invalid(self):
        self.valid = False

    def is_valid(self):
        return self.valid

# def ceshi(mazeNode1,mazeNode2,mazeNode3):
#     fringe = queue.PriorityQueue()
#     fringe.put(mazeNode1)
#     fringe.put(mazeNode2)
#     fringe.put(mazeNode3)
#     while not fringe.empty():
#         current = fringe.get()
#         print(current.f)
def calculateHeuristic(maze,curr,goal):
    if maze.heuristic=="euclidean":
        return euclideanDist(curr,goal)
    elif maze.heuristic == "manhattan":
        return manhattanDist(curr,goal)
    elif maze.heuristic == "chebyshev":
        return chebyshevDist(curr,goal)

def manhattanDist(curr,goal):
    x1 = curr[0]
    y1 = curr[1]
    x2 = goal[0]
    y2 = goal[1]
    return abs(x1 - x2) + abs(y1 - y2)
def euclideanDist(curr,goal):
    x1 = curr[0]
    y1 = curr[1]
    x2= goal[0]
    y2 = goal[1]
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
def chebyshevDist(curr,goal):
    x1 = curr[0]
    y1 = curr[1]
    x2 = goal[0]
    y2 = goal[1]
    return max(abs(x1 - x2), abs(y1 - y2))
def isValid(point,maze):
    x = point[0]
    y = point[1]
    if(x>=0 and x<maze.dim and y>=0  and y<maze.dim):
        return True
    else:
        return False
def generateChildren(current,maze):
    children = []
    dir = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for i in range(4):
        new_x = dir[i][0]+current.loc[0]
        new_y = dir[i][1]+current.loc[1]
        newPoint = (new_x,new_y)
        if isValid(newPoint,maze) and maze.grid[new_x][new_y]==False:
            children.append(MazeNode(newPoint,10000,10000))
    # for p in children:
    #     print(p)
    return children


def AStar(start,goal,maze):
    fringe = queue.PriorityQueue()
    openList = {(-1,-1)}
    closedList = {(-1,-1)}
    path = []
    startNode = MazeNode(start,0,calculateHeuristic(maze,start,goal))

    fringe.put(startNode)
    openList.add(startNode.loc)
    while not fringe.empty():
        current = fringe.get()
        # fringe.remove()
        openList.remove(current.loc)
        if current.loc == goal:
            path = []
            while current.parent:
                path.append(current.loc)
                current = current.parent
            path.append(current.loc)
            return path[::-1]
        closedList.add(current.loc)
        for node in generateChildren(current,maze):
            if node.loc in closedList:
                continue
            node.parent = current
            if node.loc in openList:
                newG = current.g+1
                if newG<node.g:
                    node.g = newG
                    MazeNode.set_f_value(node)
                    node.parent = current

            else:
                node.g = current.g+1
                node.h = calculateHeuristic(maze,node.loc,goal)
                MazeNode.set_f_value(node)
                node.parent = current
                fringe.put(node)
                openList.add(node.loc)
    return path

def drawSolution(path,maze):
    path_map = np.zeros((maze.dim, maze.dim), dtype=bool)
    for p in path:
        path_map[p[0]][p[1]] = True
    plt.imshow(path_map, cmap='Greys', interpolation='nearest')
    plt.show()
# 用 dictionary 优化
def densityVSSolvability():
    x = np.linspace(0,1,20,endpoint=False)
    y = []
    for p in x:
        count = 0
        for i in range(1,11):
            mymaze = Maze(101,p)
            path = AStar(mymaze.start,mymaze.goal,mymaze)
            if len(path)!=0:
                count = count+1
        y.append(count/float(10))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x,y,'g--')
    plt.show()
def bestHeuristic():
    x = np.linspace(0,0.4,30,endpoint=False)
    y1 = []
    y2 = []
    y3 = []
    t1 = []
    t2 = []
    t3 = []



    for p in x:
        count = 0
        timeList = []
        pathLength = []
        while count <10:
            mymaze = Maze(51,p,"manhattan")
            tic = time.time()
            path = AStar(mymaze.start,mymaze.goal,mymaze)
            toc = time.time()
            timeList.append(toc-tic)
            if len(path)!=0:
                count = count + 1
                pathLength.append(len(path))
        y1.append(np.average(pathLength))
        t1.append(np.average(timeList))

    for p in x:
        count = 0
        pathLength = []
        while count <10:
            mymaze = Maze(51,p,"euclidean")
            path = AStar(mymaze.start,mymaze.goal,mymaze)
            if len(path)!=0:
                count = count + 1
                pathLength.append(len(path))
        y2.append(np.average(pathLength))

    for p in x:
        count = 0
        pathLength = []
        while count <10:
            mymaze = Maze(51,p,"chebyshev")
            path = AStar(mymaze.start,mymaze.goal,mymaze)
            if len(path)!=0:
                count = count + 1
                pathLength.append(len(path))
        y3.append(np.average(pathLength))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(x, y1, 'g')
    # ax.plot(x,y2,'r')
    # ax.plot(x,y3,'b')
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, t1, 'g')
    plt.show()












    # openList.append(MazeNode((0,0),0,0))
    # print(openList[0].loc[0])
    # print(openList[0].loc[1])


if __name__ == '__main__':

    mymaze = Maze(101,0.1,"euclidean")

    # plt.imshow(mymaze.grid, cmap='Greys', interpolation='nearest')
    # plt.show()
    # start = time.time()
    # path =AStar(mymaze.start,mymaze.goal,mymaze)
    # end = time.time()
    # print(end - start)
    # drawSolution(path,mymaze)
    # densityVSSolvability()
    bestHeuristic()








