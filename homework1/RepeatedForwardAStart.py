import matplotlib.pyplot as plt
import numpy as np
import queue
import math
from collections import deque
from math import sqrt, fabs


class Maze():

    def __init__(self, dim, prob):
        self.grid = np.random.choice(
            a=[True, False],
            size=(dim, dim),
            p=[prob, 1 - prob])
        self.grid[0][0] = False
        self.grid[dim - 1][dim - 1] = False
        self.dim = dim
        self.prob = prob
        self.start = (0, 0)
        self.goal = ((dim - 1), (dim - 1))
        self.unseen = np.zeros((dim, dim), dtype=bool)

    def set_unseen(self, loc):
        self.unseen[loc[0]][loc[1]] = self.grid[loc[0]][loc[1]]


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
        self.f = self.g + self.h

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
def manhattanDist(curr, goal):
    x1 = curr[0]
    y1 = curr[1]
    x2 = goal[0]
    y2 = goal[1]
    return abs(x1 - x2) + abs(y1 - y2)


def euclideanDist(curr, goal):
    x1 = curr[0]
    y1 = curr[1]
    x2 = goal[0]
    y2 = goal[1]
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def chebyshevDist(curr, goal):
    x1 = curr[0]
    y1 = curr[1]
    x2 = goal[0]
    y2 = goal[1]
    return max(abs(x1 - x2), abs(y1 - y2))


def isValid(point, maze):
    x = point[0]
    y = point[1]
    if (x >= 0 and x < maze.dim and y >= 0 and y < maze.dim):
        return True
    else:
        return False


def generateChildrenUnseen(current, maze):
    children = []
    dir = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for i in range(4):
        new_x = dir[i][0] + current.loc[0]
        new_y = dir[i][1] + current.loc[1]
        newPoint = (new_x, new_y)
        if isValid(newPoint, maze) and maze.unseen[new_x][new_y] == False:
            children.append(MazeNode(newPoint, 10000, 10000))
    # for p in children:
    #     print(p)
    return children


def AStarUnseen(startNode, goalNode, maze):
    fringe = queue.PriorityQueue()
    openList = []
    closedList = []
    path = []

    fringe.put(startNode)
    openList.append(startNode.loc)
    while not fringe.empty():
        current = fringe.get()
        # fringe.remove()
        openList.remove(current.loc)
        if current.loc == goalNode.loc:
            path = []
            while current.parent:
                path.append(current.loc)
                current = current.parent
            path.append(current.loc)
            return path[::-1]
        closedList.append(current.loc)
        for node in generateChildrenUnseen(current, maze):
            if node.loc in closedList:
                continue
            node.parent = current
            if node.loc in openList:
                newG = current.g + 1
                if newG < node.g:
                    node.g = newG
                    MazeNode.set_f_value(node)
                    node.parent = current

            else:
                node.g = current.g + 1
                node.h = euclideanDist(node.loc, goalNode.loc)
                MazeNode.set_f_value(node)
                node.parent = current
                fringe.put(node)
                openList.append(node.loc)
    return path


def canSee(currPos, maze):
    children = []
    dir = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for i in range(4):
        new_x = dir[i][0] + currPos[0]
        new_y = dir[i][1] + currPos[1]
        newPoint = (new_x, new_y)
        if isValid(newPoint, maze):
            children.append(newPoint)
    # for p in children:
    #     print(p)
    # print(children)
    return children


def RepeatedForwardAStar(startNode, goalNode, maze):
    # return a  list of tuples
    finalPath = []
    for position in canSee(startNode.loc, maze):
        Maze.set_unseen(maze, position)
    path = AStarUnseen(startNode, goalNode, maze)
    if len(path) == 0:
        return path
    else:
        while len(path) != 0:

            for i in range(len(path)):
                step = path[i]
                for position in canSee(step, maze):
                    Maze.set_unseen(maze, position)
                if path[i] == goalNode.loc:
                    finalPath = finalPath + path
                    return finalPath
                if maze.grid[path[i][0]][path[i][1]] == True:
                    finalPath = finalPath + path[0:i]
                    newStartPos = (path[i - 1][0], path[i - 1][1])
                    newStartNode = MazeNode(newStartPos, 0, euclideanDist(newStartPos, goalNode.loc))
                    newPath = AStarUnseen(newStartNode, goalNode, maze)
                    path = newPath
                    break
                    # if len(path)==0:
                    #     return path
            # finalPath.append(path)

        return path


def RFAStarByBumping(startNode, goalNode, maze):
    finalPath = []
    path = AStarUnseen(startNode, goalNode, maze)
    if len(path) == 0:
        return path
    else:
        while len(path) != 0:
            for i in range(len(path)):
                step = path[i]
                if path[i] == goalNode.loc:
                    finalPath = finalPath + path
                    return finalPath
                if maze.grid[path[i][0]][path[i][1]] == True:
                    currPos = (path[i][0], path[i][1])
                    Maze.set_unseen(maze, currPos)
                    finalPath = finalPath + path[0:i]
                    newStartPos = (path[i - 1][0], path[i - 1][1])
                    newStartNode = MazeNode(newStartPos, 0, euclideanDist(newStartPos, goalNode.loc))
                    newPath = AStarUnseen(newStartNode, goalNode, maze)
                    path = newPath
                    break
        return path


def BFSUnseen(startNode, goalNode, maze):
    path = []
    fringe = deque()
    visited = []
    fringe.append(startNode)
    visited.append(startNode.loc)

    while len(fringe) > 0:
        currNode = fringe.popleft()
        visited.append(currNode.loc)
        if currNode.loc == goalNode.loc:
            path = []
            while currNode.parent:
                path.append(currNode.loc)
                currNode = currNode.parent
            path.append(currNode.loc)
            return path[::-1]
        for child in generateChildrenUnseen(currNode, maze):
            if child.loc in visited:
                continue
            if child not in fringe:
                child.parent = currNode
                fringe.append(child)
                visited.append(child.loc)
    return path


def RepeatedBFS(startNode, goalNode, maze):
    finalPath = []
    path = BFSUnseen(startNode, goalNode, maze)
    if len(path) == 0:
        return path
    else:
        while len(path) != 0:
            for i in range(len(path)):
                step = path[i]
                if path[i] == goalNode.loc:
                    finalPath = finalPath + path
                    return finalPath
                if maze.grid[path[i][0]][path[i][1]] == True:
                    currPos = (path[i][0], path[i][1])
                    Maze.set_unseen(maze, currPos)
                    finalPath = finalPath + path[0:i]
                    newStartPos = (path[i - 1][0], path[i - 1][1])
                    # we do not need
                    newStartNode = MazeNode(newStartPos, 0, euclideanDist(newStartPos, goalNode.loc))
                    newPath = BFSUnseen(newStartNode, goalNode, maze)
                    path = newPath
                    break
        return path


def drawSolution(path, maze):
    path_map = np.zeros((maze.dim, maze.dim), dtype=bool)
    for p in path:
        path_map[p[0]][p[1]] = True
    plt.imshow(path_map, cmap='Greys', interpolation='nearest')
    plt.show()


def questionNumberSix():
    x = np.linspace(0, 0.33, 33, endpoint=False)
    # y1 is the list of average trajectory length
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    ytest = []
    for p in x:
        # count is number of solvable mazes
        count = 0
        pathLength1 = []
        pathLength2 = []

        for i in range(6):
            maze1 = Maze(11, p)
            goalNode = MazeNode(maze1.goal, 0, 0)
            startNode = MazeNode((0, 0), 0, euclideanDist((0, 0), maze1.goal))
            # path1 is the average trajectory length
            # path2 is Length of Shortest Path in Final Discovered Gridworld

            path1 = RepeatedForwardAStar(startNode, goalNode, maze1)
            path2 = AStarUnseen(startNode, goalNode, maze1)

            if len(path1) != 0:
                pathLength1.append(len(path1))
                count = count + 1
            if len(path2) != 0:
                pathLength2.append(path2)

        y1.append(np.average(pathLength1))
        print(y1)
        ytest.append(np.average(pathLength2))
        print(ytest)

        y2.append(np.average(pathLength1) / float(np.average(pathLength2)))
    print(y1)
    print(y2)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(x, y1, 'g--')
    # plt.show()


if __name__ == '__main__':
    mymaze = Maze(101, 0.18)
    plt.imshow(mymaze.grid, cmap='Greys', interpolation='nearest')
    plt.show()
    goalNode = MazeNode(mymaze.goal, 0, 0)
    startNode = MazeNode((0, 0), 0, euclideanDist((0, 0), mymaze.goal))
    # path = RFAStarByBumping(startNode,goalNode,mymaze)
    path = RepeatedBFS(startNode, goalNode, mymaze)
    # path = RepeatedForwardAStar(startNode,goalNode,mymaze)
    # unseenMap = mymaze.unseen
    # plt.imshow(mymaze.unseen, cmap='Greys', interpolation='nearest')
    # plt.show()
    drawSolution(path, mymaze)
    # questionNumberSix()
