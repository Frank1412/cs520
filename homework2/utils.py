# encoding=utf-8


directions = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]


def sense(x, Maze, C):
    blocks = 0
    for i, j in directions:
        if Maze[x[0] + i][x[1] + j] == 1:
            blocks += 1
    return blocks
