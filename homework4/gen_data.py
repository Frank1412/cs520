# coding = utf-8

from repeatedAstar import *
from Astar import *


if __name__ == '__main__':
    total_num, p = 20, 0.3
    m, n = 30, 30
    mazes = np.load("maps/30x30dim.npy")
    map1 = mazes[0]

    print(map1)

