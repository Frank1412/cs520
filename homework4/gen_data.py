# coding = utf-8

from repeatedAstar import *
from Astar import *


def randomStart(map, num):
    res = []
    m, n = map.shape
    while len(res) < 20:
        i = random.randint(0, m-1)
        j = random.randint(0, n-1)
        As = AStar(map, 1)
        As.start = (i, j)
        if (i, j) != (0, 0) and (i, j) != (29, 29) and map[i][j] != 1 and (i, j) not in res and As.run():
            res.append((i, j))
            res += 1
    return res


if __name__ == '__main__':
    total_num, p = 20, 0.3
    m, n = 30, 30
    mazes = np.load("maps/30x30dim.npy")
    for i in range(len(mazes)):
        map = mazes[i]
        choices = randomStart(map, 20)
        for (x, y) in choices:
            As = AStar(map, 1)
            As.start = (x, y)
            As.run()
            label = As.trajectory[1]

        print(map)
