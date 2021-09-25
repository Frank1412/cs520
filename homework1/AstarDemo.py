# encoding=utf-8

from A_star_algo import *
import time
from datetime import datetime
import copy


def findShortestPath():
    map = Map(101, 101)
    map.start = (4, 4)  # setStartPoint((4, 4))
    obstacles = [(0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8),
                 (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1)]
    # map.setObstacles(False, 0.1, obstacles)
    map.setObstacles(True, 0.2)
    # print(map.map)
    algo = AStar(map, 1)
    sum = 0
    for i in range(100):
        map = Map(101, 101)
        map.setObstacles(True, 0.1)
        algo = AStar(map, 1)
        result = algo.run()
        print(result)
        if result:
            sum += 1
    print(sum / 100)
    path = algo.path
    last = map.end
    while last in path:
        # print(last, path[last])
        last = path[last]

    img = Image.fromarray(np.uint8(cm.gist_earth(map.map) * 255))

    # mymap = np.array(img)  # 图像转化为二维数组
    # 绘制路径
    img = np.array(img.convert('RGB'))
    print(img.shape)
    last = map.end
    while last in path:
        img[last[0]][last[1]] = [0, 255, 255]
        last = path[last]
    start = map.start
    end = map.end
    img[start[0]][start[1]] = [255, 0, 0]
    img[end[0]][end[1]] = [255, 0, 0]
    ax = plt.gca()
    # ax.set_xticks(range(101))
    # ax.set_yticks(range(101))
    plt.imshow(img)
    plt.grid(linewidth=1)
    # plt.axis('off')
    plt.show()


if __name__ == "__main__":
    findShortestPath()
