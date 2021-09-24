# encoding=utf-8

from A_star_algo import *
import time


def question8():
    for _ in range(1):
        map = Map(101, 101)
        map.setObstacles(True, 0.2)
        algo = RepeatedAStar(map, 1)
        startTime = time.time()
        result = algo.run(False)
        endTime = time.time()
        img = Image.fromarray(np.uint8(cm.gist_earth(map.map) * 255))
        path = algo.path
        img = np.array(img.convert('RGB'))
        print(img.shape)
        last = map.end
        for trace in algo.trajectory:
            img[trace[0]][trace[1]] = [0, 0, 255]
        start = map.getStartPoint()
        end = map.getEndPoint()
        img[start[0]][start[1]] = [255, 0, 0]
        img[end[0]][end[1]] = [255, 0, 0]
        plt.imshow(img)
        plt.grid(linewidth=1)

        plt.show()


if __name__ == "__main__":
    question8()
