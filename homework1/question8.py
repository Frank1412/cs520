# encoding=utf-8

from A_star_algo import *
import time


def question8():
    for _ in range(3):
        map = Map(101, 101)
        map.setObstacles(True, 0.3)
        for i in range(2):
            algo = RepeatedAStar(map, 1)
            startTime = time.time()
            print(i)
            result = algo.run(False, i)
            endTime = time.time()
            if result == False:break
            img = Image.fromarray(np.uint8(cm.gist_earth(map.map) * 255))
            img = np.array(img.convert('RGB'))
            print(img.shape)
            for trace in algo.trajectory:
                img[trace[0]][trace[1]] = [0, 0, 255]
            start = map.start
            end = map.getEndPoint()
            img[start[0]][start[1]] = [255, 0, 0]
            img[end[0]][end[1]] = [255, 0, 0]
            plt.title(str(i)+"Time consumption"+str(endTime - startTime))
            plt.imshow(img)
            plt.grid(linewidth=1)
            plt.show()


if __name__ == "__main__":
    question8()
