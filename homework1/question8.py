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
            result = algo.run(False, i)
            endTime = time.time()
            if result == False:break
            img = Image.fromarray(np.uint8(cm.gist_earth(map.map) * 255))
            img = np.array(img.convert('RGB'))
            traj = algo.trajectory
            for trace in traj:
                img[trace[0]][trace[1]] = [0, 0, 255]
            start = map.start
            end = map.end
            print(img.shape)
            img[start[0]][start[1]] = [255, 0, 0]
            img[end[0]][end[1]] = [255, 0, 0]
            improve = {
                0: "Standard Repeated A *:",
                1: "Improved Repeated A *:"
            }
            plt.title(improve.get(i)+"\nTime consumption"+str(endTime - startTime)+"\nTrajectory Length: "+str(len(traj)))
            plt.imshow(img)
            plt.grid(linewidth=1)
            plt.show()


if __name__ == "__main__":
    question8()
