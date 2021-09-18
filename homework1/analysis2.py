# encoding=utf-8

from A_star_algo import *
import time


def question5():
    for _ in range(2):
        map =  Map(101, 101)
        map.setObstacles(True, 0.2)
        for i in range(3):
            algo = AStar(map,i+1)
            startTime = time.time()
            result = algo.run()
            endTime = time.time()
            if(result==False):continue
            img = Image.fromarray(np.uint8(cm.gist_earth(map.map) * 255))
            path = algo.path
            img = np.array(img.convert('RGB'))
            print(img.shape)
            last = map.end
            for trace in algo.trajectory:
                img[trace[0]][trace[1]] = [0, 0, 255]
            while last in path:
                img[last[0]][last[1]] = [0, 255, 255]
                last = path[last]
            start = map.getStartPoint()
            end = map.getEndPoint()
            img[start[0]][start[1]] = [255, 0, 0]
            img[end[0]][end[1]] = [255, 0, 0]
            plt.imshow(img)
            plt.grid(linewidth=1)
            method = {
                0: "Manhattan",
                1: "Euclidean",
                2: "Chebyshev"
            }
            plt.title(method.get(i)+"\n"+"time consumption: "+str(endTime - startTime))
            plt.show() 
    
if __name__ == "__main__":
    question5()