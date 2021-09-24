# encoding=utf-8

from A_star_algo import *
import time
import sys
import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':

    total_num = 2  # 50
    p_list = np.linspace(0.0, 0.35, 36)
    manhattan_time_list, euclidean_time_list, chebyshev_time_list = [], [], []
    for p in p_list:
        manhattan_time, euclidean_time, chebyshev_time = 0, 0, 0
        for _ in range(total_num):
            map = Map(101, 101)
            map.setObstacles(True, p)
            manhattan = AStar(map, 1)
            while True:
                res = manhattan.run()
                if not res:
                    map.reset()
                    map.setObstacles(True, p)
                    manhattan = AStar(map, 1)
                else:
                    break
            time1 = time.time()
            manhattan.clear()
            a = manhattan.run()
            time2 = time.time()
            manhattan_time += (time2 - time1)
            euclidean = AStar(map, 2)
            euclidean.run()
            time3 = time.time()
            euclidean_time += (time3 - time2)
            chebyshev = AStar(map, 3)
            chebyshev.run()
            time4 = time.time()
            chebyshev_time += (time4 - time3)
        print(manhattan_time)
        manhattan_time_list.append(manhattan_time / total_num)
        euclidean_time_list.append(euclidean_time / total_num)
        chebyshev_time_list.append(chebyshev_time / total_num)

    plt.plot(p_list, manhattan_time_list, color='red')
    plt.plot(p_list, euclidean_time_list, color='blue')
    plt.plot(p_list, chebyshev_time_list, color='green')
    plt.legend(['Manhattan', 'Euclidean', 'Chebyshev'])
    plt.xlabel("random probability p")
    plt.ylabel("runtime")
    plt.title("dim = 101x101 distanceType")
    plt.show()
