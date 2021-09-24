# encoding=utf-8

from A_star_algo import *
import time
import sys
import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':

    test_num = 1  # 100
    p_list = np.linspace(0, 1, 101)
    solvability = []
    for p in p_list:
        success = 0.0
        for _ in range(test_num):
            map = Map(101, 101)
            map.setObstacles(True, p)
            As = AStar(map, 1)
            if As.run():
                success += 1
        print(success)
        solvability.append(success / test_num)

    plt.plot(p_list, solvability)
    plt.xlabel("random probability p")
    plt.ylabel("solvability")
    plt.title("dim = 101x101")
    plt.show()
