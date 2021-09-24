# encoding=utf-8

from A_star_algo import *
import time
import sys
import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':

    test_num = 1  # 100
    p_list = np.linspace(0, 0.33, 20)
    p = 0.2
    weighted_list = np.linspace(0.5, 1.5, 10)
    solvability = []
    original_tjt_list, weighted_tjt_list, combination_tjt_list = [], [], []
    original_time_list, weighted_time_list, combination_time_list = [], [], []
    for coef in weighted_list:

        original_tjt, weighted_tjt, combination_tjt = 0, 0, 0
        original_time, weighted_time, combination_time = 0, 0, 0
        for _ in range(test_num):
            map = Map(101, 101)
            map.setObstacles(True, p)
            As = AStar(map, 1)
            while True:
                if not As.run():
                    map.reset()
                    map.setObstacles(True, p)
                    As = AStar(map, 1)
                else:
                    break
            origin_As = AStar(map, 1)
            weighted_As = AStar(map, 1)
            combination_As = AStar(map, 1)
            # time1 = time.time()
            # origin_As.run()
            # time2 = time.time()
            # original_time += time2-time1
            # original_tjt += len(origin_As.trajectory)

            time3 = time.time()
            weighted_As.run("weighted", coef)
            time4 = time.time()
            weighted_time += time4 - time3
            weighted_tjt += len(weighted_As.trajectory)

            # time5 = time.time()
            # combination_As.run("combination")
            # time6 = time.time()
            # combination_time += time6-time5
            # combination_tjt += len(combination_As.trajectory)

        # original_time_list.append(original_time / test_num)
        # original_tjt_list.append(original_tjt / test_num)
        weighted_time_list.append(weighted_time / test_num)
        weighted_tjt_list.append(weighted_tjt / test_num)
        # combination_tjt_list.append(combination_tjt / test_num)
        # combination_time_list.append(combination_time / test_num)

    plt.plot(weighted_list, weighted_tjt_list)
    plt.xlabel("weighted coefficient")
    plt.ylabel("Length of weighted heuristic")
    plt.show()
    plt.plot(weighted_list, weighted_time_list)
    plt.xlabel("weighted coefficient")
    plt.ylabel("runtime of weighted heuristic")
    plt.show()

    # plt.plot(p_list, original_time_list, color='blue')
    # plt.plot(p_list, weighted_time_list, color='red')
    # plt.plot(p_list, combination_time_list, color='green')
    # plt.xlabel("density")
    # plt.ylabel("runtime of admissible and inadmissible heuristic function")
    # plt.title("dim = 101x101 comparison between admissible and inadmissible function")
    # plt.show()
    # plt.plot(p_list, original_tjt_list, color='blue')
    # plt.plot(p_list, weighted_tjt_list, color='red')
    # plt.plot(p_list, combination_tjt_list, color='green')
    # plt.xlabel("density")
    # plt.ylabel("Length of trajectory of admissible and inadmissible heuristic function")
    # plt.title("dim = 101x101 comparison between admissible and inadmissible function")
    # plt.show()
