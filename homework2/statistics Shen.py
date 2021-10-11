# encoding=utf-8

from homework1.A_star_algo import *
import time
from datetime import datetime
import copy
from inference_search import *

if __name__ == "__main__":
    test_num = 50   # 50
    p_list = np.linspace(0, 0.33, 34)   # 0, 0.33, 34
    ATL_list = []   # Average Trajectory Length
    ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    ALSPFDG_LSPFG_list = []   # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    ANCPR_list = []  # Density vs Average Number of Cells Processed by Repeated A*
    bump_ATL_list = []  # Average Trajectory Length
    bump_ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    bump_ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    bump_ANCPR_list = []
    example_ATL_list = []  # Average Trajectory Length
    example_ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    example_ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    example_ANCPR_list = []
    timeFirst = []
    timeSecond = []
    timeThird = []


    for p in p_list:
        ATL, ALT_LSPFDG, ALSPFDG_LSPFG, ANCPR = 0, 0, 0, 0
        bump_ATL, bump_ALT_LSPFDG, bump_ALSPFDG_LSPFG, bump_ANCPR = 0, 0, 0, 0
        example_ATL, example_ALT_LSPFDG, example_ALSPFDG_LSPFG, example_ANCPR = 0, 0, 0, 0
        timeA, timeB, timeC = 0,0,0
        for _ in range(test_num):
            r1 = False
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
            time1 = time.time()
            algo = RepeatedAStar(map, 1)
            algo.run(bumpInto=False)
            time2 = time.time()
            timeA += time2 - time1
            time3 = time.time()
            bump_algo = RepeatedAStar(copy.deepcopy(map), 1)
            bump_algo.run(bumpInto=True)
            time4 = time.time()
            timeB += time4 - time3
            time5 = time.time()
            example_al = InferenceSearch(copy.deepcopy(map))
            example_al.run()
            time6 = time.time()
            timeC += time6 - time5
            # average length
            ATL += len(algo.trajectory)
            bump_ATL += len(bump_algo.trajectory)
            example_ATL += len(example_al.trajectory)

        ATL_list.append(ATL / test_num)
        # bump into
        bump_ATL_list.append(bump_ATL / test_num)

        # example
        example_ATL_list.append(example_ATL / test_num)
        timeFirst.append(timeA / test_num)
        timeSecond.append(timeB / test_num)
        timeThird.append(timeC / test_num)

    plt.plot(p_list, ATL_list)
    plt.plot(p_list, bump_ATL_list, color="red")
    plt.plot(p_list, example_ATL_list, color="green")
    plt.legend(['original', 'bump into the cell', "example inference"])
    plt.xlabel("density")
    plt.ylabel("Average Trajectory Length")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p_list, timeFirst, 'green', label="original")
    ax.plot(p_list, timeSecond, 'red', label="bump into the cell")
    ax.plot(p_list, timeThird, 'blue', label="example inference")
    plt.xlabel("density")
    plt.ylabel("Runtime")
    plt.legend()
    plt.show()