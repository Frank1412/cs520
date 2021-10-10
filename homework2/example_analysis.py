# encoding=utf-8

from A_star_algo import *
import time
from datetime import datetime
import copy
from inference_search import *

if __name__ == "__main__":
    test_num = 2   # 50
    p_list = np.linspace(0, 0.33, 10)   # 0, 0.33, 34
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

    for p in p_list:
        ATL, ALT_LSPFDG, ALSPFDG_LSPFG, ANCPR = 0, 0, 0, 0
        bump_ATL, bump_ALT_LSPFDG, bump_ALSPFDG_LSPFG, bump_ANCPR = 0, 0, 0, 0
        example_ATL, example_ALT_LSPFDG, example_ALSPFDG_LSPFG, example_ANCPR = 0, 0, 0, 0
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
            algo = RepeatedAStar(map, 1)
            algo.run(bumpInto=False)
            bump_algo = RepeatedAStar(copy.deepcopy(map), 1)
            time1 = time.time()
            bump_algo.run(bumpInto=True)
            time2 = time.time()
            example_al = InferenceSearch(copy.deepcopy(map))
            time3 = time.time()
            example_al.run()
            time4 = time.time()
            print(time2-time1, time4-time3)
            # average length
            ATL += len(algo.trajectory)
            bump_ATL += len(bump_algo.trajectory)
            example_ATL += len(example_al.trajectory)

        ATL_list.append(ATL / test_num)
        # bump into
        bump_ATL_list.append(bump_ATL / test_num)

        # example
        example_ATL_list.append(example_ATL / test_num)

    plt.plot(p_list, ATL_list)
    plt.plot(p_list, bump_ATL_list, color="red")
    plt.plot(p_list, example_ATL_list, color="green")
    plt.legend(['original', 'bump into the cell', "example inference"])
    plt.xlabel("density")
    plt.ylabel("Average Trajectory Length")
    plt.show()
