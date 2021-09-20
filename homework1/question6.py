# encoding=utf-8

from A_star_algo import *
import time
from datetime import datetime
import copy

if __name__ == "__main__":
    nominator = 0
    denominator = 0
    test_num = 50
    p_list = np.linspace(0, 0.23, 24)
    ATL_list = []   # Average Trajectory Length
    ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    ALSPFDG_LSPFG_list = []   # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    ANCPR_list = []  # Density vs Average Number of Cells Processed by Repeated A*
    for p in p_list:
        ATL, ALT_LSPFDG, ALSPFDG_LSPFG, ANCPR = 0, 0, 0, 0
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
            algo.run()
            # q1
            ATL += len(algo.trajectory)
            # q2
            As_gridWorld = AStar(algo.gridWorld, 1)
            As_gridWorld.run()
            ALT_LSPFDG += (len(algo.trajectory)*1.0 / len(As_gridWorld.trajectory))
            # q3
            ALT_LSPFDG += (len(As_gridWorld.trajectory)*1.0 / len(As.trajectory))
            # q4
            ANCPR += len(algo.cells)
        ATL_list.append(ATL / test_num)
        ALT_LSPFDG_list.append(ALT_LSPFDG / test_num)
        ALSPFDG_LSPFG_list.append(ALSPFDG_LSPFG / test_num)
        ANCPR_list.append(ANCPR / test_num)

    plt.plot(p_list, ATL_list)
    plt.xlabel("density")
    plt.ylabel("Average Trajectory Length")
    plt.show()
    plt.plot(p_list, ALT_LSPFDG_list)
    plt.xlabel('density')
    plt.ylabel('Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld')
    plt.show()
    plt.plot(p_list, ALSPFDG_LSPFG_list)
    plt.xlabel("density")
    plt.ylabel("Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld")
    plt.show()
    plt.plot(p_list, ANCPR_list)
    plt.xlabel('density')
    plt.ylabel('Average Number of Cells Processed by Repeated A*')
    plt.show()
