# encoding=utf-8

from A_star_algo import *
import time
from datetime import datetime
import copy

if __name__ == "__main__":
    test_num = 50   # 50
    p_list = np.linspace(0, 0.33, 10)   # 0, 0.33, 34
    ATL_list = []   # Average Trajectory Length
    ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    ALSPFDG_LSPFG_list = []   # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    ANCPR_list = []  # Density vs Average Number of Cells Processed by Repeated A*
    bump_ATL_list = []  # Average Trajectory Length
    bump_ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    bump_ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    bump_ANCPR_list = []
    for p in p_list:
        ATL, ALT_LSPFDG, ALSPFDG_LSPFG, ANCPR = 0, 0, 0, 0
        bump_ATL, bump_ALT_LSPFDG, bump_ALSPFDG_LSPFG, bump_ANCPR = 0, 0, 0, 0
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
            bump_algo.run(bumpInto=True)
            # q1
            ATL += len(algo.trajectory)
            bump_ATL += len(bump_algo.trajectory)
            # q2
            algo.gridWorld.setStartPoint((0, 0))
            As_gridWorld = AStar(algo.gridWorld, 1)
            As_gridWorld.run()
            ALT_LSPFDG += (len(algo.trajectory)*1.0 / len(As_gridWorld.trajectory))
            bump_algo.gridWorld.setStartPoint((0, 0))
            bump_As_gridWorld = AStar(bump_algo.gridWorld, 1)
            bump_As_gridWorld.run()
            bump_ALT_LSPFDG += (len(bump_algo.trajectory) * 1.0 / len(bump_As_gridWorld.trajectory))

            # q3
            ALSPFDG_LSPFG += (len(As_gridWorld.trajectory)*1.0 / len(As.trajectory))
            bump_ALSPFDG_LSPFG += (len(bump_As_gridWorld.trajectory) * 1.0 / len(As.trajectory))
            # q4
            ANCPR += algo.cells
            bump_ANCPR += bump_algo.cells
            # print(len(algo.trajectory))
        ATL_list.append(ATL / test_num)
        ALT_LSPFDG_list.append(ALT_LSPFDG / test_num)
        ALSPFDG_LSPFG_list.append(ALSPFDG_LSPFG / test_num)
        ANCPR_list.append(ANCPR / test_num)
        # bump into
        bump_ATL_list.append(bump_ATL / test_num)
        bump_ALT_LSPFDG_list.append(bump_ALT_LSPFDG / test_num)
        bump_ALSPFDG_LSPFG_list.append(bump_ALSPFDG_LSPFG / test_num)
        bump_ANCPR_list.append(bump_ANCPR / test_num)

    plt.plot(p_list, ATL_list)
    plt.plot(p_list, bump_ATL_list, color="red")
    plt.legend(['original', 'bump into the cell'])
    plt.xlabel("density")
    plt.ylabel("Average Trajectory Length")
    plt.show()
    plt.plot(p_list, ALT_LSPFDG_list)
    plt.plot(p_list, bump_ALT_LSPFDG_list, color="red")
    plt.legend(['original', 'bump into the cell'])
    plt.xlabel('density')
    plt.ylabel('Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld')
    plt.show()
    plt.plot(p_list, ALSPFDG_LSPFG_list)
    plt.plot(p_list, bump_ALSPFDG_LSPFG_list, color="red")
    plt.legend(['original', 'bump into the cell'])
    plt.yticks(np.linspace(0.9, 1.05, 16))
    plt.xlabel("density")
    plt.ylabel("Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld")
    plt.show()
    plt.plot(p_list, ANCPR_list)
    plt.plot(p_list, bump_ANCPR_list, color="red")
    plt.legend(['original', 'bump into the cell'])
    plt.xlabel('density')
    plt.ylabel('Average Number of Cells Processed by Repeated A*')
    plt.show()
