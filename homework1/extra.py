# encoding=utf-8

from A_star_algo import *
import time
from datetime import datetime
import copy


def q6():
    test_num = 10  # 50
    p_list = np.linspace(0, 0.33, 20)  # 0, 0.33, 34
    bfs_ATL_list = []  # Average Trajectory Length
    bfs_ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    bfs_ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    bfs_ANCPR_list = []  # Density vs Average Number of Cells Processed by Repeated A*
    ATL_list = []  # Average Trajectory Length
    ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    ANCPR_list = []
    for p in p_list:
        bfs_ATL, bfs_ALT_LSPFDG, bfs_ALSPFDG_LSPFG, bfs_ANCPR = 0, 0, 0, 0
        ATL, ALT_LSPFDG, ALSPFDG_LSPFG, ANCPR = 0, 0, 0, 0
        for _ in range(test_num):
            r1 = False
            map = Map(101, 101)
            map.setObstacles(True, p)
            bfs = BFS(map)
            while True:
                if not bfs.run():
                    map.reset()
                    map.setObstacles(True, p)
                    bfs = BFS(map)
                else:
                    break
            As = AStar(map, 1)
            As.run()
            repeatedBfs = RepeatedBFS(map)
            repeatedBfs.run()
            algo = RepeatedAStar(copy.deepcopy(map), 1)
            algo.run()
            # q1
            bfs_ATL += len(repeatedBfs.trajectory)
            ATL += len(algo.trajectory)
            # q2
            repeatedBfs.gridWorld.start = (0, 0)    #setStartPoint((0, 0))
            As_gridWorld = AStar(repeatedBfs.gridWorld, 1)
            As_gridWorld.run()
            bfs_ALT_LSPFDG += (len(repeatedBfs.trajectory) * 1.0 / len(As_gridWorld.trajectory))
            ALT_LSPFDG += (len(algo.trajectory) * 1.0 / len(As_gridWorld.trajectory))
            # q3
            bfs_ALSPFDG_LSPFG += (len(As_gridWorld.trajectory) * 1.0 / len(As.trajectory))
            ALSPFDG_LSPFG += (len(As_gridWorld.trajectory) * 1.0 / len(As.trajectory))
            # q4
            bfs_ANCPR += repeatedBfs.cells
            ANCPR += algo.cells
            # print(len(algo.trajectory))
        # bfs
        bfs_ATL_list.append(bfs_ATL / test_num)
        bfs_ALT_LSPFDG_list.append(bfs_ALT_LSPFDG / test_num)
        bfs_ALSPFDG_LSPFG_list.append(bfs_ALSPFDG_LSPFG / test_num)
        bfs_ANCPR_list.append(bfs_ANCPR / test_num)
        # repeated A*
        ATL_list.append(ATL / test_num)
        ALT_LSPFDG_list.append(ALT_LSPFDG / test_num)
        ALSPFDG_LSPFG_list.append(ALSPFDG_LSPFG / test_num)
        ANCPR_list.append(ANCPR / test_num)

    plt.plot(p_list, bfs_ATL_list)
    plt.plot(p_list, ATL_list, color="red")
    plt.legend(['Repeated BFS', 'Repeated A*'])
    plt.xlabel("density")
    plt.ylabel("Average Trajectory Length")
    plt.show()
    plt.plot(p_list, bfs_ALT_LSPFDG_list)
    plt.plot(p_list, ALT_LSPFDG_list, color="red")
    plt.legend(['Repeated BFS', 'Repeated A*'])
    plt.xlabel('density')
    plt.ylabel('Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld')
    plt.show()
    plt.plot(p_list, bfs_ALSPFDG_LSPFG_list)
    plt.plot(p_list, ALSPFDG_LSPFG_list, color="red")
    plt.legend(['Repeated BFS', 'Repeated A*'])
    plt.yticks(np.linspace(0.9, 1.05, 16))
    plt.xlabel("density")
    plt.ylabel("Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld")
    plt.show()
    plt.plot(p_list, bfs_ANCPR_list)
    plt.plot(p_list, ANCPR_list, color="red")
    plt.legend(['Repeated BFS', 'Repeated A*'])
    plt.xlabel('density')
    plt.ylabel('Average Number of Cells Processed by Repeated A*')
    plt.show()


def q7():
    test_num = 10  # 20
    p_list = np.linspace(0.1, 0.33, 20)  # 0, 0.33, 34
    ATL_list = []  # Average Trajectory Length
    ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    ANCPR_list = []  # Density vs Average Number of Cells Processed by Repeated A*
    bump_ATL_list = []  # Average Trajectory Length
    bump_ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    bump_ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    bump_ANCPR_list = []
    for p in p_list:
        ATL, ALT_LSPFDG, ALSPFDG_LSPFG, ANCPR = 0, 0, 0, 0
        bump_ATL, bump_ALT_LSPFDG, bump_ALSPFDG_LSPFG, bump_ANCPR = 0, 0, 0, 0
        for _ in range(test_num):
            map = Map(101, 101)
            map.setObstacles(True, p)
            bfs = BFS(map)
            while True:
                if not bfs.run():
                    map.reset()
                    map.setObstacles(True, p)
                    bfs = BFS(map)
                else:
                    break
            As = AStar(map, 1)
            As.run()
            repeated_bfs = RepeatedBFS(map)
            repeated_bfs.run()
            bump_bfs = RepeatedBFS(copy.deepcopy(map))
            bump_bfs.bump_into()
            # q1
            ATL += len(repeated_bfs.trajectory)
            bump_ATL += len(bump_bfs.trajectory)
            # q2
            repeated_bfs.gridWorld.start = (0, 0)   #setStartPoint((0, 0))
            As_gridWorld = AStar(repeated_bfs.gridWorld, 1)
            As_gridWorld.run()
            ALT_LSPFDG += (len(repeated_bfs.trajectory) * 1.0 / len(As_gridWorld.trajectory))
            bump_bfs.gridWorld.start = (0, 0)   # setStartPoint((0, 0))
            bump_As_gridWorld = AStar(bump_bfs.gridWorld, 1)
            bump_As_gridWorld.run()
            bump_ALT_LSPFDG += (len(bump_bfs.trajectory) * 1.0 / len(bump_As_gridWorld.trajectory))

            # q3
            ALSPFDG_LSPFG += (len(As_gridWorld.trajectory) * 1.0 / len(As.trajectory))
            bump_ALSPFDG_LSPFG += (len(bump_As_gridWorld.trajectory) * 1.0 / len(As.trajectory))
            # q4
            ANCPR += repeated_bfs.cells
            bump_ANCPR += bump_bfs.cells
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
    plt.title("Repeated BFS")
    plt.legend(['original', 'bump into the cell'])
    plt.xlabel("density")
    plt.ylabel("Average Trajectory Length")
    plt.show()
    plt.plot(p_list, ALT_LSPFDG_list)
    plt.plot(p_list, bump_ALT_LSPFDG_list, color="red")
    plt.title("Repeated BFS")
    plt.legend(['original', 'bump into the cell'])
    plt.xlabel('density')
    plt.ylabel('Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld')
    plt.show()
    plt.plot(p_list, ALSPFDG_LSPFG_list)
    plt.plot(p_list, bump_ALSPFDG_LSPFG_list, color="red")
    plt.title("Repeated BFS")
    plt.legend(['original', 'bump into the cell'])
    plt.yticks(np.linspace(0.9, 1.05, 16))
    plt.xlabel("density")
    plt.ylabel("Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld")
    plt.show()
    plt.plot(p_list, ANCPR_list)
    plt.plot(p_list, bump_ANCPR_list, color="red")
    plt.title("Repeated BFS")
    plt.legend(['original', 'bump into the cell'])
    plt.xlabel('density')
    plt.ylabel('Average Number of Cells Processed by Repeated BFS')
    plt.show()


if __name__ == "__main__":
    # q6()
    q7()
