# encoding=utf-8
import sys
sys.path.append("..")
from homework1.A_star_algo import *
import time
import copy
from inference_search import *

if __name__ == "__main__":
    test_num = 3   # 50
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

            ATL += len(algo.trajectory)
            algo.gridWorld.start = (0, 0)   # setStartPoint((0, 0))
            As_gridWorld = AStar(algo.gridWorld, 1)
            As_gridWorld.run()
            ALT_LSPFDG += (len(algo.trajectory)*1.0 / len(As_gridWorld.trajectory))
            ALSPFDG_LSPFG += (len(As_gridWorld.trajectory)*1.0 / len(As.trajectory))
            ANCPR += algo.cells

            bump_ATL += len(bump_algo.trajectory)
            algo.gridWorld.start = (0, 0)   # setStartPoint((0, 0))
            As_gridWorld = AStar(algo.gridWorld, 1)
            As_gridWorld.run()
            bump_ALT_LSPFDG += (len(algo.trajectory)*1.0 / len(As_gridWorld.trajectory))
            bump_ALSPFDG_LSPFG += (len(As_gridWorld.trajectory)*1.0 / len(As.trajectory))
            bump_ANCPR += algo.cells           

            example_ATL += len(example_al.trajectory)
            algo.gridWorld.start = (0, 0)   # setStartPoint((0, 0))
            As_gridWorld = AStar(algo.gridWorld, 1)
            As_gridWorld.run()
            example_ALT_LSPFDG += (len(algo.trajectory)*1.0 / len(As_gridWorld.trajectory))
            example_ALSPFDG_LSPFG += (len(As_gridWorld.trajectory)*1.0 / len(As.trajectory))
            example_ANCPR += algo.cells


        ATL_list.append(ATL / test_num)
        ALT_LSPFDG_list.append(ALT_LSPFDG / test_num)
        ALSPFDG_LSPFG_list.append(ALSPFDG_LSPFG / test_num)
        ANCPR_list.append(ANCPR / test_num)       
        # bump into
        bump_ATL_list.append(bump_ATL / test_num)
        bump_ALT_LSPFDG_list.append(ALT_LSPFDG / test_num)
        bump_ALSPFDG_LSPFG_list.append(ALSPFDG_LSPFG / test_num)
        bump_ANCPR_list.append(ANCPR / test_num)

        # example
        example_ATL_list.append(example_ATL / test_num)
        example_ALT_LSPFDG_list.append(ALT_LSPFDG / test_num)
        example_ALSPFDG_LSPFG_list.append(ALSPFDG_LSPFG / test_num)
        example_ANCPR_list.append(ANCPR / test_num) 

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

    plt.plot(p_list, ALT_LSPFDG_list)
    plt.plot(p_list, bump_ALT_LSPFDG_list, color="red")
    plt.plot(p_list, example_ALT_LSPFDG_list, color="green")
    plt.legend(['original', 'bump into the cell', "example inference"])
    plt.xlabel('density')
    plt.ylabel('Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld')
    plt.show()

    plt.plot(p_list, ALSPFDG_LSPFG_list)
    plt.plot(p_list, bump_ALSPFDG_LSPFG_list, color="red")
    plt.plot(p_list, example_ALSPFDG_LSPFG_list, color="green")
    plt.yticks(np.linspace(0.9, 1.05, 16))
    plt.legend(['original', 'bump into the cell', "example inference"])
    plt.xlabel("density")
    plt.ylabel("Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld")
    plt.show()

    plt.plot(p_list, ANCPR_list)
    plt.plot(p_list, bump_ANCPR_list, color="red")
    plt.plot(p_list, example_ANCPR_list, color="green")
    plt.legend(['original', 'bump into the cell', "example inference"])
    plt.xlabel('density')
    plt.ylabel('Average Number of Cells Processed by Repeated A*')
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