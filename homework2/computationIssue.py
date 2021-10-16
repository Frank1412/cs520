# encoding=utf-8

from homework1.A_star_algo import *
import time
from datetime import datetime
import copy
from inference_search import *

if __name__ == "__main__":
    test_num = 20  # 50
    p_list = np.linspace(0, 0.33, 11)  # 0, 0.33, 34
    ATL_list = []  # Average Trajectory Length
    ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    ANCPR_list = []  # Density vs Average Number of Cells Processed by Repeated A*
    bump_ATL_list = []  # Average Trajectory Length
    bump_ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    bump_ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    bump_ANCPR_list = []
    example_ATL_list = []  # Average Trajectory Length
    example_ALT_LSPFDG_list = []  # Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld
    example_ALSPFDG_LSPFG_list = []  # Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld
    example_ANCPR_list = []
    yourOwn_ATL_list = []
    yourOwn_ALT_LSPFDG_list = []
    yourOwn_ALSPFDG_LSPFG_list = []
    yourOwn_ANCPR_list = []
    timeFirst = []
    timeSecond = []
    timeThird = []
    timeFourth = []
    example_Identified_Cells_list = []
    yourOwn_Identified_Cells_list = []
    own_ATL1_list = []
    own_ATL2_list = []
    own_ATL3_list = []
    own_ATL4_list = []

    for p in p_list:
        ATL, ALT_LSPFDG, ALSPFDG_LSPFG, ANCPR = 0, 0, 0, 0
        bump_ATL, bump_ALT_LSPFDG, bump_ALSPFDG_LSPFG, bump_ANCPR = 0, 0, 0, 0
        example_ATL, example_ALT_LSPFDG, example_ALSPFDG_LSPFG, example_ANCPR = 0, 0, 0, 0
        yourOwn_ATL, yourOwn_ALT_LSPFDG, yourOwn_ALSPFDG_LSPFG, yourOwn_ANCPR = 0, 0, 0, 0
        timeA, timeB, timeC, timeD = 0, 0, 0, 0
        example_Identified_Cells, yourOwn_Identified_Cells = 0, 0
        own_ATL1, own_ATL2, own_ATL3, own_ATL4 = 0.0, 0.0, 0.0, 0.0
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

            own_inference1 = InferenceSearch(copy.deepcopy(map))
            own_inference1.scanArea = 1
            time1 = time.time()
            own_inference1.run(trick=True)
            time2 = time.time()
            timeA += time2 - time1

            own_inference2 = InferenceSearch(copy.deepcopy(map))
            own_inference2.scanArea = 2
            time3 = time.time()
            own_inference2.run(trick=True)
            time4 = time.time()
            timeB += time4 - time3

            own_inference3 = InferenceSearch(copy.deepcopy(map))
            # own_inference3.scanArea = 3
            time5 = time.time()
            own_inference3.run(trick=True)
            time6 = time.time()
            timeC += time6 - time5

            # own_inference4 = InferenceSearch(copy.deepcopy(map))
            # time7 = time.time()
            # own_inference4.run(trick=True)
            # time8 = time.time()
            # timeD += time8-time7

            own_ATL1 += len(own_inference1.trajectory)
            own_ATL2 += len(own_inference2.trajectory)
            own_ATL3 += len(own_inference3.trajectory)
            # own_ATL4 += len(own_inference4.trajectory)

            print(len(own_inference1.trajectory), len(own_inference2.trajectory), len(own_inference3.trajectory))

        # example
        # example_ATL_list.append(example_ATL / test_num)
        # yourOwn_ATL_list.append(yourOwn_ATL / test_num)
        timeFirst.append(timeA / test_num)
        timeSecond.append(timeB / test_num)
        timeThird.append(timeC / test_num)
        # timeFourth.append(timeD/test_num)
        own_ATL1_list.append(own_ATL1 / test_num)
        own_ATL2_list.append(own_ATL2 / test_num)
        own_ATL3_list.append(own_ATL3 / test_num)
        # own_ATL4_list.append(own_ATL4 / test_num)
        # example_ALT_LSPFDG_list.append(example_ALT_LSPFDG / test_num)
        # example_ALSPFDG_LSPFG_list.append(example_ALSPFDG_LSPFG / test_num)
        # example_ANCPR_list.append(example_ATL / test_num)
        # yourOwn_ALT_LSPFDG_list.append(yourOwn_ALT_LSPFDG / test_num)
        # yourOwn_ALSPFDG_LSPFG_list.append(yourOwn_ALSPFDG_LSPFG / test_num)
        # yourOwn_ANCPR_list.append(yourOwn_ATL / test_num)
        #
        # example_Identified_Cells_list.append(example_Identified_Cells / test_num)
        # yourOwn_Identified_Cells_list.append(yourOwn_Identified_Cells / test_num)

    # Density vs Average Trajectory Length
    # plt.plot(p_list, example_ATL_list, color="green")
    # plt.plot(p_list, yourOwn_ATL_list, color="red")
    # plt.legend(["example inference", "Own_Inference"])
    # plt.xlabel("density")
    # plt.ylabel("Average Trajectory Length")
    # plt.show()

    # Density vs Average (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(p_list, example_ALT_LSPFDG_list, 'green', label="example inference")
    # ax.plot(p_list, yourOwn_ALT_LSPFDG_list, 'red', label="Own_Inference")
    # plt.xlabel("density")
    # plt.ylabel("Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld")
    # plt.legend()
    # plt.show()

    # Density vs Average (Length of Shortest Path in Final Discovered Gridworld / Length of Shortest
    # Path in Full Gridworld)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(p_list, example_ALSPFDG_LSPFG_list, 'green', label="example inference")
    # ax.plot(p_list, yourOwn_ALSPFDG_LSPFG_list, 'red', label="Own_Inference")
    # plt.xlabel("density")
    # plt.ylabel("Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld")
    # plt.legend()
    # plt.show()

    # Density vs Average Number of Cells Processed by Repeated A*

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(p_list, example_ANCPR_list, 'green', label="example inference")
    # ax.plot(p_list, yourOwn_ANCPR_list, 'red', label="Own_Inference")
    # plt.xlabel("density")
    # plt.ylabel("Density vs Average Number of Cells Processed by Repeated A*")
    # plt.legend()
    # plt.show()

    # Runtime

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(p_list, timeFirst, 'green', label="example inference")
    # ax.plot(p_list, timeSecond, 'red', label="Own_Inference")
    # plt.xlabel("density")
    # plt.ylabel("Runtime")
    # plt.legend()
    # plt.show()

    # number of identified cells
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(p_list, example_Identified_Cells_list, 'green', label="example inference")
    # ax.plot(p_list, yourOwn_Identified_Cells_list, 'red', label="Own_Inference")
    # plt.xlabel("density")
    # plt.ylabel("number of identified cells")
    # plt.legend()
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p_list, timeFirst, 'green', label="own_Inference1")
    ax.plot(p_list, timeSecond, 'red', label="own_Inference2")
    ax.plot(p_list, timeThird, 'blue', label="own_Inference3")
    # ax.plot(p_list, timeFourth, 'black', label="own_Inference4")
    plt.xlabel("density")
    plt.ylabel("Runtime(s)")
    plt.legend()
    plt.show()

    # Density vs Average Trajectory Length
    plt.plot(p_list, own_ATL1_list, color="green")
    plt.plot(p_list, own_ATL2_list, color="red")
    plt.plot(p_list, own_ATL3_list, color="blue")
    # plt.plot(p_list, own_ATL4_list, color="black")
    plt.legend(["own_Inference1", "own_Inference2", "own_Inference3", "own_Inference4"])
    plt.xlabel("density")
    plt.ylabel("Average Trajectory Length")
    plt.show()
