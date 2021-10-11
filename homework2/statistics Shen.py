# encoding=utf-8

from homework1.A_star_algo import *
import time
from datetime import datetime
import copy
from inference_search import *

if __name__ == "__main__":
    test_num = 3   # 50
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
    yourOwn_ATL_list = []
    yourOwn_ALT_LSPFDG_list = []
    yourOwn_ALSPFDG_LSPFG_list = []
    yourOwn_ANCPR_list = []
    timeFirst = []
    timeSecond = []
    example_Identified_Cells_list = []
    yourOwn_Identified_Cells_list = []




    for p in p_list:
        ATL, ALT_LSPFDG, ALSPFDG_LSPFG, ANCPR = 0, 0, 0, 0
        bump_ATL, bump_ALT_LSPFDG, bump_ALSPFDG_LSPFG, bump_ANCPR = 0, 0, 0, 0
        example_ATL, example_ALT_LSPFDG, example_ALSPFDG_LSPFG, example_ANCPR = 0, 0, 0, 0
        yourOwn_ATL,yourOwn_ALT_LSPFDG,yourOwn_ALSPFDG_LSPFG,yourOwn_ANCPR = 0,0,0,0
        timeA, timeB, timeC = 0,0,0
        example_Identified_Cells,yourOwn_Identified_Cells = 0,0
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
            example_al = InferenceSearch(copy.deepcopy(map))
            example_al.run()
            time2 = time.time()
            # Length of Shortest Path in Final Discovered Gridworld
            # example_final = AStar(example_al.Maze, 1)
            """"""
            finalMap = copy.deepcopy(map)
            finalMap.map = copy.deepcopy(example_al.Maze)
            example_final = AStar(finalMap, 1)
            """"""
            example_final.run()

            example_Identified_Cells += sum(sum(example_al.Maze != 2))

            example_ALT_LSPFDG += (len(example_al.trajectory) * 1.0 / len(example_final.trajectory))
            print(len(example_final.trajectory), p)
            print(example_final.trajectory[:10])

            example_ALSPFDG_LSPFG += (len(example_final.trajectory) * 1.0 / len(As.trajectory))
            timeA += time2 - time1


            

            own_inference = InferenceSearch(copy.deepcopy(map))
            time3 = time.time()
            own_inference.run(trick=True)
            time4 = time.time()
            timeB += time4-time3
            # print(timeB)
            # Length of Shortest Path in Final Discovered Gridworld

            # yourOwn_final =  AStar(own_inference.Maze,1)
            finalMap1 = copy.deepcopy(map)
            finalMap1.map = copy.deepcopy(own_inference.Maze)
            yourOwn_final = AStar(finalMap1,1)

            yourOwn_final.run()

            yourOwn_Identified_Cells += sum(sum(own_inference.Maze!=2))
            # print(yourOwn_Identified_Cells)
            yourOwn_ALT_LSPFDG += (len(own_inference.trajectory)*1.0/len(yourOwn_final.trajectory))
            yourOwn_ALSPFDG_LSPFG += (len(yourOwn_final.trajectory)*1.0/len(As.trajectory))





            # average length

            example_ATL += len(example_al.trajectory)
            yourOwn_ATL += len(own_inference.trajectory)







        # example
        example_ATL_list.append(example_ATL / test_num)
        yourOwn_ATL_list.append(yourOwn_ATL/test_num)
        timeFirst.append(timeA / test_num)
        timeSecond.append(timeB / test_num)
        example_ALT_LSPFDG_list.append(example_ALT_LSPFDG/test_num)
        example_ALSPFDG_LSPFG_list.append(example_ALSPFDG_LSPFG/test_num)
        example_ANCPR_list.append(example_ATL / test_num)
        yourOwn_ALT_LSPFDG_list.append(yourOwn_ALT_LSPFDG/test_num)
        yourOwn_ALSPFDG_LSPFG_list.append(yourOwn_ALSPFDG_LSPFG/test_num)
        yourOwn_ANCPR_list.append(yourOwn_ATL/test_num)




   #Density vs Average Trajectory Length
    plt.plot(p_list, example_ATL_list, color="green")
    plt.plot(p_list, yourOwn_ATL_list, color="red")
    plt.legend(["example inference","Own_Inference"])
    plt.xlabel("density")
    plt.ylabel("Average Trajectory Length")
    plt.show()

#Density vs Average (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p_list, example_ALT_LSPFDG_list, 'green', label="example inference")
    ax.plot(p_list, yourOwn_ALT_LSPFDG_list, 'red', label="Own_Inference")
    plt.xlabel("density")
    plt.ylabel("Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld")
    plt.legend()
    plt.show()

    #Density vs Average (Length of Shortest Path in Final Discovered Gridworld / Length of Shortest
    #Path in Full Gridworld)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p_list, example_ALSPFDG_LSPFG_list, 'green', label="example inference")
    ax.plot(p_list, yourOwn_ALSPFDG_LSPFG_list, 'red', label="Own_Inference")
    plt.xlabel("density")
    plt.ylabel("Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld")
    plt.legend()
    plt.show()

    #Density vs Average Number of Cells Processed by Repeated A*

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p_list,example_ANCPR_list, 'green', label="example inference")
    ax.plot(p_list, yourOwn_ANCPR_list, 'red', label="Own_Inference")
    plt.xlabel("density")
    plt.ylabel("Density vs Average Number of Cells Processed by Repeated A*")
    plt.legend()
    plt.show()

    #Runtime

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p_list, timeFirst, 'green', label="example inference")
    ax.plot(p_list, timeSecond, 'red', label="Own_Inference")
    plt.xlabel("density")
    plt.ylabel("Runtime")
    plt.legend()
    plt.show()