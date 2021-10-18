# encoding=utf-8

from homework1.A_star_algo import *
import time
from datetime import datetime
import copy
from inference_search import *
from agent5_implement import *

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
    prob_ATL_list = []
    prob_ALT_LSPFDG_list = []
    prob_ALSPFDG_LSPFG_list = []
    prob_ANCPR_list = []
    timeFirst, timeSecond, timeThird, timeForth, timeFifth = [], [], [], [], []
    example_Identified_Cells_list = []
    yourOwn_Identified_Cells_list = []

    for p in p_list:
        ATL, ALT_LSPFDG, ALSPFDG_LSPFG, ANCPR = 0.0, .0, .0, .0
        bump_ATL, bump_ALT_LSPFDG, bump_ALSPFDG_LSPFG, bump_ANCPR = .0, .0, .0, .0
        example_ATL, example_ALT_LSPFDG, example_ALSPFDG_LSPFG, example_ANCPR = .0, .0, .0, .0
        yourOwn_ATL, yourOwn_ALT_LSPFDG, yourOwn_ALSPFDG_LSPFG, yourOwn_ANCPR = .0, .0, .0, .0
        prob_ATL, prob_ALT_LSPFDG, prob_ALSPFDG_LSPFG, prob_ANCPR = .0, .0, .0, .0
        timeA, timeB, timeC, timeD, timeE = .0, .0, .0, .0, .0
        Identified_Cells1, Identified_Cells2, Identified_Cells3, Identified_Cells4, Identified_Cells5 = .0, .0, .0, .0, .0
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
            time1 = time.time()
            algo.run(bumpInto=False)
            time2 = time.time()
            # print(time2-time1)
            timeA += time2 - time1

            bump_algo = RepeatedAStar(copy.deepcopy(map), 1)
            time1_2 = time.time()
            bump_algo.run(bumpInto=True)
            time2_2 = time.time()
            timeB += time2_2 - time1_2

            example_al = InferenceSearch(copy.deepcopy(map))
            time1_3 = time.time()
            example_al.run()
            time2_3 = time.time()
            timeC += time2_3 - time1_3

            own_inference = InferenceSearch(copy.deepcopy(map))
            time1_4 = time.time()
            own_inference.run(trick=True)
            time2_4 = time.time()
            timeD += time2_4 - time1_4

            agent_5 = InferProbSearch(map, p)
            time1_5 = time.time()
            agent_5.run()
            time2_5 = time.time()
            timeE += time2_5 - time1_5

            finalMap1, finalMap2, finalMap3, finalMap4, finalMap5 = [copy.deepcopy(map)] * 5
            finalMap1.map = copy.deepcopy(algo.gridWorld.map)
            finalAs1 = AStar(finalMap1, 1)
            finalAs1.run()
            finalMap2.map = copy.deepcopy(bump_algo.gridWorld.map)
            finalAs2 = AStar(finalMap2, 1)
            finalAs2.run()
            finalMap3.map = copy.deepcopy(example_al.Maze)
            finalAs3 = AStar(finalMap3, 1)
            finalAs3.run()
            finalMap4.map = copy.deepcopy(own_inference.Maze)
            finalAs4 = AStar(finalMap4, 1)
            finalAs4.run()
            finalMap5.map = copy.deepcopy(agent_5.Maze)
            finalAs5 = AStar(finalMap5, 1)
            finalAs5.run()

            # Identified_Cells1 += sum(sum(finalMap1.Maze != 2))
            # Identified_Cells2 += sum(sum(finalMap2.Maze != 2))
            # Identified_Cells3 += sum(sum(finalMap3.Maze != 2))
            # Identified_Cells4 += sum(sum(finalMap4.Maze != 2))
            # Identified_Cells5 += sum(sum(finalMap5.Maze != 2))

            """average trajectory length"""
            ATL += len(algo.trajectory)
            bump_ATL += len(bump_algo.trajectory)
            example_ATL += len(example_al.trajectory)
            yourOwn_ATL += len(own_inference.trajectory)
            prob_ATL += len(agent_5.trajectory)

            """ALT_LSPFDG"""
            ALT_LSPFDG += (len(algo.trajectory) * 1.0 / len(finalAs1.trajectory))
            bump_ALT_LSPFDG += (len(bump_algo.trajectory) * 1.0 / len(finalAs2.trajectory))
            example_ALT_LSPFDG += (len(example_al.trajectory) * 1.0 / len(finalAs3.trajectory))
            yourOwn_ALT_LSPFDG += (len(own_inference.trajectory) * 1.0 / len(finalAs4.trajectory))
            prob_ALT_LSPFDG += (len(agent_5.trajectory) * 1.0 / len(finalAs5.trajectory))

            """ALSPFDG_LSPFG"""
            ALSPFDG_LSPFG += (len(finalAs1.trajectory) * 1.0 / len(As.trajectory))
            bump_ALSPFDG_LSPFG += (len(finalAs2.trajectory) * 1.0 / len(As.trajectory))
            example_ALSPFDG_LSPFG += (len(finalAs3.trajectory) * 1.0 / len(As.trajectory))
            yourOwn_ALSPFDG_LSPFG += (len(finalAs4.trajectory) * 1.0 / len(As.trajectory))
            prob_ALSPFDG_LSPFG += (len(finalAs5.trajectory) * 1.0 / len(As.trajectory))

            """ANCPR"""
            ANCPR += algo.cells
            bump_ANCPR += bump_algo.cells
            example_ANCPR += len(example_al.trajectory)
            yourOwn_ANCPR += len(own_inference.trajectory)
            prob_ANCPR += len(agent_5.trajectory)
            print("agent1-5")
        ATL_list.append(ATL / test_num)
        ALT_LSPFDG_list.append(ALT_LSPFDG / test_num)
        ALSPFDG_LSPFG_list.append(ALSPFDG_LSPFG / test_num)
        ANCPR_list.append(ANCPR / test_num)
        timeFirst.append(timeA / test_num)
        print(timeE/test_num)

        bump_ATL_list.append(bump_ATL / test_num)
        bump_ALT_LSPFDG_list.append(bump_ALT_LSPFDG / test_num)
        bump_ALSPFDG_LSPFG_list.append(bump_ALSPFDG_LSPFG / test_num)
        bump_ANCPR_list.append(bump_ANCPR / test_num)
        timeSecond.append(timeB / test_num)

        example_ATL_list.append(example_ATL / test_num)
        example_ALT_LSPFDG_list.append(example_ALT_LSPFDG / test_num)
        example_ALSPFDG_LSPFG_list.append(example_ALSPFDG_LSPFG / test_num)
        example_ANCPR_list.append(example_ANCPR / test_num)
        timeThird.append(timeC / test_num)

        yourOwn_ATL_list.append(yourOwn_ATL / test_num)
        yourOwn_ALT_LSPFDG_list.append(yourOwn_ALT_LSPFDG / test_num)
        yourOwn_ALSPFDG_LSPFG_list.append(yourOwn_ALSPFDG_LSPFG / test_num)
        yourOwn_ANCPR_list.append(yourOwn_ANCPR / test_num)
        timeForth.append(timeD / test_num)

        prob_ATL_list.append(prob_ATL / test_num)
        prob_ALT_LSPFDG_list.append(prob_ALT_LSPFDG / test_num)
        prob_ALSPFDG_LSPFG_list.append(prob_ALSPFDG_LSPFG / test_num)
        prob_ANCPR_list.append(prob_ANCPR / test_num)
        timeFifth.append(timeE / test_num)

    plt.plot(p_list, ATL_list, color="blue")
    plt.plot(p_list, bump_ATL_list, color="green")
    plt.plot(p_list, example_ATL_list, color="brown")
    plt.plot(p_list, yourOwn_ATL_list, color="red")
    plt.plot(p_list, prob_ATL_list, color="yellow")
    plt.legend(["agent1", "agent2", "agent3", "agent4", "agent5"])
    plt.xlabel("density")
    plt.ylabel("Average Trajectory Length")
    plt.title("agent1-5 average trajectory length comparison")
    plt.show()

    # Density vs Average (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)
    plt.plot(p_list, ALT_LSPFDG_list, color="blue")
    plt.plot(p_list, bump_ALT_LSPFDG_list, color="green")
    plt.plot(p_list, example_ALT_LSPFDG_list, color="brown")
    plt.plot(p_list, yourOwn_ALT_LSPFDG_list, color="red")
    plt.plot(p_list, prob_ALT_LSPFDG_list, color="yellow")
    plt.legend(["agent1", "agent2", "agent3", "agent4", "agent5"])
    plt.xlabel("density")
    plt.ylabel("Length of Trajectory/Length of Shortest Path in Final Discovered Gridworld ")
    plt.title("agent1-5 Length of Trajectory vs Length of Shortest Path in Final Discovered Gridworld")
    plt.show()

    plt.plot(p_list, ALSPFDG_LSPFG_list, color="blue")
    plt.plot(p_list, bump_ALSPFDG_LSPFG_list, color="green")
    plt.plot(p_list, example_ALSPFDG_LSPFG_list, color="brown")
    plt.plot(p_list, yourOwn_ALSPFDG_LSPFG_list, color="red")
    plt.plot(p_list, prob_ALSPFDG_LSPFG_list, color="yellow")
    plt.legend(["agent1", "agent2", "agent3", "agent4", "agent5"])
    plt.xlabel("density")
    plt.ylabel("Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld")
    plt.title("agent1-5 Length of Shortest Path in Final Discovered Gridworld vs Length of Shortest Path in Full Gridworld")
    plt.show()

    plt.plot(p_list, ANCPR_list, color="blue")
    plt.plot(p_list, bump_ANCPR_list, color="green")
    plt.plot(p_list, example_ANCPR_list, color="brown")
    plt.plot(p_list, yourOwn_ANCPR_list, color="red")
    plt.plot(p_list, prob_ANCPR_list, color="yellow")
    plt.legend(["agent1", "agent2", "agent3", "agent4", "agent5"])
    plt.xlabel("density")
    plt.ylabel("Average Number of Cells Processed")
    plt.title("agent1-5 Average Number of Cells Processed")
    plt.show()

    plt.plot(p_list, timeFirst, color="blue")
    plt.plot(p_list, timeSecond, color="green")
    plt.plot(p_list, timeThird, color="brown")
    plt.plot(p_list, timeForth, color="red")
    plt.plot(p_list, timeFifth, color="yellow")
    plt.legend(["agent1", "agent2", "agent3", "agent4", "agent5"])
    plt.xlabel("density")
    plt.ylabel("runtime")
    plt.title("agent1-5 runtime comparison")
    plt.show()
