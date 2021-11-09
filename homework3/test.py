# encoding=utf-8

import copy
import numpy as np
from random import random
from utils import *
from Astar import AStar
import time
import pandas as pd
import ast
import gc
from probImpl import *


if __name__ == '__main__':
    # allMaze = loadMaze("../maps", "density0.3.json")
    n = 10
    allMaze = loadMaze("./full_connected_maps", "dim50_11.json")
    print(allMaze[0].shape)
    map, terrain = allMaze[0], allMaze[1]
    timeAgent6, timeAgent7 = 0, 0
    tjtAgent6, tjtAgent7 = 0, 0

    df = pd.read_csv(r"./data/map11.csv")
    df = df[df['agent'] == 6]
    df['1'] = df['1'].apply(ast.literal_eval)
    df['3'] = df['3'].apply(ast.literal_eval)
    a = df['1'].tolist()
    b = df['3'].tolist()
    # print(len(a))

    for i in range(n):
        # target = randomInitialize(map.shape[0], map.shape[1], map, True)
        # start = randomInitialize(map.shape[0], map.shape[1], map, True)
        # goal = randomInitialize(map.shape[0], map.shape[1], map, False)
        target = a[i]
        start = b[i]
        print(start, target)
        # goal =  (27, 14)

        # terrain = generateTerrain(map.shape[0], map.shape[1])
        # terrain[target[0]][target[1]] = 2

        # agent6 = ProbAgent(map, target)
        # agent6.start = start
        # agent6.terrain = terrain
        #
        # agent7 = ProbAgent(map, target)
        # agent7.start = start
        # agent7.terrain = terrain

        agent8 = ProbAgent(map, target)
        agent8.start = start
        agent8.terrain = terrain

        # agent6.agentType = 6
        # time1 = time.time()
        # agent6.agent6()
        # time2 = time.time()
        # print("agent6 true, time={time}, movement={movement}, examination={examination}, ratio={ratio}".format(time=time2 - time1, movement=len(agent6.trajectory), examination=agent6.examination, ratio=len(agent6.trajectory)/agent6.examination))
        # timeAgent6 += time2 - time1
        # tjtAgent6 += len(agent6.trajectory)

        # agent7.agentType = 7
        # time3 = time.time()
        # agent7.agent7()
        # time4 = time.time()
        # print("agent7 true, time={time}, movement={movement}, examination={examination}, ratio={ratio}".format(time=time4 - time3, movement=len(agent7.trajectory), examination=agent7.examination, ratio=len(agent7.trajectory)/agent7.examination))
        # timeAgent7 += time4 - time3
        # tjtAgent7 += len(agent7.trajectory)

        agent8.agentType = 7
        time5 = time.time()
        agent8.agent8()
        time6 = time.time()
        print("agent8 true, time={time}, movement={movement}, examination={examination}, ratio={ratio}".format(time=time6 - time5, movement=len(agent8.trajectory), examination=agent8.examination,
                                                                                                               ratio=len(agent8.trajectory) / agent8.examination))

        # break
    # print("agent6 time={timeAgent6}, trajectory length={len}".format(timeAgent6=timeAgent6 / n, len=tjtAgent6 / n))
    # print("agent7 time={timeAgent7}, trajectory length={len}".format(timeAgent7=timeAgent7 / n, len=tjtAgent7 / n))
