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
import csv

def write_csv(path, row):
    """
    Write a line of data to a csv file
    :param path: Path
    :param row: A line of data,
    ['..\\corpus\\W99-0632word.docx.txt', '110', '18.93%', '7', '1.20%', '2']
    :return:
    """
    with open(path, 'a+', newline='') as wf:
        csv_write = csv.writer(wf)
        # list
        data_row = row
        csv_write.writerow(data_row)


def create_csv(path, titles):
    """
    Create a csv file in the path
    :param path: Path
    :param titles: The first line of the csv file is the title of each column,
    ['CP', 'PERSON_count', 'PERSON_percentage', 'PRODUCT_count', 'PRODUCT_percentage', 'EVENT_count', 'EVENT_percentage']
    :return:
    """
    # path = "../result/result.csv"
    with open(path, 'w+', newline='') as wf:
        csv_write = csv.writer(wf)
        csv_head = titles
        csv_write.writerow(csv_head)





if __name__ == '__main__':
    list = [6,7,12,14,15]

    for i in list:
        print("map number {}".format(i))
        # allMaze = loadMaze("../maps", "density0.3.json")
        n = 30
        allMaze = loadMaze("./full_connected_maps", "dim50_{}.json".format(i))
        print(allMaze[0].shape)
        map, terrain = allMaze[0], allMaze[1]
        timeAgent6, timeAgent7 = 0, 0
        tjtAgent6, tjtAgent7 = 0, 0

        df = pd.read_csv(r"./agent_8_results/agent_6，7，8_map_{}.csv".format(i))
        df = df[df['agent'] == 6]
        df['start'] = df['start'].apply(ast.literal_eval)
        df['target'] = df['target'].apply(ast.literal_eval)
        a = df['start'].tolist()
        b = df['target'].tolist()


        path = ("./agent_8_results/agent_8_map_{}.csv".format(i))
        titles = ['terrain','start', 'target', 'agent', 'time', 'movement', 'examinations', 'ratio', 'sum']
        create_csv(path, titles)

        for i in range(n):
            #row1 = []
            # row2 = []
            row3 = []

            start = a[i]
            target = b[i]
            #print(start, target)

            # target = randomInitialize(map.shape[0], map.shape[1], map, True)
            # start = randomInitialize(map.shape[0], map.shape[1], map, True)
            # goal = randomInitialize(map.shape[0], map.shape[1], map, False)
            # target = (19, 35)
            # start = (33, 44)
            # goal =  (27, 14)

            # terrain = generateTerrain(map.shape[0], map.shape[1])
            # terrain[target[0]][target[1]] = 0

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
            # print("agent6 true, time={time}, movement={movement}, examination={examination}, ratio={ratio}".format(
            #     time=time2 - time1, movement=len(agent6.trajectory), examination=agent6.examination,
            #     ratio=len(agent6.trajectory) / agent6.examination))
            # timeAgent6 += time2 - time1
            # tjtAgent6 += len(agent6.trajectory)
            #
            # row1.append(agent6.terrain[target[0]][target[1]])
            # row1.append(start)
            # row1.append(target)
            # row1.append(6)
            # time11 = time2 - time1
            # row1.append(time11)
            # movement = len(agent6.trajectory)
            # row1.append(movement)
            # examination = agent6.examination
            # row1.append(examination)
            # ratio = len(agent6.trajectory) / agent6.examination
            # row1.append(ratio)
            # row1.append((movement + examination))
            #
            # agent7.agentType = 7
            # time3 = time.time()
            # agent7.agent7()
            # time4 = time.time()
            # print("agent7 true, time={time}, movement={movement}, examination={examination}, ratio={ratio}".format(
            #     time=time4 - time3, movement=len(agent7.trajectory), examination=agent7.examination,
            #     ratio=len(agent7.trajectory) / agent7.examination))
            # timeAgent7 += time4 - time3
            # tjtAgent7 += len(agent7.trajectory)
            #
            # row2.append(agent7.terrain[target[0]][target[1]])
            # row2.append(start)
            # row2.append(target)
            # row2.append(7)
            # time22 = time4 - time3
            # row2.append(time22)
            # movement = len(agent7.trajectory)
            # row2.append(movement)
            # examination = agent7.examination
            # row2.append(examination)
            # ratio = len(agent7.trajectory) / agent7.examination
            # row2.append(ratio)
            # row2.append((movement + examination))

            agent8.agentType = 7
            time5 = time.time()
            agent8.agent8()
            time6 = time.time()
            print("agent8 true, time={time}, movement={movement}, examination={examination}, ratio={ratio}".format(
                time=time6 - time5, movement=len(agent8.trajectory), examination=agent8.examination,
                ratio=len(agent8.trajectory) / agent8.examination))

            row3.append(agent8.terrain[target[0]][target[1]])
            row3.append(start)
            row3.append(target)
            row3.append(8)
            time33 = time6 - time5
            row3.append(time33)
            movement = len(agent8.trajectory)
            row3.append(movement)
            examination = agent8.examination
            row3.append(examination)
            ratio = len(agent8.trajectory) / agent8.examination
            row3.append(ratio)
            row3.append((movement + examination))

            # write_csv(path, row1)
            # write_csv(path, row2)
            write_csv(path, row3)

            # break
        # print("agent6 time={timeAgent6}, trajectory length={len}".format(timeAgent6=timeAgent6 / n, len=tjtAgent6 / n))
        # print("agent7 time={timeAgent7}, trajectory length={len}".format(timeAgent7=timeAgent7 / n, len=tjtAgent7 / n))