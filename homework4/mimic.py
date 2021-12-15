# coding = utf-8

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.functional import cross_entropy
from sklearn import metrics
import copy
from transformers import AdamW
from agent1 import *
from agent2 import *
import agent1
from inference_search import *


def mimic2(modelType="CNN"):
    mazes = np.load("maps/test_30x30dim.npy")
    print(mazes.shape)
    soft_max = nn.Softmax(1)
    lenList = []
    for idx, map in enumerate(mazes[0:]):
        # gridWorld = np.full(map.shape, 2)
        # gridWorld[0][0], gridWorld[0][1], gridWorld[1][0] = 0, map[0][1], map[1][0]
        gridWorld, N, C, B, E, H = initialize(map)
        visit = np.zeros(map.shape)
        cur = (0, 0)
        if modelType == "CNN":
            state_dict = torch.load("./pics/proj2/overbestCNN.pt")
            # state_dict = torch.load("./pics/proj2/bestCNN.pt")
            model = Agent2CNN()
        else:
            state_dict = torch.load("./model/proj2/2layerNN_800.pt")
            model = Agent2NN()
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()

        trajectory = []
        visited = np.full(map.shape, False)
        # visited[0][0] = True
        res = dfsCNN(model, gridWorld, map, (0, 0), trajectory, soft_max, visited, N, C, B, E, H)
        # print(trajectory)
        if res:
            print("{i}th trial len={len}".format(i=idx + 1, len=len(trajectory)))
            lenList.append(len(trajectory))
        else:
            print(False)
            lenList.append(-1)
    np.save("./pics/proj2/over{modelType}len".format(modelType=modelType), lenList)


def mimic1(modelType="CNN"):
    mazes = np.load("maps/test_30x30dim.npy")
    print(mazes.shape)
    soft_max = nn.Softmax(1)
    lenList = []
    for idx, map in enumerate(mazes[:]):
        gridWorld = np.full(map.shape, 2)
        # gridWorld[0][0], gridWorld[0][1], gridWorld[1][0] = 0, map[0][1], map[1][0]
        visitCount = np.zeros(map.shape)
        visitCount[0][0] = 1
        visit = np.zeros(map.shape)
        cur = (0, 0)
        inputX = agent1.inputTransform(status=gridWorld, cur=(0, 0), visit=visit)
        if modelType == "CNN":
            # state_dict = torch.load("./model/proj1/CNN_50.pt")
            state_dict = torch.load("./pics/proj1/overbestCNN.pt")
            model = Agent1CNN()
        else:
            state_dict = torch.load("./model/proj1/NN_200.pt")
            model = Agent1NN()
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()
        dataLoader = DataLoader(MyDataset(inputX, np.zeros([len(gridWorld)])), batch_size=1)
        trajectory = []
        visited = np.full(map.shape, False)
        # visited[0][0] = True
        res = agent1.dfsCNN(model, gridWorld, map, (0, 0), trajectory, soft_max, visited, visit)
        # print(trajectory)
        if res:
            print("{i}th trial len={len}".format(i=idx+1, len=len(trajectory)))
            lenList.append(len(trajectory))
        else:
            print(False)
            lenList.append(-1)
    np.save("./pics/proj1/over{modelType}len".format(modelType=modelType), lenList)


if __name__ == '__main__':
    # mimic2("CNN")
    # mimic2("NN")
    mimic1("CNN")
    # mimic1("NN")

    # mazes = np.load("maps/test_30x30dim.npy")
    # lenList = []
    # print(mazes.shape)
    # for i in range(len(mazes[:])):
    #     map = mazes[i]
    #     algo = InferenceSearch(map)
    #     res = algo.run()
    #     print("{i}th len=".format(i=i + 1), len(algo.trajectory))
    #     lenList.append(len(algo.trajectory))
