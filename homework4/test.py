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
import matplotlib.pyplot as plt


def accProject1():
    trainNN = np.load("./pics/proj1/trainingNNacc.npy")
    testNN = np.load("./pics/proj1/testingNNacc.npy")
    plt.plot(range(1, len(trainNN)+1), trainNN, color="red")
    plt.plot(range(1, len(testNN) + 1), testNN)
    plt.legend(["trainNN", "testNN"])
    plt.xlabel("epochs")
    plt.title("NN accuracy in train and test")
    plt.show()

    lossNN = np.load("./pics/proj1/NNloss.npy")
    plt.plot(range(1, len(lossNN) + 1), lossNN)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("NN average loss in training")
    plt.show()

    # lossCNN = np.load("./pics/proj1/CNNloss.npy")
    # plt.plot(range(1, len(lossCNN) + 1), lossCNN)
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.title("CNN average loss in training")
    # plt.show()
    #
    # trainCNN = np.load("./pics/proj1/trainingCNNacc.npy")
    # testCNN = np.load("./pics/proj1/testingCNNacc.npy")
    # plt.plot(range(1, len(trainCNN) + 1), trainCNN, color="red")
    # plt.plot(range(1, len(testCNN) + 1), testCNN)
    # plt.legend(["trainCNN", "testCNN"])
    # plt.xlabel("epochs")
    # plt.title("CNN accuracy in train and test")
    # plt.show()
    # print(len(testCNN))  # 40


def accProject2():

    # trainNN = np.load("./pics/proj2/trainingNNacc.npy")
    # testNN = np.load("./pics/proj2/testingNNacc.npy")
    # plt.plot(range(1, len(trainNN)+1), trainNN, color="red")
    # plt.plot(range(1, len(testNN) + 1), testNN)
    # plt.legend(["trainNN", "testNN"])
    # plt.xlabel("epochs")
    # plt.title("NN accuracy in train and test")
    # plt.show()

    # lossNN = np.load("./pics/proj2/NNloss.npy")
    # plt.plot(range(1, len(lossNN) + 1), lossNN)
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.title("NN average loss in training")
    # plt.show()
    #
    # lossCNN = np.load("./pics/proj2/CNNloss.npy")
    # plt.plot(range(1, len(lossCNN) + 1), lossCNN)
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.title("CNN average loss in training")
    # plt.show()

    trainCNN = np.load("./pics/proj2/trainingCNNacc.npy")
    testCNN = np.load("./pics/proj2/testingCNNacc.npy")
    plt.plot(range(1, len(trainCNN)+1), trainCNN, color="red")
    plt.plot(range(1, len(testCNN) + 1), testCNN)
    plt.legend(["trainCNN", "testCNN"])
    plt.xlabel("epochs")
    plt.title("CNN accuracy in train and test")
    plt.show()
    print(len(testCNN))  # 40


def proj1len():
    cnn = np.load("./pics/proj1/CNNlen.npy")
    nn = np.load("./pics/proj1/NNlen.npy")
    agent = np.load("./pics/proj1/agentLen.npy")
    totalcnn, totalnn, origin = 0, 0, 0
    num = 0
    for i in range(len(cnn)):
        if cnn[i]!=-1 and nn[i]!=-1:
            num += 1
            totalcnn += cnn[i]
            totalnn += nn[i]
            origin += agent[i]
    print(totalcnn/num, totalnn/num, origin/num, num)
    total = [totalcnn/num, totalnn/num, origin/num]
    plt.bar(range(3), total, width=0.5)
    plt.ylabel("average trajectory length")
    plt.xticks(range(3), ["CNN", "NN", "original agent"])
    plt.title("trajectory length of 3 models")
    plt.show()


def proj2len():
    cnn = np.load("./pics/proj2/CNNlen.npy")
    nn = np.load("./pics/proj2/NNlen.npy")
    agent = np.load("./pics/proj2/agentLen.npy")
    totalcnn, totalnn, origin = 0, 0, 0
    num = 0
    for i in range(len(cnn)):
        if cnn[i]!=-1 and nn[i]!=-1:
            num += 1
            totalcnn += cnn[i]
            totalnn += nn[i]
            origin += agent[i]
    print(totalcnn/num, totalnn/num, origin/num, num)
    total = [totalcnn/num, totalnn/num, origin/num]
    plt.bar(range(3), total, width=0.5)
    plt.ylabel("average trajectory length")
    plt.xticks(range(3), ["CNN", "NN", "original agent"])
    plt.title("trajectory length of 3 models")
    plt.show()


def proj2Over():
    cnn = np.load("./pics/proj2/CNNlen.npy")
    tmpCnn = np.load("./pics/proj2/overCNNlen.npy")
    totalcnn, tmplen = 0, 0
    num = 0
    for i in range(len(cnn)):
        if cnn[i] != -1 and tmpCnn[i] != -1:
            num += 1
            totalcnn += cnn[i]
            tmplen += tmpCnn[i]
    print(totalcnn / num, tmplen / num, num)

    # cnn1 = np.load("./pics/proj1/CNNlen.npy")
    # tmpCnn1 = np.load("./pics/proj1/overCNNlen.npy")
    # totalcnn1, tmplen1 = 0, 0
    # num1 = 0
    # for i in range(len(cnn1)):
    #     if cnn1[i] != -1 and tmpCnn1[i] != -1:
    #         num1 += 1
    #         totalcnn1 += cnn1[i]
    #         tmplen1 += tmpCnn1[i]
    # total1 = [totalcnn1 / num1, tmplen1 / num1]

    total = [125, 136, totalcnn / num, tmplen / num]
    plt.bar(range(4), total, width=0.1)
    plt.ylabel("average trajectory length")
    plt.xticks(range(4), ["proj1:acc=0.90", "proj1:acc=0.97", "proj2:acc=0.89", "proj2:acc=0.96"])
    plt.title("compare performance in test and practice")
    plt.show()

    # tmp = np.load("./pics/proj2/overCNNtestAcc.npy")
    # testCNN = np.load("./pics/proj2/testingCNNacc.npy")
    # newAcc = list(testCNN)+list(tmp)
    # plt.plot(range(1, len(newAcc) + 1), newAcc, color="red")
    # plt.plot(range(1, len(testCNN) + 1), testCNN)
    # plt.legend(["better performance in test", "testCNN"])
    # plt.xlabel("epochs")
    # plt.title("compare performance in test and practice")
    # plt.show()


if __name__ == '__main__':
    # accProject1()
    # accProject2()

    # proj2len()
    proj2Over()
