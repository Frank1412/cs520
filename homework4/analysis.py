# coding = utf-8

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.functional import cross_entropy
from sklearn import metrics
import copy
from transformers import AdamW
from agent2 import *
from agent1 import *
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


def eval(model, testLoader, soft_max):
    model.eval()
    predList, labelList = [], []
    for j, (x, label) in enumerate(testLoader):
        output = model(x.permute(0, 3, 1, 2))
        pred = torch.argmax(soft_max(output), dim=1)
        predList += list(pred.cpu().numpy())
        labelList += list(label.cpu().numpy())
    accuracy = metrics.accuracy_score(labelList, predList)
    model.train()
    return accuracy


def train2(data, batch_size=16, shuffle=True, learning_rate=0.001, from_scratch=True, num_iteration=200, modelType="CNN"):
    train_x, train_y = np.array(data[0]), np.array(data[1])
    # train_x = train_x.reshape(len(train_x), -1)
    dataLoader = DataLoader(MyDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle)
    test_dataX, test_dataY = np.load("./data/proj2/test_dataX.npy"), np.load("./data/proj2/test_dataY.npy")
    testLoader = DataLoader(MyDataset(test_dataX[:10000], test_dataY[:10000]), batch_size=256, shuffle=shuffle)
    model = None
    if modelType == "CNN":
        model = Agent2CNN()
    elif modelType == "NN":
        model = Agent2NN()
    model.to(device)
    if not from_scratch:
        state_dict = None
        if modelType == "CNN":
            # state_dict = torch.load("./model/proj2/3layerCNN_10.pt")
            state_dict = torch.load("./pics/proj2/bestCNN.pt")
        elif modelType == "NN":
            state_dict = torch.load("./model/proj2/2layerNN_800.pt")
        model.load_state_dict(state_dict["model"])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(state_dict["optimizer"])
        model.train()
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    soft_max = nn.Softmax(1)
    crossEntropy = nn.CrossEntropyLoss()
    lossList, accList, trainAccList = [], [], []
    for i in range(num_iteration):
        print("================{i} iteration================ ".format(i=i + 1))
        predList = []
        labelList = []
        avgLoss, num = 0.0, 0
        for j, (x, label) in enumerate(dataLoader):
            optimizer.zero_grad()
            output = model(x.permute(0, 3, 1, 2))
            loss = crossEntropy(output, label)
            loss.backward()
            optimizer.step()
            avgLoss += loss.item()
            num += 1
            pred = torch.argmax(soft_max(output), dim=1)
            predList += list(pred.cpu().numpy())
            labelList += list(label.cpu().numpy())
            # if (j + 1) % 40 == 0:
            #     accuracy = metrics.accuracy_score(label.cpu(), pred.cpu())
            #     print("loss: {loss},  accuracy: {acc}".format(acc=accuracy, loss=loss.item()))

        accuracy = metrics.accuracy_score(labelList, predList)
        acc = eval(model, testLoader, soft_max)
        trainAccList.append(acc)
        # print("############## average loss={loss}, acc={acc}".format(loss=avgLoss / num, acc=acc))
        # print(avgLoss/num, acc)
        lossList.append(avgLoss/num)
        accList.append(accuracy)
        # if acc >= 0.9:
        #     break
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    filePath = "./pics/proj2/overbest{modelType}.pt".format(modelType=modelType)
    torch.save(state_dict, filePath)

    # np.save("./pics/proj2/{modelType}loss".format(modelType=modelType), lossList)
    # np.save("./pics/proj2/{modelType}acc".format(modelType=modelType), accList)
    print(lossList)
    print(accList)

    np.save("./pics/proj2/overCNNtestAcc", trainAccList)
    np.save("./pics/proj2/overCNNtrainAcc", accList)
    np.save("./pics/proj2/overCNNloss", lossList)
    return trainAccList, lossList, accList


def train1(data, batch_size=16, shuffle=True, learning_rate=0.001, from_scratch=True, num_iteration=200, modelType="CNN"):
    train_x, train_y = np.array(data[0]), np.array(data[1])
    # train_x = train_x.reshape(len(train_x), -1)
    dataLoader = DataLoader(MyDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle)
    test_dataX, test_dataY = np.load("./data/proj1/test_dataX.npy"), np.load("./data/proj1/test_dataY.npy")
    testLoader = DataLoader(MyDataset(test_dataX[:10000], test_dataY[:10000]), batch_size=512, shuffle=shuffle)
    if modelType == "CNN":
        model = Agent1CNN()
    else:
        model = Agent1NN()
    model.to(device)
    if not from_scratch:
        if modelType == "CNN":
            # state_dict = torch.load("./model/proj1/CNN_40.pt")
            state_dict = torch.load("./model/proj1/CNN_50.pt")
        else:
            state_dict = torch.load("./model/proj1/NN_200.pt")
        model.load_state_dict(state_dict["model"])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(state_dict["optimizer"])
        model.train()
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    soft_max = nn.Softmax(1)
    crossEntropy = nn.CrossEntropyLoss()
    lossList, accList, trainAccList = [], [], []
    for i in range(num_iteration):
        print("================{i} iteration================ ".format(i=i + 1))
        predList = []
        labelList = []
        avgLoss, num = 0.0, 0
        for j, (x, label) in enumerate(dataLoader):
            optimizer.zero_grad()
            output = model(x.permute(0, 3, 1, 2))
            loss = crossEntropy(output, label)
            loss.backward()
            optimizer.step()
            num += 1
            avgLoss += loss.item()
            pred = torch.argmax(soft_max(output), dim=1)
            predList += list(pred.cpu().numpy())
            labelList += list(label.cpu().numpy())
            # if (j + 1) % 10 == 0:
            #     # print(loss.item())
            #     accuracy = metrics.accuracy_score(label.cpu(), pred.cpu())
            #     print("loss: {loss},  accuracy: {acc}".format(acc=accuracy, loss=loss.item()))
        acc = eval(model, testLoader, soft_max)
        trainAccList.append(acc)
        accuracy = metrics.accuracy_score(labelList, predList)
        lossList.append(avgLoss / num)
        accList.append(accuracy)

    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    filePath = "./pics/proj1/overbest{modelType}.pt".format(modelType=modelType)
    torch.save(state_dict, filePath)
    # np.save("./pics/proj1/{modelType}loss".format(modelType=modelType), lossList)
    # np.save("./pics/proj1/{modelType}acc".format(modelType=modelType), accList)
    print(lossList, accList)
    return trainAccList, accList, lossList


def acc_loss_plot():
    # dataX, dataY = np.load("./data/proj2/dataX.npy"), np.load("./data/proj2/dataY.npy")
    # print(dataX.shape)
    # print(np.unique(dataY, return_counts=True))
    # return train2((dataX[:], dataY[:]), batch_size=256, shuffle=True, learning_rate=0.001, from_scratch=False, num_iteration=10, modelType="CNN")
    # return train2((dataX, dataY), batch_size=512, shuffle=True, learning_rate=0.0001, from_scratch=True, num_iteration=700, modelType="NN")

    dataX, dataY = np.load("./data/proj1/dataX.npy"), np.load("./data/proj1/dataY.npy")
    print(dataX.shape)
    print(np.unique(dataY, return_counts=True))
    return train1((dataX, dataY), batch_size=512, shuffle=True, learning_rate=0.001, from_scratch=False, num_iteration=20, modelType="CNN")  # batch_size=256, shuffle=True, learning_rate=0.001
    # return train1((dataX, dataY), batch_size=1024, shuffle=True, learning_rate=0.0001, from_scratch=True, num_iteration=300, modelType="NN")  #


if __name__ == '__main__':
    # trainAccList, accList, lossList = acc_loss_plot()
    trainAccList, accList, lossList = acc_loss_plot()