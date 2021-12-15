# coding = utf-8

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.functional import cross_entropy
from sklearn import metrics
import copy
from transformers import AdamW
from util import *
from torchsummary import summary

if torch.cuda.is_available():
    dev = "cuda:0"
    # dev = "cpu"
else:
    dev = "cpu"
device = torch.device(dev)
cls = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
idx_gird = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.x = torch.FloatTensor(x).to(device)
        self.y = torch.LongTensor(y).to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Agent2NN(nn.Module):
    def __init__(self):
        super(Agent2NN, self).__init__()
        self.fc1 = nn.Linear(30 * 30 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.activate = nn.ReLU()
        # self.batchNorm = nn.BatchNorm1d()
        self.cls = nn.Linear(1024, 4)

    def forward(self, x):
        out = self.fc1(x.reshape(x.size(0), -1))
        out = self.activate(out)
        out = self.fc2(out)
        out = self.activate(out)
        out = self.fc3(out)
        out = self.activate(out)
        y = self.cls(out)
        return y


class Agent2CNN(nn.Module):
    def __init__(self):
        super(Agent2CNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            # Defining another 2D convolution layer
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4)
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(64 * 7 * 7, 128)
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4*30*30, 128)
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(128, 4)
        # )
        # self.fc4 = nn.Linear(256, 4)

    def forward(self, x):
        # print(x.shape)
        # x2 = self.fc2(x.reshape(x.size(0), -1))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        y = self.linear_layers(x)

        # x1 = self.fc1(x.reshape(x.size(0), -1))
        # xx = torch.cat([x1, x2], dim=1)
        # y = self.fc4(xx)
        return y


def train(data, batch_size=16, shuffle=True, learning_rate=0.001, from_scratch=True, num_iteration=200, modelType="CNN"):
    train_x, train_y = np.array(data[0]), np.array(data[1])
    # train_x = train_x.reshape(len(train_x), -1)
    dataLoader = DataLoader(MyDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle)
    model = None
    if modelType == "CNN":
        model = Agent2CNN()
    elif modelType == "NN":
        model = Agent2NN()
    model.to(device)
    if not from_scratch:
        state_dict = None
        if modelType == "CNN":
            state_dict = torch.load("./model/proj2/3layerCNN_10.pt")
        elif modelType == "NN":
            state_dict = torch.load("./model/proj2/2layerNN_800.pt")
        model.load_state_dict(state_dict["model"])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(state_dict["optimizer"])
        model.train()
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # model = Agent1NN()
    # model = torch.load("./model/agent1CNN.pt")
    # model.eval()
    soft_max = nn.Softmax(1)
    crossEntropy = nn.CrossEntropyLoss()
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
            # predList += list(pred.cpu().numpy())
            # labelList += list(label.cpu().numpy())
            # accurate = metrics.accuracy_score(label.cpu(), pred.cpu())
            # print(accurate)
            if (j + 1) % 10 == 0:
                accuracy = metrics.accuracy_score(label.cpu(), pred.cpu())
                print("loss: {loss},  accuracy: {acc}".format(acc=accuracy, loss=loss.item()))
        print("############## average loss={loss}".format(loss=avgLoss / num))
        # accuracy = metrics.accuracy_score(labelList, predList)
        # print("accuracy: {acc}".format(acc=accuracy))
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    filePath = "./model/proj2/3layer{modelName}_{itr}.pt".format(itr=num_iteration + 10, modelName=modelType)
    torch.save(state_dict, filePath)


def eval(data, batch_size=16):
    state_dict = torch.load("./model/proj2/agent2CNN.pt")
    model = Agent2CNN()
    model.load_state_dict(state_dict["model"])
    model.to(device)
    model.eval()
    train_x, train_y = np.array(data[0]), np.array(data[1])
    # train_x = train_x.reshape(len(train_x), -1)
    dataLoader = DataLoader(MyDataset(train_x, train_y), batch_size=batch_size)
    for j, (x, label) in enumerate(dataLoader):
        output = model(x.transpose(-1, 1))
        pred = torch.argmax(output, dim=1)
        accurate = metrics.accuracy_score(label.cpu(), pred.cpu())
        print("accuracy: {acc}".format(acc=accurate))


def inputTransform(status, cur, N, C, B, E, H):
    x = np.eye(3)[status]
    place = np.zeros(status.shape)
    # print(prev)
    place[cur[0]][cur[1]] = 1
    place = np.expand_dims(place, 2)
    # tmpN = np.expand_dims(N, 2)
    tmpC = np.expand_dims(C, 2)
    tmpB = np.expand_dims(B, 2)
    tmpE = np.expand_dims(E, 2)
    tmpH = np.expand_dims(H, 2)
    # x = np.concatenate([x, place, tmpN, tmpC, tmpB, tmpE, tmpH], axis=-1)
    x = np.concatenate([x, place, tmpC, tmpB, tmpE, tmpH], axis=-1)
    # x = np.concatenate([x, np.expand_dims(visitCount, 2)], axis=-1)
    # x = np.concatenate([x, np.expand_dims(visit, 2)], axis=-1)
    return np.array([x])


def updateUnknown(map, gridWorld, cur, N, C, B, E, H):
    x, y = cur
    m, n = map.shape
    if gridWorld[x][y] == 2:
        gridWorld[x][y] = map[x][y]
        if map[x][y] == 1:
            for (i, j) in get8Neighbors(cur, m, n):
                B[i][j] += 1
                H[i][j] -= 1
        else:
            C[x][y] = sense(cur, map)
            for (i, j) in get8Neighbors(cur, m, n):
                E[i][j] += 1
                H[i][j] -= 1


def getAllNeighbors(x, m, n):
    neighbors = []
    for dir in directions:
        x1 = x[0] + dir[0]
        y1 = x[1] + dir[1]
        if 0 <= x1 < m and 0 <= y1 < n:
            neighbors.append((x1, y1))
    return neighbors


def dfsCNN(model, gridWorld, map, cur, trajectory, soft_max, visited, N, C, B, E, H):
    trajectory.append(cur)
    if len(trajectory) > 1500:
        return False
    m, n = map.shape
    if cur == (m - 1, n - 1):
        return True
    updateUnknown(map, gridWorld, cur, N, C, B, E, H)
    if gridWorld[cur[0]][cur[1]] == 1:
        return False
    inputX = inputTransform(gridWorld, cur, N, C, B, E, H)
    dataLoader = DataLoader(MyDataset(inputX, np.zeros([len(gridWorld)])), batch_size=1)
    output, pred, dir = None, None, None
    for j, (x, label) in enumerate(dataLoader):
        output = model(x.permute(0, 3, 1, 2))
    prob, sortedIdx = torch.sort(soft_max(output), descending=True)

    reUpdate = False
    for idx in sortedIdx[0]:
        # pred = sortedIdx[0][idx]
        # print(pred)
        dir = idx_gird.get(idx.item())
        x1, y1 = (dir[0] + cur[0], dir[1] + cur[1])
        # print((x1, y1))
        # print(map[:3, :3])
        # print(gridWorld[:3, :3])
        # print(C[:3, :3])
        # print(B[:3, :3])
        # print(E[:3, :3])
        # print(H[:3, :3])
        if x1 < 0 or x1 >= m or y1 < 0 or y1 >= n or visited[x1][y1]:
            continue
        # if map[x1][y1] == 1:
        #     updateUnknown(map, gridWorld, (x1, y1), N, C, B, E, H)
        #     continue
        visited[x1][y1] = True
        if dfsCNN(model, gridWorld, map, (x1, y1), trajectory, soft_max, visited, N, C, B, E, H):
            return True
        trajectory.append(cur)
        visited[x1][y1] = False
    # trajectory.append(cur)
    return False


def repeatedCNN(modelType="CNN"):
    mazes = np.load("maps/test_30x30dim.npy")
    print(mazes.shape)
    soft_max = nn.Softmax(1)
    for idx, map in enumerate(mazes[0:]):
        # gridWorld = np.full(map.shape, 2)
        # gridWorld[0][0], gridWorld[0][1], gridWorld[1][0] = 0, map[0][1], map[1][0]
        gridWorld, N, C, B, E, H = initialize(map)
        visit = np.zeros(map.shape)
        cur = (0, 0)
        if modelType == "CNN":
            state_dict = torch.load("./model/proj2/3layerCNN_15.pt")
            model = Agent2CNN()
        else:
            state_dict = torch.load("./model/proj2/2layerNN_800.pt")
            model = Agent2NN()
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()

        # updateUnknown(map, gridWorld, cur, N, C, B, E, H)
        # inputX = inputTransform(gridWorld, cur, N, C, B, E, H)
        # dataLoader = DataLoader(MyDataset(inputX, np.zeros([len(gridWorld)])), batch_size=1)
        # pre = None
        # trajectory = [cur]
        # while True:
        #     if len(trajectory) > 3000:
        #         break
        #     output, pred, dir = None, None, None
        #     for j, (x, label) in enumerate(dataLoader):
        #         output = nn.Softmax(dim=1)(model(x.permute(0, 3, 1, 2)))
        #         # pred = torch.argmax(soft_max(output), 1)
        #     prob, sortedIdx = torch.sort(output, descending=True)
        #     k = 0
        #     x1, y1 = 0, 0
        #     # print(map[:3, :3])
        #     # print(gridWorld[:3, :3])
        #     # print(C[:3, :3])
        #     # print(B[:3, :3])
        #     # print(E[:3, :3])
        #     # print(H[:3, :3])
        #     while k < sortedIdx.size(1):  # sortedIdx (samples, num_classes)
        #         pred = sortedIdx[0][k]
        #         dir = idx_gird.get(pred.item())
        #         x1, y1 = (dir[0] + cur[0], dir[1] + cur[1])
        #         print((x1, y1))
        #         if 0 <= x1 < 30 and 0 <= y1 < 30:
        #             break
        #         k += 1
        #     pre = cur
        #     cur = (x1, y1)
        #     updateUnknown(map, gridWorld, cur, N, C, B, E, H)
        #     trajectory.append(cur)
        #     if cur == (29, 29):
        #         break
        #     if gridWorld[x1][y1] == 1:
        #         cur = pre
        #         trajectory.append(cur)
        #     # print(gridWorld.shape, map.shape, cur)
        #     inputX = inputTransform(gridWorld, cur, N, C, B, E, H)
        #     dataLoader = DataLoader(MyDataset(inputX, np.zeros([len(gridWorld)])), batch_size=1)
        # if len(trajectory) > 3000:
        #     print(False)
        # else:
        #     print(True, len(trajectory))

        trajectory = []
        visited = np.full(map.shape, False)
        # visited[0][0] = True
        res = dfsCNN(model, gridWorld, map, (0, 0), trajectory, soft_max, visited, N, C, B, E, H)
        # print(trajectory)
        if res:
            print("{i}th trial len={len}".format(i=idx + 1, len=len(trajectory)))
        else:
            print(False)


if __name__ == '__main__':

    # dataX, dataY = np.load("./data/proj2/dataX.npy"), np.load("./data/proj2/dataY.npy")
    # print(dataX.shape)
    # print(np.unique(dataY, return_counts=True))
    # train((dataX, dataY), batch_size=512, shuffle=True, learning_rate=0.001, from_scratch=False, num_iteration=5, modelType="CNN")
    # batch_size=256, shuffle=True, learning_rate=0.001
    # train((dataX, dataY), batch_size=512, shuffle=True, learning_rate=0.0001, from_scratch=False, num_iteration=200, modelType="NN")
    # eval((dataX, dataY), 512)
    #
    repeatedCNN("CNN")
    # repeatedCNN("NN")
