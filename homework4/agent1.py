# coding = utf-8

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.functional import cross_entropy
from sklearn import metrics
import copy
from transformers import AdamW

if torch.cuda.is_available():
    dev = "cuda:0"
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


class Agent1NN(nn.Module):
    def __init__(self):
        super(Agent1NN, self).__init__()
        self.fc1 = nn.Linear(30 * 30 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.activate = nn.ReLU()
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


class Agent1CNN(nn.Module):
    def __init__(self):
        super(Agent1CNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            # Defining another 2D convolution layer
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
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
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        y = self.linear_layers(x)

        return y


def train(data, batch_size=16, shuffle=True, learning_rate=0.001, from_scratch=True, num_iteration=200, modelType="CNN"):
    train_x, train_y = np.array(data[0]), np.array(data[1])
    # train_x = train_x.reshape(len(train_x), -1)
    dataLoader = DataLoader(MyDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle)
    if modelType == "CNN":
        model = Agent1CNN()
    else:
        model = Agent1NN()
    model.to(device)
    if not from_scratch:
        if modelType == "CNN":
            state_dict = torch.load("./model/proj1/CNN_40.pt")
        else:
            state_dict = torch.load("./model/proj1/NN_200.pt")
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
        for j, (x, label) in enumerate(dataLoader):
            optimizer.zero_grad()
            output = model(x.permute(0, 3, 1, 2))
            loss = crossEntropy(output, label)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(soft_max(output), dim=1)
            # predList += list(pred.cpu().numpy())
            # labelList += list(label.cpu().numpy())
            # accurate = metrics.accuracy_score(label.cpu(), pred.cpu())
            # print(accurate)
            if (j + 1) % 10 == 0:
                # print(loss.item())
                accuracy = metrics.accuracy_score(label.cpu(), pred.cpu())
                print("loss: {loss},  accuracy: {acc}".format(acc=accuracy, loss=loss.item()))
        # accuracy = metrics.accuracy_score(labelList, predList)
        # print("accuracy: {acc}".format(acc=accuracy))
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    filePath = "./model/proj1/{modelName}_{itr}.pt".format(itr=num_iteration + 200, modelName=modelType)
    torch.save(state_dict, filePath)
    # torch.save(state_dict, "./model/agent1CNN_{itr}.pt".format(itr=num_iteration))


def eval(data, batch_size=16):
    state_dict = torch.load("./model/agent1CNN.pt")
    model = Agent1CNN()
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


def inputTransform(status, cur, visit):
    x = np.eye(3)[status]
    place = np.zeros(status.shape)
    # print(prev)
    place[cur[0]][cur[1]] = 1
    place = np.expand_dims(place, 2)
    x = np.concatenate([x, place], axis=-1)
    # x = np.concatenate([x, np.expand_dims(visitCount, 2)], axis=-1)
    # x = np.concatenate([x, np.expand_dims(visit, 2)], axis=-1)
    return np.array([x])


def getAllNeighbors(x, m, n):
    neighbors = []
    for dir in directions:
        x1 = x[0] + dir[0]
        y1 = x[1] + dir[1]
        if 0 <= x1 < m and 0 <= y1 < n:
            neighbors.append((x1, y1))
    return neighbors


def updateNeighbor(gridWorld, map, cur):
    x, y = cur
    for (i, j) in getAllNeighbors(cur, 30, 30):
        gridWorld[i][j] = map[i][j]


def dfsCNN(model, gridWorld, map, cur, trajectory, soft_max, visited, visit):
    if len(trajectory) > 500:
        return False
    if cur == (29, 29):
        return True
    # if map[cur[0]][cur[1]] == 1 or visited[cur[0]][cur[1]]:
    #     return False
    trajectory.append(cur)
    visit[cur[0]][cur[1]] = 1
    # print(cur)
    # visited[cur[0]][cur[1]] = True
    updateNeighbor(gridWorld, map, cur)
    inputX = inputTransform(status=gridWorld, cur=cur, visit=visit)
    dataLoader = DataLoader(MyDataset(inputX, np.zeros([len(gridWorld)])), batch_size=1)
    output, pred, dir = None, None, None
    for j, (x, label) in enumerate(dataLoader):
        output = model(x.permute(0, 3, 1, 2))
    prob, sortedIdx = torch.sort(soft_max(output), descending=True)
    # print(sortedIdx)
    for idx in sortedIdx[0]:
        # pred = sortedIdx[0][idx]
        # print(pred)
        dir = idx_gird.get(idx.item())
        x1, y1 = (dir[0] + cur[0], dir[1] + cur[1])
        if 0 <= x1 < 30 and 0 <= y1 < 30 and not visited[x1][y1] and map[x1][y1] != 1:
            visited[x1][y1] = True
            if dfsCNN(model, gridWorld, map, (x1, y1), trajectory, soft_max, visited, visit):
                return True
            visited[x1][y1] = False
    # visited[cur[0]][cur[1]] = False
    return False


def repeatedCNN(modelType="CNN"):
    mazes = np.load("maps/30x30dim.npy")
    print(mazes.shape)
    soft_max = nn.Softmax(1)
    for idx, map in enumerate(mazes[:]):
        gridWorld = np.full(map.shape, 2)
        # gridWorld[0][0], gridWorld[0][1], gridWorld[1][0] = 0, map[0][1], map[1][0]
        visitCount = np.zeros(map.shape)
        visitCount[0][0] = 1
        visit = np.zeros(map.shape)
        cur = (0, 0)
        inputX = inputTransform(status=gridWorld, cur=(0, 0), visit=visit)
        if modelType=="CNN":
            state_dict = torch.load("./model/proj1/CNN_50.pt")
            model = Agent1CNN()
        else:
            state_dict = torch.load("./model/proj1/NN_200.pt")
            model = Agent1NN()
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()
        dataLoader = DataLoader(MyDataset(inputX, np.zeros([len(gridWorld)])), batch_size=1)
        # trajectory = [cur]
        # while True:
        #     output, pred, dir = None, None, None
        #     for j, (x, label) in enumerate(dataLoader):
        #         output = nn.Softmax(dim=1)(model(x.permute(0, 3, 1, 2)))
        #         # pred = torch.argmax(soft_max(output), 1)
        #     prob, sortedIdx = torch.sort(output, descending=True)
        #     k = 0
        #     x1, y1 = 0, 0
        #     while k < sortedIdx.size(1):  # sortedIdx (samples, num_classes)
        #         pred = sortedIdx[0][k]
        #         dir = idx_gird.get(pred.item())
        #         x1, y1 = (dir[0] + cur[0], dir[1] + cur[1])
        #         if 0 <= x1 < 30 and 0 <= y1 < 30 and gridWorld[x1][y1] != 1:
        #             break
        #         k += 1
        #     visitCount[x1][y1] += 1
        #     cur = (x1, y1)
        #     updateNeighbor(gridWorld, map, cur)
        #     trajectory.append(cur)
        #     if cur == (29, 29):
        #         break
        #     print(gridWorld.shape, map.shape, cur)
        #     inputX = inputTransform(status=gridWorld, cur=(0, 0), visit=None)
        #     dataLoader = DataLoader(MyDataset(inputX, np.zeros([len(gridWorld)])), batch_size=1)
        # print(trajectory)

        trajectory = []
        visited = np.full(map.shape, False)
        # visited[0][0] = True
        res = dfsCNN(model, gridWorld, map, (0, 0), trajectory, soft_max, visited, visit)
        # print(trajectory)
        if res:
            print("{i}th trial len={len}".format(i=idx+1, len=len(trajectory)))
        else:
            print(False)


if __name__ == '__main__':
    # dataX, dataY = np.load("./data/proj1/map_1.npy"), np.load("./data/proj1/label_1.npy")
    # print(dataX.shape)
    # for i in range(1, 200):
    #     x, y = np.load("./data/proj1/map_{t}.npy".format(t=i + 1)), np.load("./data/proj1/label_{t}.npy".format(t=i + 1))
    #     dataX = np.concatenate([x, dataX], axis=0)
    #     dataY = np.concatenate([y, dataY], axis=0)
    # print(dataX.shape)
    # print(np.unique(dataY, return_counts=True))
    # train((dataX, dataY), batch_size=256, shuffle=True, learning_rate=0.001, from_scratch=True, num_iteration=50, modelType="CNN")   # batch_size=256, shuffle=True, learning_rate=0.001
    # train((dataX, dataY), batch_size=512, shuffle=True, learning_rate=0.001, from_scratch=False, num_iteration=100, modelType="NN")  #
    # eval((dataX, dataY), 512)
    #
    # repeatedCNN("CNN")
    repeatedCNN("NN")
