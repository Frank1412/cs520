# coding = utf-8

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.functional import cross_entropy
from sklearn import metrics


if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

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
        self.fc1 = nn.Linear(30*30*4, 4096)
        self.fc2 = nn.Linear(4096, 768)
        self.cls = nn.Linear(768, 4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        y = self.cls(out)
        return y


class Agent1CNN(nn.Module):
    def __init__(self):
        super(Agent1CNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 4)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        # print(x.shape, x.size())
        x = x.view(x.size(0), -1)
        y = self.linear_layers(x)
        # x = x.view(, -1)
        return y


def train(data, batch_size=16, shuffle=True, learning_rate=0.001):
    train_x, train_y = np.array(data[0]), np.array(data[1])
    train_x = train_x.reshape(len(train_x), -1)
    dataLoader = DataLoader(MyDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle)
    # model = Agent1CNN()
    model = Agent1NN()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    crossEntropy = nn.CrossEntropyLoss()
    for i in range(1000):
        print("================{i} iteration================ ".format(i=i+1))
        PosNum = 0
        for j, (x, label) in enumerate(dataLoader):
            optimizer.zero_grad()
            output = model(x.transpose(-1, 1))
            loss = crossEntropy(output, label)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(output, dim=1)
            accurate = metrics.accuracy_score(label.cpu(), pred.cpu())
            # print(accurate)
            if (j+1) % 10 == 0:
                print(loss.item())
                print(accurate)
            # pred = torch.argmax(output, dim=1)
            # print(pred, label)
            # accurate = metrics.accuracy_score(label.cpu(), pred.cpu())
            # print("==========batch=%d   accurate == %f" % (j + 1, accurate))


if __name__ == '__main__':
    dataX, dataY = np.load("./data/map_1.npy"), np.load("./data/label_1.npy")
    for i in range(1, 50):
        x, y = np.load("./data/map_{t}.npy".format(t=i+1)), np.load("./data/label_{t}.npy".format(t=i+1))
        dataX = np.concatenate([x, dataX], axis=0)
        dataY = np.concatenate([y, dataY], axis=0)
    # print(dataX.dtype)
    train((dataX, dataY), batch_size=512, shuffle=True, learning_rate=0.005)