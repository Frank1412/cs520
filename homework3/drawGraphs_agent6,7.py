import pandas as pd
import ast
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import gc

import csv

if __name__ == '__main__':
    x = []
    movementListAgent6 = []
    examinationsListAgent6 = []
    ratioListAgent6 = []
    sumListAgent6 = []

    movementListAgent7 = []
    examinationsListAgent7 = []
    ratioListAgent7 = []
    sumListAgent7 = []

    for i in range(6, 11):
        df = pd.read_csv(r"./agent_8_results/agent_6，7，8_map_{}.csv".format(i))
        df = df[df['agent'] == 6]
        #df = df.loc[(df['movement']>500) & (df['movement']<5000)]
        df['movement'] = df['movement'].astype('float32')
        df['examinations'] = df['examinations'].astype('float32')
        df['ratio'] = df['ratio'].astype('float32')
        df['sum'] = df['sum'].astype('float32')
        averageMovement = mean(df['movement'].tolist())
        averageExaminations = mean(df['examinations'].tolist())
        averageRatio = mean(df['ratio'].tolist())
        averageSum = mean(df['sum'].tolist())
        movementListAgent6.append(averageMovement)
        examinationsListAgent6.append(averageExaminations)
        ratioListAgent6.append(averageRatio)
        sumListAgent6.append(averageSum)
        # b = df['3'].tolist()

        df = pd.read_csv(r"./agent_8_results/agent_6，7，8_map_{}.csv".format(i))
        df = df[df['agent'] == 7]
        #df = df.loc[(df['movement'] > 500) & (df['movement'] < 5000)]
        df['movement'] = df['movement'].astype('float32')
        df['examinations'] = df['examinations'].astype('float32')
        df['ratio'] = df['ratio'].astype('float32')
        df['sum'] = df['sum'].astype('float32')
        averageMovement = mean(df['movement'].tolist())
        averageExaminations = mean(df['examinations'].tolist())
        averageRatio = mean(df['ratio'].tolist())
        averageSum = mean(df['sum'].tolist())
        movementListAgent7.append(averageMovement)
        examinationsListAgent7.append(averageExaminations)
        ratioListAgent7.append(averageRatio)
        sumListAgent7.append(averageSum)

    size = 5
    x = np.arange(size)
    labels =['map6', 'map7', 'map8', 'map9', 'map10']
    #labels = ['map1', 'map2', 'map3', 'map4', 'map5', 'map6', 'map7', 'map8', 'map9', 'map10']

    width = 0.35
    #total_width, n = 0.5, 2
    #width = total_width / n
   # x = x - (total_width - width) / 2

    plt.bar(x - width/2, movementListAgent6, width=width, label='agent 6')
    plt.bar(x + width/2, movementListAgent7, width=width, label='agent 7')
    plt.xlabel('map numbers')
    plt.ylabel('movements')
    plt.xticks(x, labels)
    plt.legend()
    plt.show()

    plt.bar(x - width / 2, examinationsListAgent6, width=width, label='agent 6')
    plt.bar(x + width / 2, examinationsListAgent7, width=width, label='agent 7')
    plt.xlabel('map numbers')
    plt.ylabel('examinations')
    plt.xticks(x, labels)
    plt.legend()
    plt.show()

    plt.bar(x - width / 2, ratioListAgent6, width=width, label='agent 6')
    plt.bar(x + width / 2, ratioListAgent7, width=width, label='agent 7')
    plt.xlabel('map numbers')
    plt.ylabel('movements/examinations')
    plt.xticks(x, labels)
    plt.legend()
    plt.show()

    plt.bar(x - width / 2, sumListAgent6, width=width, label='agent 6')
    plt.bar(x + width / 2, sumListAgent7, width=width, label='agent 7')
    plt.xlabel('map numbers')
    plt.ylabel('number of actions')
    plt.xticks(x, labels)
    plt.legend()
    plt.show()