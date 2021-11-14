import pandas as pd
import ast
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import gc

import csv

if __name__ == '__main__':






    numberOfActions0Agent6 = []
    numberOfActions1Agent6 = []
    numberOfActions2Agent6 = []

    numberOfActions0Agent7 = []
    numberOfActions1Agent7 = []
    numberOfActions2Agent7 = []



    Agent6 = []
    Agent7 = []



    list = [7,8]
    for i in list:
        df = pd.read_csv(r"./agent_8_results/agent_6，7，8_map_{}.csv".format(i))
        df = df[df['agent'] == 6]

        df['sum'] = df['sum'].astype('float32')
        df0 = df[df['terrain'] == 0]
        sum0 = mean(df0['sum'].tolist())
        print(df0['sum'].tolist())

        df1 = df[df['terrain'] == 1]
        sum1 = mean(df1['sum'].tolist())
        df2 = df[df['terrain'] == 2]
        sum2 = mean(df2['sum'].tolist())
        #print(df2)
        numberOfActions0Agent6.append(sum0)
        numberOfActions1Agent6.append(sum1)
        numberOfActions2Agent6.append(sum2)

        # b = df['3'].tolist()

        df = pd.read_csv(r"./agent_8_results/agent_6，7，8_map_{}.csv".format(i))
        df = df[df['agent'] == 7]

        df['sum'] = df['sum'].astype('float32')
        df0 = df[df['terrain'] == 0]
        #df0['sum'] = df0['sum'].astype('float32')
        sum0 = mean(df0['sum'].tolist())
        df1 = df[df['terrain'] == 1]
        #df1['sum'] = df1['sum'].astype('float32')
        sum1 = mean(df1['sum'].tolist())
        df2 = df[df['terrain'] == 2]
        #df2['sum'] = df2['sum'].astype('float32')
        sum2 = mean(df2['sum'].tolist())
        numberOfActions0Agent7.append(sum0)
        numberOfActions1Agent7.append(sum1)
        numberOfActions2Agent7.append(sum2)





    Agent6.append(mean(numberOfActions0Agent6))
    Agent6.append(mean(numberOfActions1Agent6))
    Agent6.append(mean(numberOfActions2Agent6))

    Agent7.append(mean(numberOfActions0Agent7))
    Agent7.append(mean(numberOfActions1Agent7))
    Agent7.append(mean(numberOfActions2Agent7))



    size = 3
    x = np.arange(size)
    labels = ['Flat', 'Hilly', 'Forest']
    #labels = ['map1', 'map2', 'map3', 'map4', 'map5', 'map6', 'map7', 'map8', 'map9', 'map10']

    width = 0.35
    # total_width, n = 0.5, 2
    # width = total_width / n
    # x = x - (total_width - width) / 2

    plt.bar(x - width / 2, Agent6, width=width, label='agent 6')
    plt.bar(x + width / 2, Agent7, width=width, label='agent 7')
    plt.xlabel('Terrain Type')
    plt.ylabel('Number of Actions')
    plt.xticks(x, labels)
    plt.legend()
    plt.show()