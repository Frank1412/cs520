import ast
import gc
from probImpl import *
import csv

if __name__ == '__main__':
    allMaze = loadMaze("./full_connected_maps", "dim50_36.json")
    map, terrain = allMaze[0], allMaze[1]
    terrainType
    #print(terrain[0][2])

    df = pd.read_csv(r"./agent_8_results/agent_6，7，8_map_36.csv")
    df = df[df['agent'] == 6]
    df['start'] = df['start'].apply(ast.literal_eval)
    df['target'] = df['target'].apply(ast.literal_eval)
    a = df['start'].tolist()
    b = df['target'].tolist()

    for i in range(len(b)):
        print(terrain[b[i][0]][b[i][1]])