# encoding=utf-8

from A_star_algo import *
import time
import sys
import gc
sys.setrecursionlimit(1000000000)


def question1():

    # map = Map(150, 101)
    # map.setStartPoint(0, 0)
    # map.setObstacles(False, 0.1, obstacles)
    # map.setObstacles(True, 0.4)
    # algo = RepeatedAStar(map, 1)
    sum = 0
    for i in range(50):
        map = Map(101, 101)
        map.setObstacles(True, 0.1)
        algo = RepeatedAStar(map, 1)
        result = algo.run()
        print(result)
        if result:
            sum += 1
        map.reset()

        del map, algo
        gc.collect()
    print(sum/100)


def question4():
    map = Map(101, 101)
    map.setStartPoint(0, 0)
    # map.setObstacles(False, 0.1, obstacles)
    # map.setObstacles(True, 0.4)
    algo = AStar(map, 1)
    sum = 0
    for i in range(100):
        map = Map(101, 101)
        map.setObstacles(True, 0.34)
        algo = AStar(map, 1)
        result = algo.run()
        print(result)
        if result:
            sum += 1
        map.reset()
    print(sum / 100)
    path = algo.path
    last = map.end
    while last in path:
        # print(last, path[last])
        last = path[last]


def question5():
    for _ in range(3):
        map = Map(101, 101)
        map.setObstacles(True, 0.2)
        for i in range(3):
            algo = AStar(map, i + 1)
            startTime = time.time()
            result = algo.run()
            endTime = time.time()
            if (result == False): continue
            print(i+1, len(algo.trajectory))
            # if i==2:
            #     print(algo.trajectory)
            img = Image.fromarray(np.uint8(cm.gist_earth(map.map) * 255))
            path = algo.path
            img = np.array(img.convert('RGB'))
            last = map.end
            while last in path:
                img[last[0]][last[1]] = [0, 255, 255]
                last = path[last]
            start = map.getStartPoint()
            end = map.getEndPoint()
            img[start[0]][start[1]] = [255, 0, 0]
            img[end[0]][end[1]] = [255, 0, 0]
            plt.imshow(img)
            plt.grid(linewidth=1)
            method = {
                0: "Manhattan",
                1: "Euclidean",
                2: "Chebyshev"
            }
            plt.title(method.get(i) + ":" + str(endTime - startTime))
            plt.show()


def question6():
    p_list = np.linspace(0, 0.33, 34)  # 34
    # plt.plot(p_list, [1]*34)
    # plt.xlabel('density')
    # plt.ylabel('Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld')
    # plt.show()
    trajectory_len = []  # average trajectory length list
    avg_trajectory_div_shortestPath = []  # average (trajectory length / length of shortest path) list
    avg_cell_processed = []  # Average Number of Cells Processed list
    map = Map(101, 101)

    for p in p_list:
        num = 0
        avg_trajectory = 0  # average trajectory length
        avg_trj_div_stp = 0  # average (trajectory length / length of shortest path)
        cell_num = 0
        while num < 50:
            map.setObstacles(True, p)
            algo = AStar(map, 1)
            result = algo.run()
            print(result)
            if result:
                num += 1
                trajectory = len(algo.trajectory)
                avg_trajectory += trajectory
                shortestPath = algo.cost.get(map.end)
                avg_trj_div_stp += trajectory / shortestPath
                cell_num += len(algo.visited)
                # print(len(algo.visited))
            map.reset()
        trajectory_len.append(avg_trajectory / num)
        avg_trajectory_div_shortestPath.append(avg_trj_div_stp / num)
        avg_cell_processed.append(cell_num / num)

    plt.plot(p_list, trajectory_len)
    plt.xlabel('density')
    plt.ylabel('Average Trajectory Length')
    plt.show()
    plt.plot(p_list, avg_trajectory_div_shortestPath)
    plt.xlabel('density')
    plt.ylabel('Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld')
    plt.show()
    plt.plot(p_list, avg_cell_processed)
    plt.xlabel('density')
    plt.ylabel('Average Number of Cells Processed by Repeated A*')
    plt.show()
    print(p_list)
    print(trajectory_len)
    print(avg_trajectory_div_shortestPath)
    print(avg_cell_processed)


if __name__ == "__main__":
    question1()
    # question4()
    # question5()
    # question6()
