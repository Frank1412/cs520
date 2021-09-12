# encoding=utf-8

from A_star_algo import *


def question4():
    map = Map(101, 101)
    # map.setStartPoint(0, 0)
    # map.setObstacles(False, 0.1, obstacles)
    map.setObstacles(True, 0.01)
    algo = AStar(map, 1)
    sum = 0
    for i in range(100):
        map = Map(101, 101)
        map.setObstacles(True, 0.20)
        algo = AStar(map, 1)
        result = algo.run()
        print(result)
        if result:
            sum += 1
    print(sum / 100)
    path = algo.path
    last = map.end
    while last in path:
        # print(last, path[last])
        last = path[last]


def question6():
    p_list = np.linspace(0, 0.33, 5)  # 34
    trajectory_len = []  # average trajectory length list
    avg_trajectory_div_shortestPath = []   # average (trajectory length / length of shortest path) list
    avg_cell_processed = []    # Average Number of Cells Processed list
    map = Map(101, 101)

    for p in p_list:
        num = 0
        path_len = 0    # average trajectory length
        avg_trj_div_stp = 0    # average (trajectory length / length of shortest path)
        cell_num = 0
        while num < 1:
            map.setObstacles(True, p)
            algo = AStar(map, 1)
            result = algo.run()
            print(result)
            if result:
                num += 1
                path_len += len(algo.path)
                shortestPath = algo.cost.get(map.end)
                avg_trj_div_stp += path_len/shortestPath

            map.reset()
        trajectory_len.append(path_len / num)
        avg_trajectory_div_shortestPath.append(avg_trj_div_stp/num)

    plt.plot(p_list, trajectory_len)
    plt.xlabel('density')
    plt.ylabel('Average Trajectory Length')
    plt.show()
    plt.plot(p_list, avg_trajectory_div_shortestPath)
    plt.xlabel('density')
    plt.ylabel('Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld')
    plt.show()
    print(p_list)


if __name__ == "__main__":
    # question4()
    question6()
