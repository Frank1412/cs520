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
    p_list = np.linspace(0, 0.33, 34)
    trajectory_len = []
    for p in p_list:
        map = Map(101, 101)
        map.setObstacles(True, p)
        algo = AStar(map, 1)
        sum = 0
        for i in range(1):
            result = algo.run()
            print(result)
            if result:
                sum += 1
        path = algo.path
        trajectory_len.append(len(path))
        map.reset()

    plt.plot(p_list, trajectory_len)
    plt.xlabel('density')
    plt.ylabel('Average Trajectory Length')
    plt.show()
    print(p_list)


if __name__ == "__main__":

    # question4()
    question6()