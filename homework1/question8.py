# encoding=utf-8

from A_star_algo import *
import time


def question8():
    test_num = 20
    p_list = np.linspace(0, 0.33, 20)
    traj_length, improved_traj_list = [], []
    normal_time_list, improved_time_list = [], []
    for p in p_list:
        improved_traj, normal_traj, normal_time, improved_time = 0, 0, 0, 0
        for _ in range(test_num):
            map = Map(101, 101)
            map.setObstacles(True, p)
            As = AStar(map, 1)
            while True:
                if not As.run():
                    map.reset()
                    map.setObstacles(True, p)
                    As = AStar(map, 1)
                else:
                    break
            normal_As = RepeatedAStar(copy.deepcopy(map), 1)
            algo = RepeatedAStar(copy.deepcopy(map), 1)
            time1 = time.time()
            normal_As.run()
            time2 = time.time()
            algo.run(bumpInto=False, improvement=True)
            time3 = time.time()
            improved_time += time3-time2
            normal_time += time2-time1
            normal_traj += len(normal_As.trajectory)
            improved_traj += len(algo.trajectory)

        improved_traj_list.append(improved_traj / test_num)
        traj_length.append(normal_traj / test_num)
        normal_time_list.append(normal_time / test_num)
        improved_time_list.append(improved_time / test_num)
    plt.plot(p_list, traj_length, color='red')
    plt.plot(p_list, improved_traj_list, color='blue')
    plt.legend(['Standard Repeated A *', 'Improved Repeated A *'])
    plt.xlabel("random probability p")
    plt.ylabel("trajectory length")
    plt.title("dim = 101x101 Manhattan")
    plt.show()
    plt.plot(p_list, normal_time_list, color='red')
    plt.plot(p_list, improved_time_list, color='blue')
    plt.legend(['Standard Repeated A *', 'Improved Repeated A *'])
    plt.xlabel("random probability p")
    plt.ylabel("running time")
    plt.title("dim = 101x101 Manhattan")
    plt.show()


if __name__ == "__main__":
    question8()
