# encoding=utf-8

from A_star_algo import *
import time


def question8():
    p_list = np.linspace(0, 0.33, 8)
    traj_length, improved_traj_list = [],[]
    for p in p_list:
        map = Map(101, 101)
        map.setObstacles(True, p)
        for i in range(2):
            algo = RepeatedAStar(map, 1)
            startTime = time.time()
            result = algo.run(False, i)
            endTime = time.time()
            print(result)
            # if result == False:break
            # img = Image.fromarray(np.uint8(cm.gist_earth(map.map) * 255))
            # img = np.array(img.convert('RGB'))
            traj = algo.trajectory
            # for trace in traj:
            #     img[trace[0]][trace[1]] = [0, 0, 255]
            # start = map.start
            # end = map.end
            # print(img.shape)
            # img[start[0]][start[1]] = [255, 0, 0]
            # img[end[0]][end[1]] = [255, 0, 0]
            # improve = {
            #     0: "Standard Repeated A *:",
            #     1: "Improved Repeated A *:"
            # }
            # plt.title(improve.get(i)+"\nTime consumption: "+str(endTime - startTime)+"\nTrajectory Length: "+str(len(traj)))
            # plt.imshow(img)
            # plt.grid(linewidth=1)
            # plt.show()
            if i:
                improved_traj_list.append(len(traj))
            else:
                traj_length.append(len(traj))
    plt.plot(p_list, traj_length, color='red')
    plt.plot(p_list, improved_traj_list, color='blue')
    plt.legend(['Standard Repeated A *', 'Improved Repeated A *'])
    plt.xlabel("random probability p")
    plt.ylabel("trajectory length")
    plt.title("dim = 101x101 distanceType")
    plt.show()


if __name__ == "__main__":
    question8()
