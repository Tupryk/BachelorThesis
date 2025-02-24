#!/usr/bin/python3
import time
import numpy as np
from crazyflie_py import *

from rand_path_gen import gen_rand_flight_path


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # TODO: Check battery voltage

    controller_types = ["lee", "lee_indi", "lee_nn", "lee_nn_indi"]
    for tp in controller_types:
        
        allcfs.setParam('usd.logging', 1)

        if tp == "lee_indi":
            allcfs.setParam("indi", 1)
            allcfs.setParam("nn", 0)

        elif tp == "lee_nn":
            allcfs.setParam("indi", 0)
            allcfs.setParam("nn", 1)

        elif tp == "lee_nn_indi":
            allcfs.setParam("indi", 1)
            allcfs.setParam("nn", 1)

        else:
            allcfs.setParam("indi", 0)
            allcfs.setParam("nn", 0)
    
        allcfs.takeoff(targetHeight=1.0, duration=2.0)
        timeHelper.sleep(2.5)

        # start recording to sdcard
        allcfs.setParam("usd.logging", 1)

        cf = allcfs.crazyflies[0]
        speed = 1.0

        bbox = [[-1.0, -1.0,  0.1],
                [ 1.0,  1.0,  1.0]]
        traj = gen_rand_flight_path(bbox)

        last_pos = traj[0]
        cf.goTo(last_pos, 0, 4)
        timeHelper.sleep(4)
    
        for pos in traj[1:]:
            dist = np.linalg.norm(pos - last_pos)
            time_to_move = dist / speed
            cf.goTo(pos, 0, time_to_move)
            timeHelper.sleep(time_to_move)
            last_pos = pos

        # stop recording to sdcard
        allcfs.setParam("usd.logging", 0)

        allcfs.land(targetHeight=0.02, duration=3.0)
        timeHelper.sleep(3.0)


if __name__ == "__main__":
    main()