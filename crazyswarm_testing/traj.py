#!/usr/bin/env python

import numpy as np
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory

HEIGHT = 0.5
TIMESCALE = 3.0
CONTROLLER = 1
FILE = "circle0.csv"

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    for cf in allcfs.crazyflies:
        cf.setParam("stabilizer.controller", CONTROLLER)

    allcfs.setParam("motorPowerSet.enable", 0) # make sure mocap can see us

    traj1 = Trajectory()
    traj1.loadcsv(Path(get_package_share_directory('jana_ros')) / "data" / FILE)
    ev = traj1.eval(0)

    TRIALS = 1
    for i in range(TRIALS):
        for cf in allcfs.crazyflies:
            cf.uploadTrajectory(0, 0, traj1)

        allcfs.takeoff(targetHeight=HEIGHT, duration=3.0)
        timeHelper.sleep(5)
        for cf in allcfs.crazyflies:
            pos = np.array(cf.initialPosition) + np.array([0, 0, HEIGHT])
            cf.goTo(pos, ev.yaw, 2.0)
        timeHelper.sleep(3.0)

        for cf in allcfs.crazyflies:
            cf.setParam("usd.logging", 1)

        allcfs.startTrajectory(0, timescale=TIMESCALE)
        timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)
        # allcfs.startTrajectory(0, timescale=TIMESCALE, reverse=True)
        # timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)

        for cf in allcfs.crazyflies:
            cf.setParam("usd.logging", 0)

        for cf in allcfs.crazyflies:
            pos = np.array(cf.initialPosition) + np.array([0, 0, HEIGHT])
            cf.goTo(pos, 0, 2.0)
        timeHelper.sleep(3.0)

        allcfs.land(targetHeight=0.06, duration=2.0)
        timeHelper.sleep(3.0)


if __name__ == "__main__":
    main()
