import pygame # Using pygame for keyborad input
from pygame.locals import *
import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper


URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    pygame.init()
    pygame.display.set_mode((100, 100))

    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        with MotionCommander(scf) as motion_commander:
            loop = True
            while loop:
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_w:
                            motion_commander.start_linear_motion(0, .1, 0)
                        elif event.key == K_n:
                            motion_commander.start_linear_motion(0, 0, 0)
                        elif event.key == K_s:
                            motion_commander.start_linear_motion(0, -1., 0)
                        elif event.key == K_k:
                            loop = False
                time.sleep(.1)
