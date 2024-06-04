import numpy as np
from multirotor_config import MultirotorConfig
import rowan
from sklearn import preprocessing

ms2s = MultirotorConfig.ms2s
g = MultirotorConfig.GRAVITATION
d2r = MultirotorConfig.deg2rad
I = MultirotorConfig.INERTIA
d = MultirotorConfig.DISTANCE_ARM
m = MultirotorConfig.MASS
ms2g = MultirotorConfig.ms2g
g2N = MultirotorConfig.g2N


def thrust_torque(pwm_1, pwm_2, pwm_3, pwm_4, mv):
    f_1 = (11.09-39.08*pwm_1-9.53*mv + 20.57*pwm_1**2 + 38.43*pwm_1*mv)*g2N
    f_2 = (11.09-39.08*pwm_2-9.53*mv + 20.57*pwm_2**2 + 38.43*pwm_2*mv)*g2N
    f_3 = (11.09-39.08*pwm_3-9.53*mv + 20.57*pwm_3**2 + 38.43*pwm_3*mv)*g2N
    f_4 = (11.09-39.08*pwm_4-9.53*mv + 20.57*pwm_4**2 + 38.43*pwm_4*mv)*g2N
    l = MultirotorConfig.DISTANCE_ARM
    arm = MultirotorConfig.ARM
    t2t = MultirotorConfig.t2t
    B0 = np.array([
        [1, 1, 1, 1],
        [0, l, 0, -l],
        [-l, 0, l, 0],
        [t2t, -t2t, t2t, -t2t]
    ])

    u = B0 @ np.array([f_1, f_2, f_3, f_4])
    return u


def thrust_torque_rpm(rpm_1, rpm_2, rpm_3, rpm_4):
    rpm2radseg = 0.10472
    w = np.array([rpm_1, rpm_2, rpm_3, rpm_4]) * rpm2radseg
    print(w)
    w *= w
    print(w)
    l = MultirotorConfig.DISTANCE_ARM
    t2t = MultirotorConfig.t2t
    B0 = np.array([
        [1, 1, 1, 1],
        [0, l, 0, -l],
        [-l, 0, l, 0],
        [t2t, -t2t, t2t, -t2t]
    ])

    u = B0 @ w
    return u


def angular_acceleration(a_vel, prev_a_vel, prev_time, time):
    t = (time - prev_time) * ms2s
    a_acc = (a_vel-prev_a_vel)/t
    return a_acc


def disturbance_forces(m, acc, R, f_u):
    g_m = np.array([0, 0, -g])
    f_a = m*acc - m*g_m - R@f_u
    return f_a


def disturbance_torques(a_acc, a_vel, tau_u):
    tau_a = I@a_acc - np.cross(I@a_vel, a_vel) - tau_u
    return tau_a


def residual(data):

    start_time = data['timestamp'][0]
    m = MultirotorConfig.MASS
    f = []
    tau = []
    prev_time = start_time

    pwm_1 = preprocessing.normalize(data['pwm.m1_pwm'][None])[0]
    pwm_2 = preprocessing.normalize(data['pwm.m2_pwm'][None])[0]
    pwm_3 = preprocessing.normalize(data['pwm.m3_pwm'][None])[0]
    pwm_4 = preprocessing.normalize(data['pwm.m4_pwm'][None])[0]
    mv = preprocessing.normalize(data['pm.vbatMV'][None])[0]

    for i in range(1, len(data['timestamp'])):
        time = data['timestamp'][i]
        a_vel = np.array([data['gyro.x'][i], data['gyro.y'][i],
                          data['gyro.z'][i]])*d2r
        prev_a_vel = a_vel = np.array(
            [data['gyro.x'][i-1], data['gyro.y'][i-1], data['gyro.z'][i-1]])*d2r
        acc = np.array([data['acc.x'][i], data['acc.y'][i],
                        data['acc.z'][i]-1])*g

        quat = np.array([data['stateEstimate.qw'][i], data['stateEstimate.qx'][i],
                         data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]])
        R = rowan.to_matrix(quat)
        u = thrust_torque(pwm_1[i], pwm_2[i], pwm_3[i], pwm_4[i], mv[i])
        a_acc = angular_acceleration(a_vel, prev_a_vel, prev_time, time)
        f_u = np.array([0, 0, u[0]])
        f_a = disturbance_forces(m, acc, R, f_u)
        f.append(f_a)
        tau_u = np.array([u[1], u[2], u[3]])
        tau_a = disturbance_torques(a_acc, a_vel, tau_u)
        tau.append(tau_a)

    f = np.array(f)
    tau = np.array(tau)

    return f, tau
