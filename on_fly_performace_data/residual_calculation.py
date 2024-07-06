import rowan
import numpy as np
from sklearn import preprocessing
from multirotor_config import MultirotorConfig

RPM2RADSEG = .10472
ms2s = MultirotorConfig.ms2s
g = MultirotorConfig.GRAVITATION
d2r = MultirotorConfig.deg2rad
I = MultirotorConfig.INERTIA
d = MultirotorConfig.DISTANCE_ARM
# m = MultirotorConfig.MASS
m = 0.037
ms2g = MultirotorConfig.ms2g
g2N = MultirotorConfig.g2N


def brushless_thrust_torque(pwm):
    d00 = 0.16609668469985447
    d10 = 0.02297914810965436
    d20 = 0.0001878280261464696

    a = d20
    b = d10
    c = d00 - pwm
    thrustGram0 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    thrustGram1 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    return thrustGram0 * g2N

def thrust_torque(pwm_1, pwm_2, pwm_3, pwm_4, mv):
    f_1 = (11.09-39.08*pwm_1-9.53*mv + 20.57*pwm_1**2 + 38.43*pwm_1*mv)*g2N
    f_2 = (11.09-39.08*pwm_2-9.53*mv + 20.57*pwm_2**2 + 38.43*pwm_2*mv)*g2N
    f_3 = (11.09-39.08*pwm_3-9.53*mv + 20.57*pwm_3**2 + 38.43*pwm_3*mv)*g2N
    f_4 = (11.09-39.08*pwm_4-9.53*mv + 20.57*pwm_4**2 + 38.43*pwm_4*mv)*g2N

    # f_1 = brushless_thrust_torque(pwm_1)
    # f_2 = brushless_thrust_torque(pwm_2)
    # f_3 = brushless_thrust_torque(pwm_3)
    # f_4 = brushless_thrust_torque(pwm_4)

    # poly_vals = [1.65049399e-09, 9.44396129e-05, -3.77748052e-01]
    # f_1 = np.polyval(poly_vals, pwm_1) * g2N
    # f_2 = np.polyval(poly_vals, pwm_2) * g2N
    # f_3 = np.polyval(poly_vals, pwm_3) * g2N
    # f_4 = np.polyval(poly_vals, pwm_4) * g2N

    arm_length = 0.046  # m
    arm = 0.707106781 * arm_length
    t2t = 0.006  # thrust-to-torque ratio
    B0 = np.array([
        [1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t]
    ])

    u = B0 @ np.array([f_1, f_2, f_3, f_4])
    return u


def thrust_torque_rpm(rpm_1, rpm_2, rpm_3, rpm_4):
    
    nums = [2.40375893e-08, -3.74657423e-05, -7.96100617e-02]
    f_1 = np.polyval(nums, rpm_1)
    f_2 = np.polyval(nums, rpm_2)
    f_3 = np.polyval(nums, rpm_3)
    f_4 = np.polyval(nums, rpm_4)

    arm_length = 0.046  # m
    arm = 0.707106781 * arm_length
    t2t = 0.006  # thrust-to-torque ratio
    B0 = np.array([
        [1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t]
    ])

    f = np.array([f_1, f_2, f_3, f_4]) * g2N
    u = B0 @ f
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


def residual(data, use_rpm=False, rot=True):

    start_time = data['timestamp'][0]
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
        prev_a_vel = np.array(
            [data['gyro.x'][i-1], data['gyro.y'][i-1], data['gyro.z'][i-1]])*d2r

        quat = np.array([data['stateEstimate.qw'][i], data['stateEstimate.qx'][i],
                         data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]])
        R = rowan.to_matrix(quat)
        
        if rot:
            acc = R @ np.array([data['acc.x'][i], data['acc.y'][i], data['acc.z'][i]])
        else:
            acc = np.array([data['acc.x'][i], data['acc.y'][i], data['acc.z'][i]])
            # acc = np.array([data['stateEstimate.ax'][i], data['stateEstimate.ay'][i], data['stateEstimate.az'][i]])

        acc[2] -= 1.
        acc *= g


        a_acc = angular_acceleration(a_vel, prev_a_vel, prev_time, time)
        
        if use_rpm:
            u = thrust_torque_rpm(*[data[f'rpm.m{j}'][i] for j in range(1, 5)])
        else:
            # u = thrust_torque(*[data[f'pwm.m{j}_pwm'][i] for j in range(1, 5)], data['pm.vbatMV'][i])
            u = thrust_torque(pwm_1[i], pwm_2[i], pwm_3[i], pwm_4[i], mv[i])

        f_u = np.array([0, 0, u[0]])
        f_a = disturbance_forces(m, acc, R, f_u)
        f.append(f_a)
        tau_u = np.array([u[1], u[2], u[3]])
        tau_a = disturbance_torques(a_acc, a_vel, tau_u)
        tau.append(tau_a)

    f = np.array(f)
    tau = np.array(tau)

    return f, tau

def payload_residual(data):
    f = []

    for i in range(1, len(data['timestamp'])):
        quat = np.array([data['stateEstimate.qw'][i], data['stateEstimate.qx'][i],
                         data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]])
        R = rowan.to_matrix(quat)
        
        acc = R @ np.array([data['acc.x'][i], data['acc.y'][i], data['acc.z'][i]])
        acc[2] -= 1.
        acc *= g
        
        u = data['ctrlLeeP.thrustSI'][i]

        f_u = np.array([0, 0, u])
        f_a = disturbance_forces(m, acc, R, f_u)
        f.append(f_a)

    f = np.array(f)

    return f

def residual_v2(data):
    K_af = np.array([
        [-10.2506, -0.3177,  -0.4332],
        [-0.3177,  -10.2506, -0.4332],
        [-7.7050,  -7.7050,  -7.5530]
    ]) * 10**-7

    f = []
    for i in range(1, len(data['timestamp'])):
        quat = np.array([data['stateEstimate.qw'][i], data['stateEstimate.qx'][i],
                         data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]])
        R = rowan.to_matrix(quat) # the rotation of the body frame with respect to the inertial frame (???)

        vel = np.array([
            data['stateEstimate.vx'][i],
            data['stateEstimate.vy'][i],
            data['stateEstimate.vz'][i]])
        
        theta = 0
        for n in range(1, 5):
            theta += np.abs(data[f'rpm.m{n}'][i]*RPM2RADSEG)
        
        f_ = R @ (K_af * theta @ R.T @ vel)
        f.append(f_)

    f = np.array(f)
    return f


def project(this, on_that):
    projections = []
    for j in range(len(on_that)):
        on_that[j] /= np.linalg.norm(on_that[j])
        dot_product = np.dot(this[j], on_that[j])

        projected = (dot_product / np.dot(on_that[j], on_that[j])) * on_that[j]

        projections.append(projected)

    projections = np.array(projections)
    return projections

def project_onto_plane(v, n):
    projections = []
    for j in range(len(v)):
        n[j] = n[j] / np.linalg.norm(n[j])
        dot_product = np.dot(v[j], n[j])
        projection_on_normal = dot_product * n[j]
        projection_on_plane = v[j] - projection_on_normal
    
        projections.append(projection_on_plane)

    projections = np.array(projections)
    return projections

def brushless_residual(data_usd, use_rpm=True):
    q = np.array([
        data_usd['stateEstimate.qw'],
        data_usd['stateEstimate.qx'],
        data_usd['stateEstimate.qy'],
        data_usd['stateEstimate.qz']]).T

    rpm = np.array([
        data_usd['rpm.m1'],
        data_usd['rpm.m2'],
        data_usd['rpm.m3'],
        data_usd['rpm.m4']]).T
    
    pwm = np.array([
        data_usd['pwm.m1_pwm'],
        data_usd['pwm.m2_pwm'],
        data_usd['pwm.m3_pwm'],
        data_usd['pwm.m4_pwm']]).T
    
    kw = 4.310657321921365e-08
    force_in_grams = kw * rpm**2
    force_in_grams_from_pwm = -5.360718677769569 + pwm * 0.0005492858445116151

    acc_body = np.array([
        data_usd['acc.x'],
        data_usd['acc.y'],
        data_usd['acc.z']]).T * 9.81
    
    acc_world = rowan.rotate(q, acc_body)

    mass = 0.038 # kg

    arm_length = 0.046  # m
    arm = 0.707106781 * arm_length
    t2t = 0.006  # thrust-to-torque ratio
    B0 = np.array([
        [1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t]
        ])
    if use_rpm:
        force = force_in_grams / 1000 * 9.81
    else:
        force = force_in_grams_from_pwm / 1000 * 9.81
    eta = np.empty((force.shape[0], 4))
    f_u = np.empty((force.shape[0], 3))
    tau_u = np.empty((force.shape[0], 3))

    for k in range(force.shape[0]):
        eta[k] = np.dot(B0, force[k])
        f_u[k] = np.array([0, 0, eta[k,0]])
        tau_u[k] = np.array([eta[k,1], eta[k,2], eta[k,3]])

    f_a = mass * acc_world - rowan.rotate(q, f_u)
    return f_a, tau_u
