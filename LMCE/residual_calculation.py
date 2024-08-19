import rowan
import numpy as np


t2t = 0.006  # thrust-to-torque ratio
g2N = 0.00981
GRAVITY = 9.81
ARM_LENGTH = 0.046  # m


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


def residual(data_usd: dict, is_brushless: bool = False, has_payload: bool = False, use_rpm: bool = True, total_mass: float = .0347):

    # Get acceleration in world frame
    acc_body = np.array([
        data_usd['acc.x'],
        data_usd['acc.y'],
        data_usd['acc.z']]).T

    q = np.array([
        data_usd['stateEstimate.qw'],
        data_usd['stateEstimate.qx'],
        data_usd['stateEstimate.qy'],
        data_usd['stateEstimate.qz']]).T

    acc_world = rowan.rotate(q, acc_body)
    acc_world[:, 2] -= 1.
    acc_world *= 9.81

    # Get total forces
    if use_rpm:
        rpm = np.array([
            data_usd['rpm.m1'],
            data_usd['rpm.m2'],
            data_usd['rpm.m3'],
            data_usd['rpm.m4']]).T
        if is_brushless:
            force_in_grams = 4.310657321921365e-08 * rpm**2
        else:
            force_in_grams = 2.40375893e-08 * rpm**2 + - \
                3.74657423e-05 * rpm + -7.96100617e-02

    else:  # pwm
        pwm = np.array([
            data_usd['pwm.m1_pwm'],
            data_usd['pwm.m2_pwm'],
            data_usd['pwm.m3_pwm'],
            data_usd['pwm.m4_pwm']]).T
        if is_brushless:
            force_in_grams = -5.360718677769569 + pwm * 0.0005492858445116151
        else:
            force_in_grams = 1.65049399e-09 * pwm**2 + \
                9.44396129e-05 * pwm + -3.77748052e-01

    force = force_in_grams * g2N

    eta = np.empty((force.shape[0], 4))
    f_u = np.empty((force.shape[0], 3))
    tau_u = np.empty((force.shape[0], 3))  # Does not get used for now

    # Get residual forces
    arm = 0.707106781 * ARM_LENGTH
    B0 = np.array([
        [1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t]
    ])

    for k in range(force.shape[0]):
        eta[k] = np.dot(B0, force[k])
        f_u[k] = np.array([0, 0, eta[k, 0]])
        tau_u[k] = np.array([eta[k, 1], eta[k, 2], eta[k, 3]])

    f_a = total_mass * acc_world - rowan.rotate(q, f_u)

    if has_payload:
        payload_dir = np.array([data_usd["stateEstimateZ.px"] - data_usd["stateEstimate.x"],
                                data_usd["stateEstimateZ.py"] - data_usd["stateEstimate.y"],
                                data_usd["stateEstimateZ.pz"] - data_usd["stateEstimate.z"]], dtype=np.float32).T
        # t_q = project(f_a, payload_dir[1:])
        f_a = project_onto_plane(f_a, payload_dir)

    return f_a, tau_u
