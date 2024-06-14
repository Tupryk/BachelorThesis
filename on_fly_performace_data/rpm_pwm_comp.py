import cfusdlog
import numpy as np
from sklearn import preprocessing
from multirotor_config import MultirotorConfig
import matplotlib.pyplot as plt


g2N = MultirotorConfig.g2N

def thrust_torque(pwm_1, pwm_2, pwm_3, pwm_4, mv):
    g2N = MultirotorConfig.g2N
    f_1 = (11.09-39.08*pwm_1-9.53*mv + 20.57*pwm_1**2 + 38.43*pwm_1*mv)*g2N
    f_2 = (11.09-39.08*pwm_2-9.53*mv + 20.57*pwm_2**2 + 38.43*pwm_2*mv)*g2N
    f_3 = (11.09-39.08*pwm_3-9.53*mv + 20.57*pwm_3**2 + 38.43*pwm_3*mv)*g2N
    f_4 = (11.09-39.08*pwm_4-9.53*mv + 20.57*pwm_4**2 + 38.43*pwm_4*mv)*g2N
    l = MultirotorConfig.DISTANCE_ARM
    t2t = MultirotorConfig.t2t
    B0 = np.array([
        [1, 1, 1, 1],
        [0, -l, 0, l],
        [-l, 0, l, 0],
        [-t2t, t2t, -t2t, t2t]
    ])

    u = B0 @ np.array([f_1, f_2, f_3, f_4])
    return u[0]


def thrust_torque_rpm(rpm_1, rpm_2, rpm_3, rpm_4):
    rpm = np.mean(np.array([rpm_1, rpm_2, rpm_3, rpm_4]))
    numbs = [2.40375893e-08, -3.74657423e-05, -7.96100617e-02]
    u0 = np.polyval(numbs, rpm)
    return u0


data = cfusdlog.decode("../flight_data/jana00")['fixedFrequency']

pwm_1 = data['pwm.m1_pwm']
pwm_2 = data['pwm.m2_pwm']
pwm_3 = data['pwm.m3_pwm']
pwm_4 = data['pwm.m4_pwm']
mv = data['pm.vbatMV']

pwm_u = []
rpm_u = []
for i in range(1, len(data['timestamp'])):
    tmp = thrust_torque(pwm_1[i], pwm_2[i], pwm_3[i], pwm_4[i], mv[i])
    pwm_u.append(tmp)
    tmp = thrust_torque_rpm(data['rpm.m1'][i], data['rpm.m2'][i], data['rpm.m3'][i], data['rpm.m4'][i])
    rpm_u.append(tmp)

# plt.plot(data["timestamp"][1:], pwm_u, label="thrust with pwm")
# plt.plot(data["timestamp"][1:], rpm_u, label="thrust with rpm")

g2N = MultirotorConfig.g2N
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
pwm_1 = [np.polyval([1.65049399e-09, 9.44396129e-05, -3.77748052e-01], m) for m in pwm_1]
# pwm_1 = [(11.09-39.08*pm-9.53*mv[i] + 20.57*(pm**2) + 38.43*pm*mv[i])*g2N for i, pm in enumerate(pwm_1)]
rpm_1 = [np.polyval([2.40375893e-08, -3.74657423e-05, -7.96100617e-02], m) for m in data['rpm.m1']]
axs[0, 0].set_title('motor 1')
axs[0, 0].plot(data["timestamp"], pwm_1, label="pwm")
axs[0, 0].plot(data["timestamp"], rpm_1, label="rpm")
axs[0, 0].legend()

axs[0, 1].set_title('motor 2')
pwm_2 = [np.polyval([1.65049399e-09, 9.44396129e-05, -3.77748052e-01], m) for m in pwm_2]
rpm_2 = [np.polyval([2.40375893e-08, -3.74657423e-05, -7.96100617e-02], m) for m in data['rpm.m2']]
axs[0, 1].plot(data["timestamp"], pwm_2, label="pwm")
axs[0, 1].plot(data["timestamp"], rpm_2, label="rpm")
axs[0, 1].legend()

axs[1, 0].set_title('motor 3')
pwm_3 = [np.polyval([1.65049399e-09, 9.44396129e-05, -3.77748052e-01], m) for m in pwm_3]
rpm_3 = [np.polyval([2.40375893e-08, -3.74657423e-05, -7.96100617e-02], m) for m in data['rpm.m3']]
axs[1, 0].plot(data["timestamp"], pwm_3, label="pwm")
axs[1, 0].plot(data["timestamp"], rpm_3, label="rpm")
axs[1, 0].legend()

axs[1, 1].set_title('motor 4')
pwm_4 = [np.polyval([1.65049399e-09, 9.44396129e-05, -3.77748052e-01], m) for m in pwm_4]
rpm_4 = [np.polyval([2.40375893e-08, -3.74657423e-05, -7.96100617e-02], m) for m in data['rpm.m4']]
axs[1, 1].plot(data["timestamp"], pwm_4, label="pwm")
axs[1, 1].plot(data["timestamp"], rpm_4, label="rpm")
axs[1, 1].legend()

plt.legend()
plt.tight_layout()
plt.show()
