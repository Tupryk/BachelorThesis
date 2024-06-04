import argparse
import numpy as np
import matplotlib.pyplot as plt


def loadFile(filename):
    fileData = np.loadtxt(filename, delimiter=',', skiprows=1, ndmin=2)
    return fileData

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='+')
    args = parser.parse_args()

    data = None
    for f in args.filename:
        data_f = loadFile(f)
        if data is None:
            data = data_f
        else:
            data = np.vstack((data, data_f))

    thrust = data[:,0] / 4  # g, per motor
    pwm = data[:,1]         # PWM value
    vbat = data[:,2]        # V, battery voltage, 
    rpm = np.mean(data[:,3:7],axis=1) # average over all motors
    m1  = data[:,3]
    m2  = data[:,4]
    m3  = data[:,5]
    m4  = data[:,6]
    vSid =  data[:, 7]      # Volts at system id deck
    amp =  data[:, 8]       # Amps
    pwr = data[:, 9]        # Power in watts

    fig, ax = plt.subplots(2)
    ax[0].plot(m1, label='M1')
    ax[0].plot(m2, label='M2')
    ax[0].plot(m3, label='M3')
    ax[0].plot(m4, label='M4')
    ax[0].plot(rpm, label='Mean')

    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('RPM')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(rpm, thrust, label='data')
    ax[1].set_xlabel('rpm')
    ax[1].set_ylabel('thrust [g]')
    # ax[1].legend()
    # ax[1].grid(True)

    # rpmvsthrust1 = Polynomial.fit(m1, thrust, 2)
    rpmvsthrust2 = np.polyfit(rpm, thrust, deg=2)
    print(rpmvsthrust2)
    # print(rpmvsthrust1.coef,'\n', rpmvsthrust2)
    # exit()
    # rpmvsthEval1 = np.polyval(rpmvsthrust1.coef, rpm)
    rpmvsthEval2 = np.polyval(rpmvsthrust2, rpm)

    ax[1].plot(rpm, rpmvsthEval2, label='fit')
    # ax[2].set_xlabel('rpm')
    # ax[2].set_ylabel('fitted thrust [g]')
    ax[1].legend()
    ax[1].grid(True)

    plt.show()
