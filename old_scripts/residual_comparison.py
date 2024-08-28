import cfusdlog
import numpy as np
import matplotlib.pyplot as plt
from residual_calculation import residual
from single_uav_residual_calculations.residual_calculation import residual2


data_path = "../flight_data/jana02"

data = cfusdlog.decode(data_path)['fixedFrequency']
f, _ = residual(data, use_rpm=False, rot=True)
f1, _ = residual(data, use_rpm=True, rot=True)
f2, _ = residual2(data_path)

plt.plot(f[:, 0], label="system_id polynomial (pwm)", alpha=.5)
plt.plot(f1[:, 0], label="system_id polynomial (rpm)", alpha=.5)
plt.plot(f2[:, 0], label="jana polynomial (pwm)", alpha=.2)
plt.title("Residual in x")
plt.legend()
plt.show()

plt.plot(f[:, 1])
plt.plot(f2[:, 1])
plt.show()

plt.plot(f[:, 2])
plt.plot(f2[:, 2])
plt.show()

