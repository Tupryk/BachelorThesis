import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import LMCE.cfusdlog as cfusdlog
import matplotlib.pyplot as plt
from LMCE.residual_calculation import residual


data_path = "../crazyflie-data-collection/jana_flight_data/jana02"

data = cfusdlog.decode(data_path)['fixedFrequency']
f, _ = residual(data, use_rpm=False)
f1, _ = residual(data, use_rpm=True)

for i, v in enumerate(["x", "y", "z"]):
    plt.plot(f[:, i], label="PWM", alpha=1.)
    plt.plot(f1[:, i], label="RPM", alpha=1.)
    plt.title(f"Residuals in {v}")
    plt.legend()
    plt.show()
