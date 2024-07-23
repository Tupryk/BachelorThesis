import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../on_fly_performace_data/"))
import cfusdlog # type: ignore
import numpy as np


if __name__ == "__main__":
    data = cfusdlog.decode("../flight_data/jana30")['fixedFrequency']
    for i in range(1, 5):
        print(f"pwm{i} scaler", np.linalg.norm(data[f'pwm.m{i}_pwm']))
    print(f"mv scaler", np.linalg.norm(data['pm.vbatMV']))
