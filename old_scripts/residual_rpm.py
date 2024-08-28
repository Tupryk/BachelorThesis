import cfusdlog
import numpy as np
import matplotlib.pyplot as plt


data = cfusdlog.decode("../flight_data/jana01")['fixedFrequency']
print(data.keys())

thrust = data[:,0] / 4  # g, per motor
rpm = np.mean(data[:,3:7],axis=1) # average over all motors

# rpmvsthrust1 = Polynomial.fit(m1, thrust, 2)
rpmvsthrust2 = np.polyfit(rpm, thrust, deg=2)
print(rpmvsthrust2)
# print(rpmvsthrust1.coef,'\n', rpmvsthrust2)
# exit()
# rpmvsthEval1 = np.polyval(rpmvsthrust1.coef, rpm)
rpmvsthEval2 = np.polyval(rpmvsthrust2, rpm)
