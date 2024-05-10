import numpy as np

class MultirotorConfig:
    MASS = 0.0347 #kg
    INERTIA = np.array([[16.571710, 0.830306, 0.718277], [0.830806, 16.655602, 1.800197], [0.718277, 1.800197, 29.261652]])*10**(-6) #kg*m²
    DISTANCE_ARM  = 0.046 #meters
    ARM = 0.707106781 * DISTANCE_ARM # Arm

    ms2s = 0.001 # milliseconds to seconds
    GRAVITATION = 9.81 #gravitation and g-unit to m/s² 
    rad2deg = 57.29578 # rad to degree
    deg2rad = 0.017453 # degree to rad
    ms2g = 0.101972 #m/s² to g-unit
    g2N = 0.00981 #grams to Newton
    t2t = 0.006  # thrust-to-torque ratio