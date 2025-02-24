# Learning-based Multirotor Control Enhancements
The rise in popularity of quadrotors has lead to a need in new methods for predicting their interactions with aerodynamic forces. A simple physical model of a quadrotors behaviour can not perfectly model the real movement of a quadrotor. Therefore there is a need for models that can learn these inaccuracies.

In previous work researchers have been able to develop models for this purpose. In a previous thesis from the IMRC Lab, a bachelor student designed two types of models to solve this issue. They can be divided into two categories, these being neural networks and decision trees (or rather decision tree ensembles).

In this thesis we will take the two best performing models from the previous thesis and introduce them into a crazyflie's firmware to test how much performance gain there is both in simulation as well as real world flights.
