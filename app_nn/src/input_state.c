#include <stdlib.h>

#include "input_state.h"

#include "log.h"

#define GRAVITATION 9.81f
#define DEG2RAD .017453f

void load_input_vector(float* input_vec)
{
    //// Acceleration ////
    logVarId_t idAccX = logGetVarId("acc", "x");
    logVarId_t idAccY = logGetVarId("acc", "y");
    logVarId_t idAccZ = logGetVarId("acc", "z");
    input_vec[0] = logGetFloat(idAccX)*GRAVITATION;
    input_vec[1] = logGetFloat(idAccY)*GRAVITATION;
    input_vec[2] = logGetFloat(idAccZ)*GRAVITATION;

    //// Gyroscope ////
    logVarId_t idGyroX = logGetVarId("gyro", "x");
    logVarId_t idGyroY = logGetVarId("gyro", "y");
    logVarId_t idGyroZ = logGetVarId("gyro", "z");
    input_vec[3] = logGetFloat(idGyroX)*DEG2RAD;
    input_vec[4] = logGetFloat(idGyroY)*DEG2RAD;
    input_vec[5] = logGetFloat(idGyroZ)*DEG2RAD;

    //// Quaternion ////
    logVarId_t idSEqw = logGetVarId("stateEstimate", "qw");
    logVarId_t idSEqx = logGetVarId("stateEstimate", "qx");
    logVarId_t idSEqy = logGetVarId("stateEstimate", "qy");
    logVarId_t idSEqz = logGetVarId("stateEstimate", "qz");
    double q0 = logGetFloat(idSEqw);
    double qx = logGetFloat(idSEqx);
    double qy = logGetFloat(idSEqy);
    double qz = logGetFloat(idSEqz);
    double q02 = q0*q0;
    double qx2 = qx*qx;
    double qz2 = qy*qy;
    double qy2 = qz*qz;
    // Calculate the first two rows of the rotation matrix from the quaternion
    input_vec[6] = q02 + qx2 - qy2 - qz2;
    input_vec[7] = 2*qx*qy - 2*q0*qz;
    input_vec[8] = 2*qx*qy + 2*q0*qz;
    input_vec[9] = q02 - qx2 + qy - qz2;
    input_vec[10] = 2*qx*qz - 2*q0*qy;
    input_vec[11] = 2*qy*qz + 2*q0*qx;
}
