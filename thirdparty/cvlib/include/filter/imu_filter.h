// 15-DOF IMU error-state Kalman filter helpers (strap-down).

#ifndef CVLIB_FILTER_IMU_FILTER_H_
#define CVLIB_FILTER_IMU_FILTER_H_

#include "filter/iekf.h"
#include "filter/states.h"
#include "types.h"
#include "error_codes.h"

namespace cvlib {
namespace filter {

// Single IMU sample: body-frame accel + gyro + integration step.
struct ImuMeasurement {
    Vector    accel;
    Vector    gyro;
    float64_t dt;
};

/*
Strap-down predict step. Integrates the nominal state with the supplied
accel + gyro and propagates the 15-DOF error-state covariance.

@param state In-out ImuState.
@param meas Body-frame measurement bundle.
@param gravity World-frame gravity (length 3, e.g. [0, 0, -9.81]).
@param Q Either a 15-by-15 discrete covariance on the tangent, or a 12-by-12
         IMU input-noise covariance ordered [ng, na, nbg, nba].
@returns ErrorCode.
*/
ErrorCode imu_predict(ImuState* state, ImuMeasurement meas,
                      const Vector* gravity, const Matrix* Q);

/*
Position update from a world-frame measurement (e.g. GPS).

@param state In-out ImuState.
@param p_meas Measured position (length 3).
@param R 3-by-3 measurement noise covariance.
@returns ErrorCode.
*/
ErrorCode imu_update_position(ImuState* state, const Vector* p_meas,
                              const Matrix* R);

/*
Pose update from a world-frame measurement.

@param state In-out ImuState.
@param R_meas Measured rotation (3-by-3).
@param p_meas Measured position (length 3).
@param R 6-by-6 measurement noise covariance ([dtheta, dp]).
@returns ErrorCode.
*/
ErrorCode imu_update_pose(ImuState* state,
                          const Matrix* R_meas, const Vector* p_meas,
                          const Matrix* R);

/*
Iterated (IEKF) pose update, mirroring pose_iekf_update_pose: the residual
and Jacobian are re-linearized at the corrected nominal until the tangent
delta drops below opts.tol or opts.max_iter is reached. With
opts.max_iter = 1 this matches imu_update_pose exactly; the iteration only
pays off for strongly nonlinear measurement models.

@param state In-out ImuState.
@param R_meas Measured rotation (3-by-3).
@param p_meas Measured position (length 3).
@param R 6-by-6 measurement noise covariance ([dtheta, dp]).
@param opts Iteration cap and convergence tolerance.
@returns ErrorCode.
*/
ErrorCode imu_iekf_update_pose(ImuState* state,
                               const Matrix* R_meas, const Vector* p_meas,
                               const Matrix* R, IekfOptions opts);

/*
Gravity-augmented strap-down predict step. Uses state.g as the world-frame
gravity estimate and propagates the 18-DOF error-state covariance
([dp dv dtheta dba dbg dg]); gravity itself is a random constant
(no process noise rows in the 12-DOF input mapping).

@param state In-out ImuGravityState.
@param meas Body-frame measurement bundle.
@param Q Either an 18-by-18 discrete covariance on the tangent, or a
         12-by-12 IMU input-noise covariance ordered [ng, na, nbg, nba].
@returns ErrorCode.
*/
ErrorCode imu_gravity_predict(ImuGravityState* state, ImuMeasurement meas,
                              const Matrix* Q);

/*
Position update for the gravity-augmented state.

@param state In-out ImuGravityState.
@param p_meas Measured position (length 3).
@param R 3-by-3 measurement noise covariance.
@returns ErrorCode.
*/
ErrorCode imu_gravity_update_position(ImuGravityState* state,
                                      const Vector* p_meas, const Matrix* R);

/*
Pose update for the gravity-augmented state.

@param state In-out ImuGravityState.
@param R_meas Measured rotation (3-by-3).
@param p_meas Measured position (length 3).
@param R 6-by-6 measurement noise covariance ([dtheta, dp]).
@returns ErrorCode.
*/
ErrorCode imu_gravity_update_pose(ImuGravityState* state,
                                  const Matrix* R_meas, const Vector* p_meas,
                                  const Matrix* R);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_IMU_FILTER_H_
