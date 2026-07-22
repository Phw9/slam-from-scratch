// Point-to-point joint trajectory generation.

#ifndef CVLIB_KINEMATICS_TRAJECTORY_H_
#define CVLIB_KINEMATICS_TRAJECTORY_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace kinematics {

/*
Samples a quintic point-to-point joint trajectory with zero boundary
velocity and acceleration: q(t) = q_start + (q_end - q_start) * s(u)
with s(u) = 10u^3 - 15u^4 + 6u^5 and u = t / duration. Samples are
uniform over [0, duration] including both endpoints.

@param q_start Start joint vector, length n.
@param q_end End joint vector, length n.
@param duration Total time (> 0).
@param num_samples Number of samples (>= 2).
@param positions Output num_samples-by-n joint positions.
@param velocities Optional output num_samples-by-n; may be null.
@param accelerations Optional output num_samples-by-n; may be null.
@returns ErrorCode.

*/

ErrorCode quintic_trajectory(const Vector* q_start, const Vector* q_end,
                             float64_t duration, int32_t num_samples,
                             Matrix* positions, Matrix* velocities,
                             Matrix* accelerations);

/*
Samples a synchronized trapezoidal-velocity joint trajectory.

The joint with the largest travel runs at the shared velocity and
acceleration limits (falling back to a triangular profile when the
travel is too short to reach max_velocity) and sets the duration;
every other joint scales to the same time law, so no joint exceeds
the limits. Zero total travel yields a constant trajectory with zero
duration. Samples are uniform over [0, duration] including both
endpoints.

@param q_start Start joint vector, length n.
@param q_end End joint vector, length n.
@param max_velocity Peak joint speed limit (> 0).
@param max_acceleration Joint acceleration limit (> 0).
@param num_samples Number of samples (>= 2).
@param positions Output num_samples-by-n joint positions.
@param velocities Optional output num_samples-by-n; may be null.
@param accelerations Optional output num_samples-by-n; may be null.
@param duration_out Output total time (>= 0).
@returns ErrorCode.

*/

ErrorCode trapezoidal_trajectory(const Vector* q_start,
                                 const Vector* q_end,
                                 float64_t max_velocity,
                                 float64_t max_acceleration,
                                 int32_t num_samples,
                                 Matrix* positions, Matrix* velocities,
                                 Matrix* accelerations,
                                 float64_t* duration_out);

}  // namespace kinematics
}  // namespace cvlib

#endif  // CVLIB_KINEMATICS_TRAJECTORY_H_
