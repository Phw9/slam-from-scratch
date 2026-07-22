// Chain dynamics helpers under a point-mass link model.

#ifndef CVLIB_KINEMATICS_DYNAMICS_H_
#define CVLIB_KINEMATICS_DYNAMICS_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace kinematics {

/*
Computes the joint torques that hold the chain statically under
gravity.

Each link is modeled as a point mass at its DH frame origin, so
tau = -sum_i m_i * J_v,i(q)^T * g with J_v,i the positional Jacobian
of link i's origin. Rotational link inertia and off-origin centers of
mass are not modeled; fold a center-of-mass offset into the DH table
when it matters.

@param dh_params Rows-by-4 table [a, alpha, d, theta_offset] per joint.
@param joint_angles Joint vector, length equals the number of DH rows.
@param link_masses Point mass per link (>= 0), same length.
@param gravity Gravity acceleration in the base frame, length 3
       (e.g. (0, 0, -9.81)).
@param torques Output joint torques, same length.
@returns ErrorCode.

*/

ErrorCode gravity_torques(const Matrix* dh_params,
                          const Vector* joint_angles,
                          const Vector* link_masses,
                          const Vector* gravity, Vector* torques);

/*
Computes the joint-space inertia matrix under the same point-mass
model: M(q) = sum_i m_i * J_v,i(q)^T * J_v,i(q). The result is
symmetric positive semidefinite; rotational link inertia is not
modeled.

@param dh_params Rows-by-4 table [a, alpha, d, theta_offset] per joint.
@param joint_angles Joint vector, length equals the number of DH rows.
@param link_masses Point mass per link (>= 0), same length.
@param m_out Output n-by-n inertia matrix; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode mass_matrix(const Matrix* dh_params,
                      const Vector* joint_angles,
                      const Vector* link_masses, Matrix* m_out);

}  // namespace kinematics
}  // namespace cvlib

#endif  // CVLIB_KINEMATICS_DYNAMICS_H_
