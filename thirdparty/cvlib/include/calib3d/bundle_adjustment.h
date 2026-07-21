// Multi-view bundle adjustment over poses and structure.

#ifndef CVLIB_CALIB3D_BUNDLE_ADJUSTMENT_H_
#define CVLIB_CALIB3D_BUNDLE_ADJUSTMENT_H_

#include "../types.h"
#include "../error_codes.h"
#include "../optimize/lm.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

/*
Bundle adjustment input/output bundle.
Observations matrix is K x 4 with columns [cam_idx, point_idx, u, v];
indices must be finite integer values in range.

Stereo observations are optional rows from a rectified stereo pair whose
right camera sits at +baseline along the left camera x-axis. Each row is
[cam_idx, point_idx, u_left, v, u_right] and contributes a third residual
u_right_pred = u_left_pred - fx * baseline / z_cam, which pins the point
depth metrically and removes the scale gauge freedom of mono-only bundle
adjustment. Rectification implies undistorted pixels, so stereo
observations require dist_coeff == null.

*/

struct BAData {
    Matrix*       poses;           // M x 12 (R 9 row-major + t 3) in/out
    Matrix*       points;          // N x 3 in/out
    const Matrix* observations;    // K x 4
    const Matrix* k;               // shared intrinsics (3x3)
    const Vector* dist_coeff;      // may be null
    const Matrix* stereo_observations = nullptr;  // K_s x 5, may be null
    float64_t     stereo_baseline = 0.0;  // required > 0 with stereo rows
};

// Linear-solver selection for the LM normal equations.
// kBASolverDense forms the full (6M + 3N)^2 system and factors it densely;
// kBASolverSchur eliminates the point blocks first (Schur complement) and
// factors only the 6M x 6M reduced camera system, so memory and solve cost
// scale with the camera count instead of the point count. Both solvers use
// the same damping schedule and termination criteria and converge to the
// same solution up to floating-point rounding.

static constexpr int32_t kBASolverDense = 0;
static constexpr int32_t kBASolverSchur = 1;

// Jacobian-mode selection for the mono observation blocks.
// kBAJacobianAnalytic composes the closed-form projection chain
// (projection_jacobians.h); kBAJacobianAutodiff propagates Jet<9> dual
// numbers through a templated reprojection functor. Both evaluate the same
// derivative at delta = 0 and agree to floating-point rounding; stereo
// observation blocks always use autodiff.

static constexpr int32_t kBAJacobianAnalytic = 0;
static constexpr int32_t kBAJacobianAutodiff = 1;

// Solver options for bundle_adjustment.
//
// min_camera_observations guards against underdetermined camera blocks:
// a free camera whose observation count (mono + stereo) falls below the
// floor makes its 6x6 normal-equation block (near-)singular and the
// solver would silently return garbage, so such problems are rejected
// with kInvalidDimension before solving. Cameras held by
// fixed_pose_count are exempt. The check is opt-in: the default of 0
// disables it, and 3 is the recommended floor for free 6-DOF cameras.
//
// fixed_pose_count holds the leading N poses exactly fixed (their
// Jacobian blocks are dropped and the manifold update skips them),
// which also anchors the gauge: with N >= 1 the optimized solution is
// expressed relative to the fixed pose instead of drifting freely.

struct BAOptions {
    int32_t              perturb_mode;
    int32_t              solver;         // kBASolverDense | kBASolverSchur
    int32_t              jacobian_mode;  // kBAJacobianAnalytic | kBAJacobianAutodiff
    int32_t              min_camera_observations;  // 0 disables the floor
    int32_t              fixed_pose_count;         // leading poses held fixed
    optimize::LMOptions  lm;
};

BAOptions default_ba_options();

/*
Runs bundle adjustment with shared intrinsics; refines all poses and points
jointly via Levenberg-Marquardt on SE(3) x R^3 manifold.

@param data Problem data; poses and points modified in place.
@param options Optional solver options; null uses default_ba_options().
@param report Optional output report; may be null.
@returns ErrorCode.

*/

ErrorCode bundle_adjustment(BAData* data,
                            const BAOptions* options = nullptr,
                            optimize::OptimizeReport* report = nullptr);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_BUNDLE_ADJUSTMENT_H_
