// Multi-view bundle adjustment over poses and structure.

#ifndef CVLIB_CALIB3D_BUNDLE_ADJUSTMENT_H_
#define CVLIB_CALIB3D_BUNDLE_ADJUSTMENT_H_

#include "types.h"
#include "error_codes.h"
#include "optimize/lm.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

/*
Bundle adjustment input/output bundle.
Observations matrix is K x 4 with columns [cam_idx, point_idx, u, v];
indices must be finite integer values in range.

*/

struct BAData {
    Matrix*       poses;           // M x 12 (R 9 row-major + t 3) in/out
    Matrix*       points;          // N x 3 in/out
    const Matrix* observations;    // K x 4
    const Matrix* k;               // shared intrinsics (3x3)
    const Vector* dist_coeff;      // may be null
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

// Solver options for bundle_adjustment.

struct BAOptions {
    int32_t              perturb_mode;
    int32_t              solver;        // kBASolverDense | kBASolverSchur
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
