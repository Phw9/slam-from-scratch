// Perspective-n-Point: DLT initialization and LM refinement.

#ifndef CVLIB_CALIB3D_PNP_H_
#define CVLIB_CALIB3D_PNP_H_

#include "../types.h"
#include "../error_codes.h"
#include "../calib3d/multiview.h"
#include "../optimize/lm.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

static constexpr int32_t kPnpMinPoints = 6;
static constexpr int32_t kPnpStereoMinRefinePoints = 2;

// Options for solve_pnp.
struct SolvePnpOptions {
    int32_t              perturb_mode;
    optimize::LMOptions  lm;
};

SolvePnpOptions default_solve_pnp_options();

/*
Direct Linear Transform PnP estimation.
Computes (R, t) from N >= 6 known correspondences.
World points are 3D (N x 3); image points are pixel coords (N x 2).
Outputs use world-to-camera extrinsics: p_cam = R * p_world + t.

@param world_pts N-by-3 world points.
@param image_pts N-by-2 pixel observations.
@param k Camera intrinsics (3x3).
@param r_3x3 Output rotation (pre-allocated 3x3).
@param t_3 Output translation (pre-allocated length 3).
@param dist_coeff Optional distortion coefficients; null for no distortion.
@returns ErrorCode.
*/
ErrorCode dlt_pnp(const Matrix* world_pts, const Matrix* image_pts,
                  const Matrix* k,
                  Matrix* r_3x3, Vector* t_3,
                  const Vector* dist_coeff = nullptr);

/*
Solves PnP using LM refinement on SE(3); auto-initializes with DLT when
no initial pose is supplied.
Outputs use world-to-camera extrinsics: p_cam = R * p_world + t.

@param world_pts N-by-3 world points.
@param image_pts N-by-2 pixel observations.
@param k Camera intrinsics (3x3).
@param r_out Output rotation (3x3, must be allocated).
@param t_out Output translation (length 3, must be allocated).
@param dist_coeff Optional distortion coefficients; null for no distortion.
@param r_init Optional initial rotation; null triggers DLT initialization.
@param t_init Optional initial translation; null triggers DLT initialization.
@param options Optional solver options; null uses defaults.
@param report Optional output optimization report; may be null.
@returns ErrorCode.
*/
ErrorCode solve_pnp(const Matrix* world_pts, const Matrix* image_pts,
                    const Matrix* k,
                    Matrix* r_out, Vector* t_out,
                    const Vector* dist_coeff = nullptr,
                    const Matrix* r_init = nullptr,
                    const Vector* t_init = nullptr,
                    const SolvePnpOptions* options = nullptr,
                    optimize::OptimizeReport* report = nullptr);

/*
RANSAC PnP: minimal 6-point DLT hypotheses scored by pixel reprojection
error, followed by an LM refit (solve_pnp) on the consensus inlier set
seeded with the best hypothesis. Points behind the camera under a
hypothesis are scored as outliers rather than failing the iteration.
Outputs use world-to-camera extrinsics: p_cam = R * p_world + t.

@param world_pts N-by-3 world points (N >= 6).
@param image_pts N-by-2 pixel observations.
@param k Camera intrinsics (3x3).
@param params RANSAC parameters; inlier_thresh is the reprojection error
       in pixels and min_inliers must be >= 6.
@param r_out Output rotation (3x3, must be allocated).
@param t_out Output translation (length 3, must be allocated).
@param inlier_mask Optional output mask, length N (0/1 per row).
@param num_inliers Output number of inliers used in the refit.
@param dist_coeff Optional distortion coefficients; null for none.
@param options Optional solver options for the refit; null uses defaults.
@param report Optional output optimization report; may be null.
@returns ErrorCode (kNotConverged when no consensus reaches min_inliers).
*/
ErrorCode solve_pnp_ransac(const Matrix* world_pts, const Matrix* image_pts,
                           const Matrix* k, RansacParams params,
                           Matrix* r_out, Vector* t_out,
                           int32_t* inlier_mask, int32_t* num_inliers,
                           const Vector* dist_coeff = nullptr,
                           const SolvePnpOptions* options = nullptr,
                           optimize::OptimizeReport* report = nullptr);

/*
Stereo PnP: pose estimation from rectified stereo observations with the
3-row residual (u_left, v, u_right) per point. The right camera sits at
+baseline along the left camera x-axis, so the third residual row
u_right = u_left - fx * baseline / z pins the depth metrically from a
single frame. Rectified pixels are undistorted, so there is no
distortion input. Jacobians propagate SE(3)-tangent dual numbers
through the same reprojection functor as the bundle-adjustment stereo
rows. Auto-initializes with mono DLT on the (u_left, v) columns when no
initial pose is supplied (requires N >= 6); with an initial pose,
N >= 2 points already overdetermine the 6-DOF refinement.
Outputs use world-to-camera extrinsics for the left camera.

@param world_pts N-by-3 world points.
@param image_pts N-by-3 stereo observations [u_left, v, u_right].
@param k Camera intrinsics (3x3).
@param baseline Stereo baseline (> 0), same unit as world_pts.
@param r_out Output rotation (3x3, must be allocated).
@param t_out Output translation (length 3, must be allocated).
@param r_init Optional initial rotation; null triggers DLT initialization.
@param t_init Optional initial translation; null triggers DLT initialization.
@param options Optional solver options; null uses defaults.
@param report Optional output optimization report; may be null.
@returns ErrorCode.
*/
ErrorCode solve_pnp_stereo(const Matrix* world_pts, const Matrix* image_pts,
                           const Matrix* k, float64_t baseline,
                           Matrix* r_out, Vector* t_out,
                           const Matrix* r_init = nullptr,
                           const Vector* t_init = nullptr,
                           const SolvePnpOptions* options = nullptr,
                           optimize::OptimizeReport* report = nullptr);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_PNP_H_
