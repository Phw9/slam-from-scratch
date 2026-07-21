// 3D-3D point-set alignment: rigid (rotation + translation) and
// similarity (rotation + translation + scale) registration.

#ifndef CVLIB_CALIB3D_ALIGNMENT_H_
#define CVLIB_CALIB3D_ALIGNMENT_H_

#include "../types.h"
#include "../error_codes.h"
#include "../ransac_params.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

// Shared RANSAC parameters; see cvlib/ransac_params.h.
using RansacParams = ::cvlib::RansacParams;

/*
Computes the rigid transform that aligns src to dst: dst ~= R src + t.

Solves the least-squares rotation from the SVD of the centered
cross-covariance, with a determinant sign correction so the result is a
proper rotation (no reflection).

@param src Source 3D points, N-by-3 (N >= 3).
@param dst Destination 3D points, N-by-3.
@param r_out Output 3-by-3 rotation.
@param t_out Output translation, length 3.
@returns ErrorCode.
*/
ErrorCode rigid_transform_3d(const Matrix* src, const Matrix* dst,
                             Matrix* r_out, Vector* t_out);

/*
Computes the similarity transform that aligns src to dst:
dst ~= scale * R src + t.

Same construction as rigid_transform_3d plus the least-squares scale
factor, which is the ratio of the sign-corrected singular-value sum of
the cross-covariance to the centered source variance. Degenerate source
sets (all points coincident) return kSingularMatrix.

@param src Source 3D points, N-by-3 (N >= 3).
@param dst Destination 3D points, N-by-3.
@param r_out Output 3-by-3 rotation.
@param t_out Output translation, length 3.
@param scale_out Output scale factor (> 0 for well-posed inputs).
@returns ErrorCode.
*/
ErrorCode similarity_align(const Matrix* src, const Matrix* dst,
                           Matrix* r_out, Vector* t_out,
                           float64_t* scale_out);

/*
RANSAC similarity-transform estimator over 3D-3D correspondences.
Minimal 3-point similarity hypotheses are scored by the Euclidean
residual ||dst_i - (scale * R src_i + t)||, and the largest consensus
set (earliest hypothesis wins ties) is refit with similarity_align.
Samples through the shared deterministic core: same input and seed
produce the same output on every platform.

@param src Source 3D points, N-by-3 (N >= 3).
@param dst Destination 3D points, N-by-3.
@param params RANSAC parameters; inlier_thresh is the Euclidean
       residual and min_inliers must be >= 3.
@param r_out Output 3-by-3 rotation (refit on inliers).
@param t_out Output translation, length 3.
@param scale_out Output scale factor.
@param inlier_mask Optional output mask, length N (0/1 per row).
@param num_inliers Output number of inliers used in the refit.
@returns ErrorCode (kNotConverged when no consensus reaches
  min_inliers).
*/
ErrorCode similarity_transform_ransac(const Matrix* src, const Matrix* dst,
                                      RansacParams params,
                                      Matrix* r_out, Vector* t_out,
                                      float64_t* scale_out,
                                      int32_t* inlier_mask,
                                      int32_t* num_inliers);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_ALIGNMENT_H_
