// Shared RANSAC estimator parameters.

#ifndef CVLIB_RANSAC_PARAMS_H_
#define CVLIB_RANSAC_PARAMS_H_

#include <cstdint>

namespace cvlib {

using float64_t = double;

/*
Tunable parameters shared by every RANSAC-based robust estimator.

Determinism contract: for the same inputs and the same seed, every RANSAC
function produces the same output on every platform. Estimators must
sample through cvlib::internal::RansacSampler and run exactly max_iters
fixed iterations.

@param max_iters Fixed number of hypothesis iterations (> 0).
@param inlier_thresh Estimator-specific inlier residual threshold (> 0).
@param min_inliers Minimum consensus size for success.
@param seed Sampler seed; equal seeds reproduce the sample sequence.
*/
struct RansacParams {
    int32_t   max_iters;
    float64_t inlier_thresh;
    int32_t   min_inliers;
    uint32_t  seed;
};

}  // namespace cvlib

#endif  // CVLIB_RANSAC_PARAMS_H_
