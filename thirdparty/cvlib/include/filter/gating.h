// Innovation gating and filter-consistency statistics.

#ifndef CVLIB_FILTER_GATING_H_
#define CVLIB_FILTER_GATING_H_

#include "types.h"
#include "error_codes.h"

namespace cvlib {
namespace filter {

/*
Computes the squared Mahalanobis distance d^2 = r^T S^-1 r of an
innovation r with covariance S. This is the Normalized Innovation
Squared (NIS) statistic used for chi-square filter consistency checks.
S is inverted with the shared SPD helper (Cholesky solve with a
pseudoinverse fallback); tiny negative results from the fallback are
clamped to zero.

@param innovation Innovation (measurement residual), length m.
@param innovation_cov Innovation covariance S, m-by-m SPD.
@param dist_sq Output squared Mahalanobis distance (>= 0).
@returns ErrorCode.
*/
ErrorCode mahalanobis_sq(const Vector* innovation,
                         const Matrix* innovation_cov, float64_t* dist_sq);

/*
Chi-square innovation gate. Accepts the measurement when the NIS
r^T S^-1 r is at or below gate_threshold. Pass the chi-square quantile
matching the measurement dimension and confidence, e.g. 7.815 for
3 DOF at 95%.

@param innovation Innovation (measurement residual), length m.
@param innovation_cov Innovation covariance S, m-by-m SPD.
@param gate_threshold Chi-square acceptance threshold (> 0).
@param accepted Output acceptance flag.
@param dist_sq Optional output of the computed NIS; may be null.
@returns ErrorCode.
*/
ErrorCode innovation_gate(const Vector* innovation,
                          const Matrix* innovation_cov,
                          float64_t gate_threshold, bool* accepted,
                          float64_t* dist_sq = nullptr);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_GATING_H_
