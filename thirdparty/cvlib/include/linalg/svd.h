// Singular value decomposition (full U, S, Vt).

#ifndef CVLIB_LINALG_SVD_H_
#define CVLIB_LINALG_SVD_H_

#include "../types.h"
#include "../defs.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace linalg {

// Default SVD convergence tolerance.
static constexpr float64_t kSvdDefaultTolerance = 1e-10;

/*
Computes the factorization A = U S V^T with a one-sided Jacobi method
(no external LAPACK). U and Vt are optional and independent: pass
nullptr to skip allocating and writing the corresponding factor.

Output shapes are strict: any other shape returns kInvalidShape rather
than a clamped partial result.

@param a Input matrix, m-by-n.
@param s Output m-by-n matrix; singular values are written descending on
  the diagonal and every off-diagonal entry is zeroed.
@param u Optional m-by-m left singular vectors; null skips computation.
@param vt Optional n-by-n transpose of right singular vectors; null skips.
@param max_iter Upper bound on Jacobi sweeps (default kSvdMaxIterations).
@param tol User convergence tolerance (default kSvdDefaultTolerance).
  Internally capped at 1e-14 (values above the cap are tightened to it)
  so U and Vt stay numerically orthogonal; pass a smaller value only to
  request an even stricter sweep threshold. kNotConverged reports the
  capped tolerance not being reached within max_iter sweeps.
@returns ErrorCode.

*/

ErrorCode svd(const Matrix* a, Matrix* s,
              Matrix* u = nullptr, Matrix* vt = nullptr,
              int32_t max_iter = kSvdMaxIterations,
              float64_t tol = kSvdDefaultTolerance);

/*
Spectral condition number: the ratio of the largest to the smallest
singular value of a.

@param a Input matrix, m-by-n.
@param result Output condition number (>= 1 for nonzero matrices).
@returns ErrorCode (kSingularMatrix when the smallest singular value
  is not positive).
*/
ErrorCode cond(const Matrix* a, float64_t* result);

}  // namespace linalg
}  // namespace cvlib

#endif  // CVLIB_LINALG_SVD_H_
