// Real-valued matrix decompositions: QR, eigen (symmetric), determinant.

#ifndef CVLIB_LINALG_DECOMPOSITION_H_
#define CVLIB_LINALG_DECOMPOSITION_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace linalg {

/*
Sorts indices so values[i] appear in descending magnitude order.

@param values Input array of length n.
@param n Length of values and indices.
@param indices Output permutation of indices, length n.
@returns ErrorCode.
*/

ErrorCode argsort_desc(const float64_t* values, int32_t n, int32_t* indices);

/*
Computes the determinant of a real square matrix via LU partial pivoting.

@param m Input n-by-n matrix.
@param result Output determinant.
@returns ErrorCode.
*/

ErrorCode det(const Matrix* m, float64_t* result);

/*
Real QR factorization A = Q R via Modified Gram-Schmidt (reduced form).
R diagonal entries are non-negative, matching numpy.linalg.qr conventions.

Rank-deficient input is rejected: when an orthogonalized column collapses
below kDefaultTolerance times its original norm (duplicate, linearly
dependent, or zero columns), the function returns kSingularMatrix instead
of silently emitting a non-orthonormal Q. Near-singular but numerically
full-rank inputs still factorize; use rank() or svd() to diagnose
conditioning.

@param a Input m-by-n matrix with m >= n.
@param q Output m-by-n with orthonormal columns; pre-allocated.
@param r Output n-by-n upper triangular; pre-allocated.
@returns ErrorCode (kSingularMatrix on rank-deficient input).
*/

ErrorCode qr(const Matrix* a, Matrix* q, Matrix* r);

// Default Jacobi-eigen iteration cap and convergence tolerance.
static constexpr int32_t   kEighDefaultMaxIter   = 1000;
static constexpr float64_t kEighDefaultTolerance = 1e-12;

/*
Symmetric eigendecomposition A = V D V^T via cyclic Jacobi rotations.
Inputs are auto-symmetrized; eigenvalues sorted in descending order.
Output matches numpy.linalg.eigh / cv::eigen for symmetric matrices.

@param a Input n-by-n (treated as symmetric).
@param eigenvalues Output length-n vector, descending order; pre-allocated.
@param eigenvectors Output n-by-n orthogonal matrix (columns); pre-allocated.
@param max_iter Maximum Jacobi sweeps (default kEighDefaultMaxIter).
@param tol Off-diagonal convergence tolerance (default kEighDefaultTolerance).
@returns ErrorCode.
*/

ErrorCode eigh(const Matrix* a, Vector* eigenvalues, Matrix* eigenvectors,
               int32_t max_iter = kEighDefaultMaxIter,
               float64_t tol = kEighDefaultTolerance);

/*
LU decomposition with partial pivoting: P A = L U for a general square
matrix. L is unit lower triangular, U upper triangular, and perm
encodes the row permutation (row i of P A is row perm[i] of A).

@param a Input n-by-n matrix.
@param l Output n-by-n unit lower triangular factor; pre-allocated.
@param u Output n-by-n upper triangular factor; pre-allocated.
@param perm Output row permutation, length n.
@returns ErrorCode (kSingularMatrix when a pivot vanishes).
*/
ErrorCode lu(const Matrix* a, Matrix* l, Matrix* u, int32_t* perm);

/*
Solves A x = b from the lu() factors by permuted forward and back
substitution.

@param l Unit lower triangular factor from lu().
@param u Upper triangular factor from lu().
@param perm Row permutation from lu(), length n.
@param b Right-hand side, length n.
@param x Output solution, length n; pre-allocated.
@returns ErrorCode (kSingularMatrix on a zero diagonal in u).
*/
ErrorCode lu_solve(const Matrix* l, const Matrix* u, const int32_t* perm,
                   const Vector* b, Vector* x);

/*
Projects a symmetric matrix onto the nearest symmetric
positive-(semi)definite matrix in the Frobenius norm: symmetrize,
eigendecompose, clamp eigenvalues to min_eigenvalue, and recompose.
Use to recondition covariances that drift in long-running filters.

@param a Input n-by-n matrix (auto-symmetrized).
@param min_eigenvalue Eigenvalue floor (>= 0); 0 yields the PSD
       projection, a positive floor yields strict SPD.
@param result Output n-by-n symmetric matrix; pre-allocated.
@returns ErrorCode.
*/
ErrorCode nearest_spd(const Matrix* a, float64_t min_eigenvalue,
                      Matrix* result);

}  // namespace linalg
}  // namespace cvlib

#endif  // CVLIB_LINALG_DECOMPOSITION_H_
