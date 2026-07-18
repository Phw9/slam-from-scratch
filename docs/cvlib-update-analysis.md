# Why the 2026-07 cvlib update improved odometry

MVO's odometry got noticeably better after `148da95 chore(cvlib): update
bundled cvlib to latest` (2026-07-17): scale holds through long KITTI 00
runs, and the bundle commit itself recorded frame-2 reprojection RMSE
dropping 0.52 -> 0.32 with PnP succeeding 1698/1698 over 1700 frames.

This document pins down every reason by diffing the two cvlib source
snapshots behind the bundles and tracing only the functions MVO actually
calls.

## Bundle snapshots compared

| Bundle commit (MVO) | Date | cvlib source snapshot |
| --- | --- | --- |
| `0d1dbc5` (KLT float32 bundle) | 2026-06-10 | cvlib `573107e` |
| `148da95` (this update) | 2026-07-17 | cvlib `c94a0d0` |

MVO's VO call surface into cvlib: `good_features_to_track`,
`klt_track_f32`, `find_fundamental_ransac`, `find_homography_ransac`,
`find_essential_matrix`, `recover_pose_from_essential`,
`triangulate_points`, `filter_chirality`, `solve_pnp`,
`bundle_adjustment`, plus the `linalg`/`optimize` layers underneath.

## Ruled out (byte-equivalent or off-path between the two bundles)

- **KLT and GoodFeaturesToTrack**: zero source changes between the
  snapshots. Tracking accuracy is not part of the improvement.
- **`solve_pnp` math**: the diff is accessor migration (`matrix_get` ->
  `m(i, j)`) plus a stricter `validate_camera_matrix`; the DLT +
  LM-refinement math is unchanged.
- **LM / Gauss-Newton solvers**: only fail-closed workspace-allocation
  guards were added; the step, damping, and termination math is
  identical.
- **Triangulation / 8-point / homography DLT**: same accessor-only
  refactor; the linear systems are unchanged.
- **SE(3) left/right Jacobian coupling fix (`cbf06f6`)**: real bug fix,
  but `left_jac_se3` is only used inside `sophus.cpp`; BA and PnP use
  the exp-based `se3_plus_*` manifold updates and their own projection
  Jacobians, so MVO's pipeline never evaluated the broken code.
- **`solve_triangular` scale-invariant guard (`4220a50`)**: correctness
  fix, but LM/BA solve their normal equations through
  `cholesky`/`cholesky_solve`, which has its own substitution; nothing
  on the VO hot path calls `solve_triangular`.

## Causes, ranked

### 1. Jacobi SVD/eigh non-convergence now fails closed (`9db44e7`, `09c4a4c`)

Before: `run_jacobi_loop` exited after `max_iter` sweeps even while
rotations were still active, and callers returned `kSuccess` with a
partially rotated factorization — silent garbage. Every DLT solve in the
VO path sits on this SVD: triangulation, the 8-point fundamental matrix
(plus its rank-2 projection), essential-matrix computation and pose
recovery, homography estimation, and PnP's rotation
re-orthogonalization.

Ill-conditioned inputs (low parallax, near-degenerate samples)
occasionally produced wildly wrong triangulated points or two-view
models that slipped past MVO's gates. Those points are exactly what
injects scale drift: PnP then anchors the next pose against corrupted
depths. Now the solver reports `kNotConverged`, the cvlib API returns an
error, and MVO's existing `ec == kSuccess` gates skip the bad output.

### 2. Deterministic distinct-subset RANSAC core (`9756c7f`, with `f2be8a0`, `d6d7763`)

The per-file samplers used `std::uniform_int_distribution`, whose output
is implementation-defined, and were replaced by a shared sampler using
raw `std::mt19937` words with a multiply-shift reduction and a
swap-restored partial Fisher-Yates pass (guaranteed-distinct samples,
no rejection loops). `find_fundamental_ransac` and
`find_homography_ransac` — the two-view initializer's model selection —
therefore explore a different, well-distributed hypothesis sequence for
the same seed and iteration budget.

Monocular scale is *set* by the two-view initialization and re-anchored
on every tracking-loss recovery, so better/steadier init models show up
directly as better long-run scale. This matches the observed init-stat
improvement (frame-2 RMSE 0.52 -> 0.32) with no MVO-side change.

### 3. Central deep-input validation (`c94a0d0`)

`validate_finite_matrix/vector` (NaN/Inf) and `validate_camera_matrix`
(3x3, finite, positive focal lengths) now guard the calib3d entry
points (PnP, essential, homography decomposition, projections). A NaN
that leaks out of tracking can no longer poison a pose or triangulation
silently; the call fails and MVO's failure handling takes over.

### 4. Minor precision/robustness items

- `lstsq` now solves directly from the SVD factors instead of forming
  the full pseudoinverse (`3e1c8bc`) — same minimum-norm solution,
  better conditioned; barely on the VO path.
- Internal allocation failures fail closed with `kNullPointer` across
  optimizers/linalg (`a2e26e5`, `f80b67f`) — robustness, not accuracy.

## Follow-up (2026-07-18 rebundle)

The next rebundle (after this analysis) brought in the Schur-complement
BA solver: `BAOptions.solver = kBASolverDense | kBASolverSchur`, which
eliminates the 3x3 point blocks and factors only the 6M x 6M reduced
camera system. The 07-17 bundle's `bundle_adjustment.h` predates the
`solver` field, so the rebundle also keeps the bundled header and
library in lockstep. MVO selects the solver through
`configs/parameters/bundle_adjustment.json`.
