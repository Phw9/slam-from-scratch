# Sim(3) PGO implementation plan (in progress, 2026-07-18)

Why: GT evaluation (Sim3-aligned ATE, KITTI 00 1700f) showed raw VO 30.3m
vs SE(3) PGO+BA corrected 60.7m — SE(3) cannot represent monocular scale
drift, so closing the t=0 loop bends the trajectory. Full BA does NOT need
Sim3 (reprojection handles scale implicitly); only the PGO stage does.
Published output = better of {Sim3 PGO, Sim3 PGO + Schur full BA} by ATE.

## Design (all inside src/pose_graph.cpp wrapper section; SE(3) core stays)

- `struct Sim3 { double r[9]; double t[3]; double log_s; };`
  action x_cam = s·R·x_w + t, s = exp(log_s).
  - from Pose: {r, t, 0}; to Pose: R=r, t_se3 = t / s.
  - inverse: s'=1/s, R'=R^T, t' = -(1/s)·R^T·t.
  - compose(a,b): R=Ra·Rb, s=sa·sb, t = sa·Ra·tb + ta.
- Optimization via cvlib::optimize::levenberg_marquardt directly:
  - params: n_free × 13 [r9, t3, log_s]; n_local = n_free × 7.
  - plus_fn (retraction): R←so3_exp(δφ)·R, t←t+δt, log_s←log_s+δσ
    (cvlib so3_exp with 3x3 scratch in ctx via user_data).
  - residual_fn per edge (7 values): E = Z⁻¹∘S_to∘S_from⁻¹ (Z is the
    stored SE3 measurement with s=1); r = [w_t·t_E, w_r·so3_log(R_E),
    w_s·ln(s_E)] using ctx scratch for so3_log.
- weights become K×3 [w_t, w_r, w_s]: sequential edges (1,1,1) — VO
  asserts neighbor scale equality softly; loop edges (w_t, w_r, 0) —
  relative scale unknown, freeing scale to redistribute along the chain
  is exactly what closes the loop without bending.
- Keep: node subsampling, earliest-anchor fixing (fixed leading poses),
  episode hysteresis, edge matrices (K×14), rotation gate.
- After LM: convert node Sim3 → SE3 Pose rows into the existing 12-col
  poses matrix so gap diagnostics + compute_optimized_centers +
  run_loop_global_ba (unchanged) consume it as before. Skip the SE3
  recompute cross-check for sim3 (log -1).

## Validation

1700f run then `eval_ate.py GTpose.txt trajectory_raw.txt
trajectory_corrected.txt`: expect corrected ATE well below 30.3m; compare
PGO-only vs PGO+BA (toggle gba_enabled) and publish the better one.
