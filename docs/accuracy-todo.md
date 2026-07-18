# Accuracy TODO — closing the gap to GT (2026-07-18)

Measured state on KITTI 00, scale-aligned ATE against `image/GTpose.txt`:

| run (4541 frames) | closures | PGO applied | ATE RMSE | ATE max |
| --- | --- | --- | --- | --- |
| no pose graph (`pgo_enabled=0`) | 1 | — | 150.0 m | 338.9 m |
| **measured Sim(3) loop edges** | 11 | 10 | **87.3 m** | **173.6 m** |
| scale-free loop edges, `pgo_scale_weight=1` | 16 | 2 | 143.3 m | 340.6 m |
| scale-free loop edges, `pgo_scale_weight=50` | 16 | 14 | 154.2 m | 282.6 m |

Measuring the loop Sim(3) from 3D-3D correspondences cut ATE by 42% and the
worst-case error by 49%. The two scale-free rows are what the pose graph did
before that measurement existed, kept here because they document the failure
mode below.

## 1. The scale-free loop edge let the optimizer cheat the gauge (fixed)

`append_loop_edge` used to build the constraint from the essential-matrix
rotation plus a hard zero-translation revisit assumption, with the scale weight
set to 0. A Sim(3) world has a free gauge: expanding the world coordinates by
1/λ while scaling every node's `log_s` by λ leaves all sequential residuals
untouched, because they are relative measurements between nodes that move
together. The loop edge was the exception — its measurement was `t = 0`, which
is scale-free, so its residual shrank linearly with the gauge. The optimizer
satisfied the loop by rescaling the free segment instead of bending the
trajectory, resisted only by the single boundary edge to the fixed nodes.

The evidence matched exactly: the correction scale got worse the longer the
drift it was asked to absorb (0.084 at frame 1575, 0.035 at 3288, 0.0062 at
4449, 0.0023 at 4508), the cost collapsed (24207 -> 2.7), the loop gap
"closed" (116 m -> 0.01 m), and the scale-aligned ATE did not move — a pure
gauge change.

- [x] Estimate the loop transform as a real Sim(3) from 3D-3D correspondences.
      Each side triangulates its own keypoints against a keyframe
      `metric_neighbor_gap` back, so each cloud carries the VO scale of its own
      moment; Umeyama with RANSAC over the loop inliers that have depth on both
      sides returns rotation, translation and the scale ratio. Correction
      scales now land in 0.39..1.29 instead of collapsing.
- [x] Drop closures whose Sim(3) cannot be measured (`metric_required`). No
      closure is recorded, so the next frame retries — in practice a revisit
      that fails at one frame succeeds a few frames later (1575, 1578, 1579
      failed; 1580 succeeded).
- [ ] The measurement is thin: most closures land on 12-21 similarity inliers,
      the minimum being 12. More local depth per keyframe (denser ORB, longer
      baseline, or reusing tracked map points instead of re-matching) would
      make the scale estimate less noisy.
- [ ] `pgo_scale_weight` is still a blunt penalty on the gauge direction rather
      than a physical prior; revisit now that loop edges constrain scale.

## 2. Drift between closures is unconstrained

This is now the dominant error source. The first closure lands at frame 1580,
so the frontend dead-reckons for a third of the sequence before any loop can
help, and the sequential edges fed to the graph come from that same drifting
estimate, so the graph cannot know they are wrong.

- [ ] Add keyframe selection plus a local BA over covisible keyframes. Drift
      has to be small before the loop, not repaired after it.
- [ ] Report KITTI segment-wise translation/rotation error (100..800 m) next to
      ATE. A single ATE number hides whether the problem is local drift or one
      bad stretch.
- [ ] Lower `min_score` / relax `min_consecutive_detections` only after the
      3D-consistency check below exists, and measure closure count vs. false
      positives.

## 3. The correction reaches only part of the state

`apply_loop_correction` pulls the live poses, the active map points, their
anchor poses, and those points' archive entries into the corrected frame.
Historical points do not move, because the correction that belongs to each of
them is the one of *their* reference keyframe, which is not tracked per point.

- [ ] Store a reference keyframe per map point and correct the full archive
      with that keyframe's own correction.
- [ ] Until then, treat `state->all_map_points` (the Rerun history cloud) as a
      display buffer in a stale frame; it is not usable as a map.

## 4. Global BA is disabled and does not feed back

`gba_enabled=0` by default, and `run_loop_global_ba` only returns centers for
visualization — its refined poses and points never reach tracking or the
keyframe database.

- [ ] Decide whether GBA is a publish-only stage or part of the estimate. If it
      is part of the estimate, it must return corrected poses/points and go
      through the same `apply_loop_correction` path as the pose graph.
- [ ] Guard it with the same scale sanity check as the pose graph.

## 5. Verification cannot reject a wrong closure

Geometry verification is 2D-2D only (fundamental RANSAC plus essential
decomposition). A repeated structure that matches well enough produces a
verified closure with no map-level contradiction.

- [ ] Add a 3D check: PnP the candidate keyframe's matched map points against
      the query frame and require an inlier ratio.
- [ ] Log rejected closures with the reason so false-positive rate is
      measurable.

## 6. Evaluation is ad hoc

The numbers above came from a throwaway script; `docs/sim3-pgo-plan.md` refers
to an `eval_ate.py` that is not in the repository.

- [ ] Commit one evaluation script (scale-aligned ATE + KITTI segment errors)
      and wire it into the run scripts behind a flag, so every change is
      measured the same way.
- [ ] Record a baseline table per change in this file.

## 7. The evaluation baseline moves with the configuration

`trajectory_raw.txt` holds the frontend estimate, but corrections feed back into
tracking, so it is not a fixed reference: the same binary produced 142.7 m and
149.7 m of "raw" ATE under two pose graph settings. The only honest baseline is
a separate `pgo_enabled=0` run.

- [ ] Have the evaluation script always run the no-pose-graph baseline
      alongside the configuration under test.

## Notes on what is already in place

- Loop closures at the same place are suppressed by time and distance
  (`duplicate_frame_gap`, `duplicate_distance`, `duplicate_match_window`), so a
  revisit contributes one constraint instead of dozens.
- The pose graph runs as soon as `pgo_pending_trigger` closures are pending
  instead of waiting out the episode gap.
- A correction whose scale leaves `[1/pgo_max_scale_change,
  pgo_max_scale_change]` is rejected outright, so a degenerate solution cannot
  be handed to tracking. The bound is 3.0: with measured loop edges a large
  scale correction is a legitimate answer to real drift, and the 1.5 bound that
  was right for gauge-degenerate solutions rejected valid ones.
- A measured loop scale outside `[1/metric_max_scale_ratio,
  metric_max_scale_ratio]` is discarded as a RANSAC accident before it can
  reach the graph.
- Keyframe poses are rebased onto the optimized trajectory, and
  `trajectory_raw.txt` keeps the uncorrected frontend estimate for comparison.
