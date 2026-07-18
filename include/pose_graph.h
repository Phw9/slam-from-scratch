#pragma once

#include "loop_closure.h"
#include "parameters.h"
#include "types.h"

#include "../thirdparty/cvlib/include/types.h"
#include "../thirdparty/cvlib/include/error_codes.h"
#include "../thirdparty/cvlib/include/optimize/lm.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace mvo {

/*
Pose graph input/output bundle, written in cvlib API style so the core can
move into cvlib next to calib3d/bundle_adjustment.h.
Poses matrix is M x 12 (R 9 row-major + t 3), world-to-camera.
Edges matrix is K x 14 with columns [from_idx, to_idx, R 9 row-major, t 3]
holding the measured relative transform Z = T_to * T_from^-1; indices must
be finite integer values in range.
Weights matrix is K x 2 with columns [translation_weight, rotation_weight];
null uses unit weights.
*/

struct PoseGraphData {
    cvlib::Matrix*       poses;    // M x 12 in/out
    const cvlib::Matrix* edges;    // K x 14
    const cvlib::Matrix* weights;  // K x 2, may be null
};

// Solver options for pose_graph_optimization.

struct PoseGraphOptions {
    int32_t                    fixed_pose_count;  // leading poses held fixed
    cvlib::optimize::LMOptions lm;
};

PoseGraphOptions default_pose_graph_options();

/*
Runs pose graph optimization; refines all non-fixed poses via
Levenberg-Marquardt on the SE(3) manifold. The residual of edge k is
log(Z_k^-1 * T_to * T_from^-1) weighted per edge on [rho; phi].

@param data Problem data; poses modified in place.
@param options Optional solver options; null uses default_pose_graph_options().
@param report Optional output report; may be null.
@returns ErrorCode.
*/

cvlib::ErrorCode pose_graph_optimization(
    PoseGraphData* data,
    const PoseGraphOptions* options = nullptr,
    cvlib::optimize::OptimizeReport* report = nullptr);

/*
Sim(3) right correction C = S_old^-1 * S_new of the newest graph node, where a
Sim(3) acts as x_cam = scale * R * x_w + t. The optimization moves the map into
a new frame with its own scale, so the live tracking state has to follow it:
poses update as S_live * C and world points as C^-1(X). Without this the
frontend would keep triangulating in the pre-loop scale while the graph
publishes another one, and the two estimates silently diverge.
*/

struct LoopCorrection {
    double r[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double t[3] = {0.0, 0.0, 0.0};
    double scale = 1.0;
    bool valid = false;
};

Pose correct_pose(const LoopCorrection& correction, const Pose& pose);
cv::Point3f correct_point(const LoopCorrection& correction,
                          const cv::Point3f& point);

/*
Moves the live tracking state into the corrected frame: current and previous
pose, active map point positions and their anchor poses, and the archived
positions of those points. Historical map points keep their original frame
because their own keyframe correction is not tracked per point.
*/

void apply_loop_correction(const LoopCorrection& correction,
                           TrackState* state,
                           MapArchive* archive);

/*
Builds a pose graph from stored keyframes (sequential VO edges) and verified
loop closures (loop edges with the essential-matrix rotation and a zero
translation revisit constraint), optimizes it, and returns corrected camera
centers for every keyframe.

On acceptance the keyframe poses are rewritten to the optimized trajectory and
`correction` receives the newest node's Sim(3) correction so the caller can pull
the live tracking state into the same frame; a correction whose scale leaves
[1/pgo_max_scale_change, pgo_max_scale_change] is rejected instead, because
handing tracking a large scale jump costs more than skipping the closure.

Runs with hysteresis: while the revisit episode is still producing verified
closures the constraints only accumulate, and the optimization fires once no
new closure has arrived for pgo_episode_end_gap frames, once pgo_pending_trigger
closures are waiting, or immediately when force is set for the end-of-run flush.
Poses recorded before the earliest loop anchor are held fixed so the correction
stays inside the loop segment.
*/

bool run_pose_graph_optimization(BowDatabase* db,
                                 const LoopClosureParameters& parameters,
                                 int32_t frame_id,
                                 bool force,
                                 bool debug_geometry,
                                 std::vector<cv::Point3f>* optimized_centers,
                                 std::vector<Pose>* corrected_poses,
                                 LoopCorrection* correction);

/*
Global bundle adjustment stage that runs after the pose graph: selects a
camera subset from the keyframes, collects archived map-point observations
plus loop-pair points triangulated from the PGO-corrected poses, and
refines poses and structure jointly with the cvlib Schur-complement BA
solver. Returns corrected centers for every keyframe on success.
*/

bool run_loop_global_ba(const BowDatabase& db,
                        const std::vector<Pose>& corrected_poses,
                        const MapArchive& archive,
                        const CameraIntrinsics& camera,
                        const LoopClosureParameters& parameters,
                        int32_t frame_id,
                        bool debug_geometry,
                        std::vector<cv::Point3f>* optimized_centers);

}  // namespace mvo
