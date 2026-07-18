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
Builds a pose graph from stored keyframes (sequential VO edges) and verified
loop closures (loop edges with the essential-matrix rotation and a zero
translation revisit constraint), optimizes it, and returns corrected camera
centers for every keyframe. Keyframe poses stay untouched so live tracking
keeps its own frame; the result is for logging and Rerun inspection.

Runs with hysteresis: while the revisit episode is still producing verified
closures the constraints only accumulate, and the optimization fires once no
new closure has arrived for pgo_episode_end_gap frames (or immediately when
force is set, for the end-of-run flush). Poses recorded before the earliest
loop anchor are held fixed so the correction stays inside the loop segment.
*/

bool run_pose_graph_optimization(BowDatabase* db,
                                 const LoopClosureParameters& parameters,
                                 int32_t frame_id,
                                 bool force,
                                 bool debug_geometry,
                                 std::vector<cv::Point3f>* optimized_centers,
                                 std::vector<Pose>* corrected_poses);

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
