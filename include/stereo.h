#pragma once

#include "parameters.h"
#include "types.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace mvo {

// Per-frame pose record for the stereo backend. Stereo has no loop-closure
// stage; local and full bundle adjustment read the trajectory from here.
struct StereoFramePose {
    int32_t frame_id = 0;
    Pose pose = {};
};

struct StereoBackend {
    std::vector<StereoFramePose> trajectory;
    std::vector<StereoFramePose> optimized;
    int32_t local_ba_runs = 0;
    int32_t local_ba_accepted = 0;
    bool full_ba_accepted = false;
};

/*
Matches the left-image pixel of every unpositioned map point into the right
image with KLT, gates the matches with rectified-epipolar and disparity
limits, and fills in metric positions from the known baseline. Positioned
points get their creation observation archived so bundle adjustment can use
the full track later. Returns the number of newly positioned points.
*/

int32_t triangulate_stereo_map_points(
    const cv::Mat& left,
    const cv::Mat& right,
    const std::vector<cv::Point2f>& left_points,
    const Pose& pose,
    const CameraIntrinsics& camera,
    double baseline,
    const MvoParameters& parameters,
    int32_t frame_id,
    bool debug_geometry,
    std::vector<MapPoint>* map_points,
    std::vector<cv::Point3f>* all_map_points,
    MapArchive* archive);

/*
Builds a fresh track state from a single stereo pair: detects features on the
left image and triangulates them against the right image at metric scale.
Used both for startup (identity pose) and for re-initialization after a
tracking failure (last known pose). Only commits to the state on success.
*/

bool initialize_stereo_tracking(const cv::Mat& left,
                                const cv::Mat& right,
                                const Pose& initial_pose,
                                const CameraIntrinsics& camera,
                                double baseline,
                                const MvoParameters& parameters,
                                int32_t frame_id,
                                bool debug_geometry,
                                MapArchive* archive,
                                TrackState* state);

/*
Windowed bundle adjustment over the most recent trajectory poses and the
archived observations of map points seen inside the window. On acceptance the
window poses, the live tracking pose, the live map-point positions, and the
archived positions all move to the refined estimate; the gauge is re-anchored
to the first window pose so the trajectory stays continuous at the window
boundary.
*/

bool run_stereo_local_ba(const CameraIntrinsics& camera,
                         double baseline,
                         const StereoParameters& parameters,
                         int32_t frame_id,
                         bool debug_geometry,
                         MapArchive* archive,
                         StereoBackend* backend,
                         TrackState* state);

/*
Full bundle adjustment over a strided camera subset of the whole trajectory
with the cvlib Schur-complement solver. Corrections for unselected frames are
propagated by transporting each frame's relative pose onto the nearest
optimized camera. The result is published in backend->optimized; the raw
trajectory stays untouched for comparison.
*/

bool run_stereo_full_ba(const CameraIntrinsics& camera,
                        double baseline,
                        const StereoParameters& parameters,
                        bool debug_geometry,
                        const MapArchive& archive,
                        StereoBackend* backend);

}  // namespace mvo
