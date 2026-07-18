#pragma once

#include "parameters.h"
#include "types.h"

#include "DBoW2/BowVector.h"
#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mvo {

using OrbVocabulary =
    DBoW2::TemplatedVocabulary<cv::Mat, DBoW2::FORB>;

struct LoopKeyframe {
    int32_t frame_id = 0;
    DBoW2::BowVector bow;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Point3f camera_center;
    Pose pose = {};
    // Frontend estimate as it was recorded; `pose` follows the pose graph
    // corrections, so keeping the original around preserves the baseline the
    // corrected trajectory is evaluated against.
    Pose vo_pose = {};
};

struct LoopClosureEvent {
    int32_t query_frame = 0;
    int32_t match_frame = 0;
    double score = 0.0;
    int32_t matches = 0;
    int32_t inliers = 0;
    Pose relative_pose = {};
    // Sim(3) scale of the loop transform, recovered from 3D-3D correspondences
    // between the two revisits. Without it the loop edge says nothing about
    // how long the intervening trajectory should be, and the pose graph can
    // satisfy the constraint by rescaling the gauge instead of correcting the
    // trajectory.
    double relative_scale = 1.0;
    bool has_metric_transform = false;
    int32_t metric_inliers = 0;
    cv::Point3f query_center;
    cv::Point3f match_center;
    std::vector<cv::Point2f> match_inlier_points;
    std::vector<cv::Point2f> query_inlier_points;
};

// A revisit that already produced a verified closure. Later queries at the
// same place stay suppressed until the trajectory has moved far enough away
// or enough frames have passed.
struct LoopClosureSite {
    int32_t query_frame = 0;
    int32_t match_frame = 0;
    cv::Point3f query_center;
    cv::Point3f match_center;
};

struct BowDatabase {
    OrbVocabulary vocabulary;
    bool vocabulary_loaded = false;
    std::vector<LoopKeyframe> keyframes;
    std::vector<LoopClosureEvent> closures;
    std::vector<LoopClosureSite> closed_sites;
    int32_t consistent_detections = 0;
    int32_t last_candidate_frame = -1;
    int32_t last_pgo_frame = -1;
    int32_t optimized_closure_count = 0;
    int32_t pgo_runs = 0;
    std::vector<cv::Point3f> last_corrected_centers;
};

bool load_vocabulary(const std::string& path, BowDatabase* db);
bool query_and_add_loop(const cv::Mat& image, BowDatabase* db,
                        int32_t frame_id,
                        const Pose& current_pose,
                        const CameraIntrinsics& camera,
                        const LoopClosureParameters& parameters,
                        bool debug_geometry,
                        LoopClosureEvent* closure);

}  // namespace mvo
