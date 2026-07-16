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
};

struct LoopClosureEvent {
    int32_t query_frame = 0;
    int32_t match_frame = 0;
    double score = 0.0;
    int32_t matches = 0;
    int32_t inliers = 0;
    Pose relative_pose = {};
    cv::Point3f query_center;
    cv::Point3f match_center;
};

struct BowDatabase {
    OrbVocabulary vocabulary;
    bool vocabulary_loaded = false;
    std::vector<LoopKeyframe> keyframes;
    std::vector<LoopClosureEvent> closures;
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
