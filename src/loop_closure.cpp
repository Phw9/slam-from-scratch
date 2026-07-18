#include "loop_closure.h"

#include "converter.h"

#include <calib3d/multiview.h>

#include <opencv2/features2d.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <utility>
#include <vector>

namespace mvo {
namespace {

const LoopKeyframe* find_best_candidate(const BowDatabase& db,
                                        const DBoW2::BowVector& bow,
                                        int32_t frame_id,
                                        const LoopClosureParameters& parameters,
                                        double* best_score) {
    const LoopKeyframe* best = nullptr;
    *best_score = 0.0;
    for (const LoopKeyframe& keyframe : db.keyframes) {
        if (frame_id - keyframe.frame_id > parameters.recent_exclusion) {
            const double score = db.vocabulary.score(bow, keyframe.bow);
            if (score > *best_score) {
                *best_score = score;
                best = &keyframe;
            }
        }
    }
    return best;
}

int32_t match_keyframe_points(const cv::Mat& query_descriptors,
                              const std::vector<cv::KeyPoint>& query_keypoints,
                              const LoopKeyframe& candidate,
                              double match_ratio,
                              std::vector<cv::Point2f>* query_points,
                              std::vector<cv::Point2f>* match_points) {
    query_points->clear();
    match_points->clear();
    if (!query_descriptors.empty() && !candidate.descriptors.empty()) {
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(query_descriptors, candidate.descriptors,
                         knn_matches, 2);
        for (const std::vector<cv::DMatch>& match : knn_matches) {
            if (match.size() == 2U &&
                match[0].distance <
                    static_cast<float>(match_ratio) * match[1].distance) {
                query_points->push_back(
                    query_keypoints[static_cast<std::size_t>(
                        match[0].queryIdx)].pt);
                match_points->push_back(
                    candidate.keypoints[static_cast<std::size_t>(
                        match[0].trainIdx)].pt);
            }
        }
    }
    return static_cast<int32_t>(query_points->size());
}

bool recover_loop_relative_pose(const std::vector<cv::Point2f>& match_points,
                                const std::vector<cv::Point2f>& query_points,
                                const CameraIntrinsics& camera,
                                Pose* relative_pose) {
    cvlib::Matrix pix0 = points2f_to_matrix(match_points);
    cvlib::Matrix pix1 = points2f_to_matrix(query_points);
    cvlib::Matrix norm0 = points2f_to_normalized_matrix(match_points, camera);
    cvlib::Matrix norm1 = points2f_to_normalized_matrix(query_points, camera);
    cvlib::Matrix k = make_camera_matrix(camera);
    cvlib::Matrix e = cvlib::matrix_create(3, 3);
    cvlib::Matrix r = cvlib::matrix_create(3, 3);
    cvlib::Vector t = cvlib::vector_create(3);

    cvlib::ErrorCode ec =
        cvlib::calib3d::find_essential_matrix(&pix0, &pix1, &k, &e);
    if (ec == cvlib::ErrorCode::kSuccess) {
        ec = cvlib::calib3d::recover_pose_from_essential(
            &e, &norm0, &norm1, &r, &t);
    }
    const bool ok = ec == cvlib::ErrorCode::kSuccess;
    if (ok) {
        copy_cvlib_pose(&r, &t, relative_pose);
    }

    cvlib::matrix_destroy(&pix0);
    cvlib::matrix_destroy(&pix1);
    cvlib::matrix_destroy(&norm0);
    cvlib::matrix_destroy(&norm1);
    cvlib::matrix_destroy(&k);
    cvlib::matrix_destroy(&e);
    cvlib::matrix_destroy(&r);
    cvlib::vector_destroy(&t);
    return ok;
}

bool verify_loop_geometry(const std::vector<cv::Point2f>& match_points,
                          const std::vector<cv::Point2f>& query_points,
                          const CameraIntrinsics& camera,
                          const LoopClosureParameters& parameters,
                          int32_t* inliers,
                          Pose* relative_pose,
                          std::vector<cv::Point2f>* inlier_match_out,
                          std::vector<cv::Point2f>* inlier_query_out) {
    bool ok = false;
    *inliers = 0;
    inlier_match_out->clear();
    inlier_query_out->clear();
    cvlib::Matrix pix0 = points2f_to_matrix(match_points);
    cvlib::Matrix pix1 = points2f_to_matrix(query_points);
    cvlib::Matrix f = cvlib::matrix_create(3, 3);
    std::vector<int32_t> mask(match_points.size(), 0);
    cvlib::calib3d::RansacParams ransac_params;
    ransac_params.max_iters = parameters.ransac_max_iters;
    ransac_params.inlier_thresh = parameters.inlier_threshold;
    ransac_params.min_inliers = parameters.min_inliers;
    ransac_params.seed = 0x4C4F4F50;

    const cvlib::ErrorCode f_ec = cvlib::calib3d::find_fundamental_ransac(
        &pix0, &pix1, ransac_params, &f, mask.data(), inliers);
    if (f_ec == cvlib::ErrorCode::kSuccess &&
        *inliers >= parameters.min_inliers) {
        std::vector<cv::Point2f> inlier_match_points;
        std::vector<cv::Point2f> inlier_query_points;
        inlier_match_points.reserve(static_cast<std::size_t>(*inliers));
        inlier_query_points.reserve(static_cast<std::size_t>(*inliers));
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] != 0) {
                inlier_match_points.push_back(match_points[i]);
                inlier_query_points.push_back(query_points[i]);
            }
        }
        ok = recover_loop_relative_pose(inlier_match_points,
                                        inlier_query_points, camera,
                                        relative_pose);
        if (ok) {
            *inlier_match_out = inlier_match_points;
            *inlier_query_out = inlier_query_points;
        }
        if (ok) {
            // A degenerate essential decomposition can hand back a
            // non-rotation R; a pose-graph edge built from it hides the
            // loop translation instead of closing it.
            const double rotation_error =
                rotation_orthonormality_error(*relative_pose);
            const double det = rotation_determinant(*relative_pose);
            if (rotation_error > parameters.max_rotation_error ||
                std::abs(det - 1.0) > parameters.max_rotation_error) {
                std::cout << "loop_rotation_rejected rot_err="
                          << rotation_error << " det=" << det << std::endl;
                ok = false;
            }
        }
    }

    cvlib::matrix_destroy(&pix0);
    cvlib::matrix_destroy(&pix1);
    cvlib::matrix_destroy(&f);
    return ok;
}

}  // namespace

bool load_vocabulary(const std::string& path, BowDatabase* db) {
    bool ok = false;
    if (std::filesystem::exists(path)) {
        db->vocabulary.load(path);
        db->vocabulary_loaded = !db->vocabulary.empty();
        ok = db->vocabulary_loaded;
    }
    return ok;
}

bool query_and_add_loop(const cv::Mat& image, BowDatabase* db,
                        int32_t frame_id,
                        const Pose& current_pose,
                        const CameraIntrinsics& camera,
                        const LoopClosureParameters& parameters,
                        bool debug_geometry,
                        LoopClosureEvent* closure) {
    bool closed = false;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(parameters.orb_features);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    if (db->vocabulary_loaded && !descriptors.empty()) {
        LoopKeyframe keyframe;
        keyframe.frame_id = frame_id;
        keyframe.keypoints = std::move(keypoints);
        keyframe.descriptors = descriptors;
        keyframe.camera_center = camera_center_from_pose(current_pose);
        keyframe.pose = current_pose;
        const std::vector<cv::Mat> rows = descriptor_rows(descriptors);
        db->vocabulary.transform(rows, keyframe.bow);

        double best_score = 0.0;
        const LoopKeyframe* candidate = find_best_candidate(
            *db, keyframe.bow, frame_id, parameters, &best_score);
        const int32_t best_id = candidate != nullptr ? candidate->frame_id
                                                     : -1;

        if (candidate != nullptr && best_score >= parameters.min_score) {
            // A single high BoW score is not trusted on its own: the same
            // place must win over consecutive queries before geometry runs.
            const bool same_place =
                db->last_candidate_frame >= 0 &&
                std::abs(candidate->frame_id - db->last_candidate_frame) <=
                    parameters.consistency_window;
            db->consistent_detections =
                same_place ? db->consistent_detections + 1 : 1;
            db->last_candidate_frame = candidate->frame_id;
            if (db->consistent_detections >=
                parameters.min_consecutive_detections) {
                std::vector<cv::Point2f> query_points;
                std::vector<cv::Point2f> match_points;
                const int32_t matches = match_keyframe_points(
                    keyframe.descriptors, keyframe.keypoints, *candidate,
                    parameters.match_ratio, &query_points, &match_points);
                int32_t inliers = 0;
                Pose relative_pose = {};
                std::vector<cv::Point2f> inlier_match_points;
                std::vector<cv::Point2f> inlier_query_points;
                const bool verified =
                    matches >= parameters.min_matches &&
                    verify_loop_geometry(match_points, query_points, camera,
                                         parameters, &inliers,
                                         &relative_pose,
                                         &inlier_match_points,
                                         &inlier_query_points);
                if (verified) {
                    LoopClosureEvent event;
                    event.query_frame = frame_id;
                    event.match_frame = candidate->frame_id;
                    event.score = best_score;
                    event.matches = matches;
                    event.inliers = inliers;
                    event.relative_pose = relative_pose;
                    event.match_inlier_points = std::move(inlier_match_points);
                    event.query_inlier_points = std::move(inlier_query_points);
                    event.query_center = keyframe.camera_center;
                    event.match_center = candidate->camera_center;
                    db->closures.push_back(event);
                    if (closure != nullptr) {
                        *closure = event;
                    }
                    closed = true;
                    db->consistent_detections = 0;
                    db->last_candidate_frame = -1;
                    std::cout << "loop_closure query=" << frame_id
                              << " match=" << event.match_frame
                              << " score=" << best_score
                              << " matches=" << matches
                              << " inliers=" << inliers << std::endl;
                } else if (debug_geometry) {
                    std::cout << "loop_rejected query=" << frame_id
                              << " match=" << best_id
                              << " score=" << best_score
                              << " matches=" << matches
                              << " inliers=" << inliers << std::endl;
                }
            }
        } else {
            db->consistent_detections = 0;
            db->last_candidate_frame = -1;
        }

        std::cout << "loop_query frame=" << frame_id
                  << " best_id=" << best_id
                  << " score=" << best_score
                  << " streak=" << db->consistent_detections << std::endl;
        db->keyframes.push_back(std::move(keyframe));
    }
    return closed;
}

}  // namespace mvo
