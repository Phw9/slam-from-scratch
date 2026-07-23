#include "stereo.h"

#include "converter.h"
#include "feature.h"
#include "map_data.h"

#include <calib3d/bundle_adjustment.h>
#include <calib3d/multiview.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mvo {
namespace {

Pose invert_pose(const Pose& pose) {
    Pose out;
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            out.r[row * 3 + col] = pose.r[col * 3 + row];
        }
    }
    for (int32_t row = 0; row < 3; ++row) {
        double t = 0.0;
        for (int32_t k = 0; k < 3; ++k) {
            t += out.r[row * 3 + k] * pose.t[k];
        }
        out.t[row] = -t;
    }
    return out;
}

cv::Point3f apply_pose_inverse(const Pose& pose, const cv::Point3f& point) {
    const double px = static_cast<double>(point.x) - pose.t[0];
    const double py = static_cast<double>(point.y) - pose.t[1];
    const double pz = static_cast<double>(point.z) - pose.t[2];
    return cv::Point3f(
        static_cast<float>(pose.r[0] * px + pose.r[3] * py + pose.r[6] * pz),
        static_cast<float>(pose.r[1] * px + pose.r[4] * py + pose.r[7] * pz),
        static_cast<float>(pose.r[2] * px + pose.r[5] * py + pose.r[8] * pz));
}

Pose param_row_to_pose(const cvlib::Matrix& poses, int32_t row) {
    Pose pose;
    for (int32_t i = 0; i < 9; ++i) {
        pose.r[i] = cvlib::matrix_get(&poses, row, i);
    }
    for (int32_t i = 0; i < 3; ++i) {
        pose.t[i] = cvlib::matrix_get(&poses, row, 9 + i);
    }
    return pose;
}

// Projects the rotation rows back onto SO(3). Solver output and anchor
// composition each leave ~1e-6 orthonormality drift; published poses feed
// the next window's BA input, so without the projection the drift compounds
// until it trips the degeneracy gate.
void orthonormalize_rotation(Pose* pose) {
    double r0[3] = {pose->r[0], pose->r[1], pose->r[2]};
    double r1[3] = {pose->r[3], pose->r[4], pose->r[5]};
    double r2[3];
    const double n0 = std::sqrt(r0[0] * r0[0] + r0[1] * r0[1] +
                                r0[2] * r0[2]);
    if (n0 < 1.0e-12) {
        return;
    }
    for (double& v : r0) {
        v /= n0;
    }
    r2[0] = r0[1] * r1[2] - r0[2] * r1[1];
    r2[1] = r0[2] * r1[0] - r0[0] * r1[2];
    r2[2] = r0[0] * r1[1] - r0[1] * r1[0];
    const double n2 = std::sqrt(r2[0] * r2[0] + r2[1] * r2[1] +
                                r2[2] * r2[2]);
    if (n2 < 1.0e-12) {
        return;
    }
    for (double& v : r2) {
        v /= n2;
    }
    r1[0] = r2[1] * r0[2] - r2[2] * r0[1];
    r1[1] = r2[2] * r0[0] - r2[0] * r0[2];
    r1[2] = r2[0] * r0[1] - r2[1] * r0[0];
    for (int32_t i = 0; i < 3; ++i) {
        pose->r[i] = r0[i];
        pose->r[3 + i] = r1[i];
        pose->r[6 + i] = r2[i];
    }
}

bool triangulate_seed_pair(const Pose& pose_a, const Pose& pose_b,
                           const cv::Point2f& pixel_a,
                           const cv::Point2f& pixel_b,
                           const CameraIntrinsics& camera,
                           cv::Point3f* point) {
    const std::vector<cv::Point2f> points_a = {pixel_a};
    const std::vector<cv::Point2f> points_b = {pixel_b};
    cvlib::Matrix p_a = pose_to_projection(pose_a);
    cvlib::Matrix p_b = pose_to_projection(pose_b);
    cvlib::Matrix norm_a = points2f_to_normalized_matrix(points_a, camera);
    cvlib::Matrix norm_b = points2f_to_normalized_matrix(points_b, camera);
    cvlib::Matrix points3d = cvlib::matrix_create(1, 3);
    const cvlib::ErrorCode ec = cvlib::calib3d::triangulate_points(
        &p_a, &p_b, &norm_a, &norm_b, &points3d);
    const bool ok = ec == cvlib::ErrorCode::kSuccess;
    if (ok) {
        point->x = static_cast<float>(cvlib::matrix_get(&points3d, 0, 0));
        point->y = static_cast<float>(cvlib::matrix_get(&points3d, 0, 1));
        point->z = static_cast<float>(cvlib::matrix_get(&points3d, 0, 2));
    }
    cvlib::matrix_destroy(&p_a);
    cvlib::matrix_destroy(&p_b);
    cvlib::matrix_destroy(&norm_a);
    cvlib::matrix_destroy(&norm_b);
    cvlib::matrix_destroy(&points3d);
    return ok;
}

struct StereoBaProblem {
    // Original row indices of the cameras kept in the problem, in row order;
    // observation rows use the compact index into this vector.
    std::vector<int32_t> camera_rows;
    std::vector<int32_t> point_ids;
    std::vector<cv::Point3f> point_positions;
    std::vector<std::array<double, 4>> observation_rows;
    // [cam, point, u_left, v, u_right] rows; the metric baseline makes
    // these the scale anchor of the problem.
    std::vector<std::array<double, 5>> stereo_observation_rows;
};

int64_t stereo_lookup_key(int32_t point_id, int32_t frame_id) {
    return (static_cast<int64_t>(point_id) << 32) |
           static_cast<int64_t>(static_cast<uint32_t>(frame_id));
}

/*
Collects archived observations restricted to the given frames, keeps the
best-supported points, and gates every observation on positive depth and a
bounded seed residual so a single bad row cannot stall the LM solver.
Cameras with fewer than min_camera_observations surviving observations are
excluded from the problem entirely: an unconstrained pose block makes the
reduced camera system singular and the solver can leave the SE(3) manifold.
*/

bool build_stereo_ba_problem(const MapArchive& archive,
                             const std::unordered_map<int32_t, int32_t>&
                                 frame_row,
                             const std::vector<Pose>& row_poses,
                             const CameraIntrinsics& camera,
                             int32_t min_observations,
                             int32_t min_camera_observations,
                             int32_t max_points,
                             double loss_scale,
                             bool retriangulate_seeds,
                             bool use_stereo_rows,
                             StereoBaProblem* problem) {
    // Ascending frame order matches the append order of the archive, so the
    // per-point index lists come out identical to a full archive scan.
    std::vector<int32_t> window_frames;
    window_frames.reserve(frame_row.size());
    for (const auto& entry : frame_row) {
        window_frames.push_back(entry.first);
    }
    std::sort(window_frames.begin(), window_frames.end());
    std::unordered_map<int32_t, std::vector<std::size_t>> groups;
    for (const int32_t frame_id : window_frames) {
        const auto frame_it = archive.observations_by_frame.find(frame_id);
        if (frame_it == archive.observations_by_frame.end()) {
            continue;
        }
        for (const int32_t i : frame_it->second) {
            groups[archive.observations[static_cast<std::size_t>(i)]
                       .point_id]
                .push_back(static_cast<std::size_t>(i));
        }
    }
    std::vector<std::pair<int32_t, int32_t>> eligible;
    eligible.reserve(groups.size());
    for (const auto& entry : groups) {
        const int32_t n = static_cast<int32_t>(entry.second.size());
        if (n >= min_observations &&
            archive.positions.find(entry.first) != archive.positions.end()) {
            eligible.push_back({entry.first, n});
        }
    }
    std::sort(eligible.begin(), eligible.end(),
              [](const std::pair<int32_t, int32_t>& a,
                 const std::pair<int32_t, int32_t>& b) {
                  return a.second != b.second ? a.second > b.second
                                              : a.first < b.first;
              });
    if (static_cast<int32_t>(eligible.size()) > max_points) {
        eligible.resize(static_cast<std::size_t>(max_points));
    }

    // Candidate observations per point, gated on depth and seed residual.
    struct CandidateObservation {
        int32_t row = 0;
        double u = 0.0;
        double v = 0.0;
        double right_x = 0.0;
        bool has_right = false;
    };
    struct CandidatePoint {
        int32_t id = 0;
        cv::Point3f position;
        std::vector<CandidateObservation> observations;
    };
    std::unordered_map<int64_t, double> right_lookup;
    for (const int32_t frame_id : window_frames) {
        const auto frame_it =
            archive.stereo_observations_by_frame.find(frame_id);
        if (frame_it == archive.stereo_observations_by_frame.end()) {
            continue;
        }
        for (const int32_t i : frame_it->second) {
            const StereoObservation& stereo_obs =
                archive.stereo_observations[static_cast<std::size_t>(i)];
            right_lookup[stereo_lookup_key(stereo_obs.point_id,
                                           stereo_obs.frame_id)] =
                static_cast<double>(stereo_obs.right_x);
        }
    }
    std::vector<CandidatePoint> candidates;
    const double seed_gate = 3.0 * loss_scale;
    for (const std::pair<int32_t, int32_t>& entry : eligible) {
        CandidatePoint candidate;
        candidate.id = entry.first;
        candidate.position = archive.positions.at(entry.first);
        // Archived positions of long-dead tracks carry the gauge the
        // trajectory had when the point was last seen. For the end-of-run
        // full BA, re-triangulating from the widest observation pair seeds
        // the point consistently with the current poses; the archive
        // position stays as the fallback because it holds the metric depth
        // from the stereo baseline.
        if (retriangulate_seeds) {
            std::vector<std::pair<int32_t, cv::Point2f>> views;
            views.reserve(groups[entry.first].size());
            for (const std::size_t obs_idx : groups[entry.first]) {
                const MapObservation& obs = archive.observations[obs_idx];
                views.push_back({frame_row.at(obs.frame_id), obs.pixel});
            }
            std::sort(views.begin(), views.end(),
                      [](const std::pair<int32_t, cv::Point2f>& a,
                         const std::pair<int32_t, cv::Point2f>& b) {
                          return a.first < b.first;
                      });
            if (views.size() >= 2U &&
                views.front().first != views.back().first) {
                const std::array<std::pair<std::size_t, std::size_t>, 2>
                    pairs = {{{0U, views.size() - 1U}, {0U, 1U}}};
                for (const std::pair<std::size_t, std::size_t>& pair :
                     pairs) {
                    const Pose& pose_a = row_poses[static_cast<std::size_t>(
                        views[pair.first].first)];
                    const Pose& pose_b = row_poses[static_cast<std::size_t>(
                        views[pair.second].first)];
                    cv::Point3f seed;
                    if (!triangulate_seed_pair(
                            pose_a, pose_b, views[pair.first].second,
                            views[pair.second].second, camera, &seed)) {
                        continue;
                    }
                    const double residual_a = reprojection_residual(
                        seed, views[pair.first].second, pose_a, camera);
                    const double residual_b = reprojection_residual(
                        seed, views[pair.second].second, pose_b, camera);
                    if (depth_in_pose(seed, pose_a) > 1.0e-6 &&
                        depth_in_pose(seed, pose_b) > 1.0e-6 &&
                        std::isfinite(residual_a) &&
                        std::isfinite(residual_b) &&
                        residual_a <= seed_gate &&
                        residual_b <= seed_gate) {
                        candidate.position = seed;
                        break;
                    }
                }
            }
        }
        for (const std::size_t obs_idx : groups[entry.first]) {
            const MapObservation& obs = archive.observations[obs_idx];
            const int32_t row = frame_row.at(obs.frame_id);
            const Pose& pose = row_poses[static_cast<std::size_t>(row)];
            const double residual = reprojection_residual(
                candidate.position, obs.pixel, pose, camera);
            if (depth_in_pose(candidate.position, pose) > 1.0e-6 &&
                std::isfinite(residual) && residual <= seed_gate) {
                CandidateObservation cand_obs;
                cand_obs.row = row;
                cand_obs.u = static_cast<double>(obs.pixel.x);
                cand_obs.v = static_cast<double>(obs.pixel.y);
                if (use_stereo_rows) {
                    const auto right_it = right_lookup.find(
                        stereo_lookup_key(entry.first, obs.frame_id));
                    if (right_it != right_lookup.end()) {
                        cand_obs.right_x = right_it->second;
                        cand_obs.has_right = true;
                    }
                }
                candidate.observations.push_back(cand_obs);
            }
        }
        if (static_cast<int32_t>(candidate.observations.size()) >=
            min_observations) {
            candidates.push_back(std::move(candidate));
        }
    }

    // Iteratively drop under-constrained cameras; removing a camera can
    // starve a point below min_observations, which can weaken another
    // camera, so repeat until the support is stable.
    std::vector<char> row_active(row_poses.size(), 1);
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<int32_t> row_support(row_poses.size(), 0);
        for (const CandidatePoint& candidate : candidates) {
            if (static_cast<int32_t>(candidate.observations.size()) <
                min_observations) {
                continue;
            }
            for (const CandidateObservation& obs :
                 candidate.observations) {
                ++row_support[static_cast<std::size_t>(obs.row)];
            }
        }
        for (std::size_t row = 0; row < row_poses.size(); ++row) {
            if (row_active[row] != 0 &&
                row_support[row] < min_camera_observations) {
                row_active[row] = 0;
                changed = true;
                for (CandidatePoint& candidate : candidates) {
                    std::vector<CandidateObservation> kept;
                    kept.reserve(candidate.observations.size());
                    for (const CandidateObservation& obs :
                         candidate.observations) {
                        if (obs.row != static_cast<int32_t>(row)) {
                            kept.push_back(obs);
                        }
                    }
                    candidate.observations = std::move(kept);
                }
            }
        }
    }

    std::vector<int32_t> compact_index(row_poses.size(), -1);
    for (std::size_t row = 0; row < row_poses.size(); ++row) {
        if (row_active[row] != 0) {
            compact_index[row] =
                static_cast<int32_t>(problem->camera_rows.size());
            problem->camera_rows.push_back(static_cast<int32_t>(row));
        }
    }
    for (CandidatePoint& candidate : candidates) {
        if (static_cast<int32_t>(candidate.observations.size()) <
            min_observations) {
            continue;
        }
        const int32_t point_row =
            static_cast<int32_t>(problem->point_positions.size());
        problem->point_ids.push_back(candidate.id);
        problem->point_positions.push_back(candidate.position);
        for (const CandidateObservation& obs : candidate.observations) {
            const double cam_idx = static_cast<double>(
                compact_index[static_cast<std::size_t>(obs.row)]);
            if (obs.has_right) {
                problem->stereo_observation_rows.push_back(
                    {cam_idx, static_cast<double>(point_row), obs.u, obs.v,
                     obs.right_x});
            } else {
                problem->observation_rows.push_back(
                    {cam_idx, static_cast<double>(point_row), obs.u,
                     obs.v});
            }
        }
    }
    return problem->camera_rows.size() >= 2U &&
           !problem->point_positions.empty() &&
           problem->observation_rows.size() +
                   problem->stereo_observation_rows.size() >= 2U;
}

/*
Runs the cvlib Schur BA and, on acceptance, re-anchors the gauge to the first
row pose: T'' = T' * D with D = T_first'^-1 * T_first and points moved by
D^-1 so every reprojection stays invariant. Results whose rotations left the
SE(3) manifold (orthonormality or determinant drift beyond
max_rotation_error) are rejected: publishing a collapsed rotation poisons
tracking, the stereo map, and every later re-initialization at once.
*/

bool solve_stereo_ba(const CameraIntrinsics& camera,
                     const std::vector<Pose>& row_poses,
                     double baseline,
                     int32_t jacobian_mode,
                     int32_t max_iterations,
                     double loss_scale,
                     double max_rotation_error,
                     StereoBaProblem* problem,
                     std::vector<Pose>* optimized_poses,
                     cvlib::optimize::OptimizeReport* report,
                     int32_t* status) {
    const int32_t cam_count = static_cast<int32_t>(row_poses.size());
    const int32_t point_count =
        static_cast<int32_t>(problem->point_positions.size());
    const int32_t observation_count =
        static_cast<int32_t>(problem->observation_rows.size());
    const int32_t stereo_count =
        static_cast<int32_t>(problem->stereo_observation_rows.size());

    cvlib::Matrix k = make_camera_matrix(camera);
    cvlib::Matrix poses = cvlib::matrix_create(cam_count, 12);
    cvlib::Matrix points = cvlib::matrix_create(point_count, 3);
    cvlib::Matrix observations = cvlib::matrix_create(observation_count, 4);
    cvlib::Matrix stereo_observations = {};
    if (stereo_count > 0) {
        stereo_observations = cvlib::matrix_create(stereo_count, 5);
        for (int32_t o = 0; o < stereo_count; ++o) {
            for (int32_t c = 0; c < 5; ++c) {
                cvlib::matrix_set(
                    &stereo_observations, o, c,
                    problem->stereo_observation_rows[
                        static_cast<std::size_t>(o)][
                        static_cast<std::size_t>(c)]);
            }
        }
    }
    for (int32_t c = 0; c < cam_count; ++c) {
        const Pose& pose = row_poses[static_cast<std::size_t>(c)];
        for (int32_t i = 0; i < 9; ++i) {
            cvlib::matrix_set(&poses, c, i, pose.r[i]);
        }
        for (int32_t i = 0; i < 3; ++i) {
            cvlib::matrix_set(&poses, c, 9 + i, pose.t[i]);
        }
    }
    for (int32_t p = 0; p < point_count; ++p) {
        const cv::Point3f& point =
            problem->point_positions[static_cast<std::size_t>(p)];
        cvlib::matrix_set(&points, p, 0, static_cast<double>(point.x));
        cvlib::matrix_set(&points, p, 1, static_cast<double>(point.y));
        cvlib::matrix_set(&points, p, 2, static_cast<double>(point.z));
    }
    for (int32_t o = 0; o < observation_count; ++o) {
        for (int32_t c = 0; c < 4; ++c) {
            cvlib::matrix_set(
                &observations, o, c,
                problem->observation_rows[static_cast<std::size_t>(o)][
                    static_cast<std::size_t>(c)]);
        }
    }

    cvlib::calib3d::BAOptions options =
        cvlib::calib3d::default_ba_options();
    options.solver = cvlib::calib3d::kBASolverSchur;
    options.jacobian_mode = jacobian_mode;
    options.lm.max_iter = max_iterations;
    options.lm.loss.type = cvlib::optimize::kLossHuber;
    options.lm.loss.scale = loss_scale;
    cvlib::calib3d::BAData data = {&poses, &points, &observations, &k,
                                   nullptr};
    if (stereo_count > 0) {
        data.stereo_observations = &stereo_observations;
        data.stereo_baseline = baseline;
    }
    const cvlib::ErrorCode ec =
        cvlib::calib3d::bundle_adjustment(&data, &options, report);
    *status = static_cast<int32_t>(ec);

    bool accepted = ec == cvlib::ErrorCode::kSuccess &&
                    report->final_cost <= report->initial_cost;
    if (accepted) {
        for (int32_t c = 0; c < cam_count && accepted; ++c) {
            const Pose optimized = param_row_to_pose(poses, c);
            const double ortho_error =
                rotation_orthonormality_error(optimized);
            const double det_error =
                std::abs(rotation_determinant(optimized) - 1.0);
            if (!(ortho_error <= max_rotation_error &&
                  det_error <= max_rotation_error)) {
                accepted = false;
                std::cout << "stereo_ba_rejected reason=degenerate_rotation"
                          << " cam=" << c
                          << " ortho_error=" << ortho_error
                          << " det_error=" << det_error << std::endl;
            }
        }
    }
    if (accepted) {
        const Pose first_before = row_poses[0];
        const Pose first_after = param_row_to_pose(poses, 0);
        const Pose anchor = compose_reference_relative_pose(
            first_before, invert_pose(first_after));
        optimized_poses->clear();
        optimized_poses->reserve(static_cast<std::size_t>(cam_count));
        for (int32_t c = 0; c < cam_count; ++c) {
            Pose anchored = compose_reference_relative_pose(
                anchor, param_row_to_pose(poses, c));
            orthonormalize_rotation(&anchored);
            optimized_poses->push_back(anchored);
        }
        for (int32_t p = 0; p < point_count; ++p) {
            const cv::Point3f optimized(
                static_cast<float>(cvlib::matrix_get(&points, p, 0)),
                static_cast<float>(cvlib::matrix_get(&points, p, 1)),
                static_cast<float>(cvlib::matrix_get(&points, p, 2)));
            problem->point_positions[static_cast<std::size_t>(p)] =
                apply_pose_inverse(anchor, optimized);
        }
    }

    cvlib::matrix_destroy(&k);
    cvlib::matrix_destroy(&poses);
    cvlib::matrix_destroy(&points);
    cvlib::matrix_destroy(&observations);
    if (stereo_count > 0) {
        cvlib::matrix_destroy(&stereo_observations);
    }
    return accepted;
}

}  // namespace

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
    MapArchive* archive) {
    int32_t added = 0;
    const StereoParameters& stereo = parameters.stereo;
    const std::size_t aligned_count = std::min(left_points.size(),
                                               map_points->size());
    std::vector<cv::Point2f> candidates;
    std::vector<std::size_t> owners;
    for (std::size_t i = 0; i < aligned_count; ++i) {
        if (!(*map_points)[i].has_position) {
            candidates.push_back(left_points[i]);
            owners.push_back(i);
        }
    }
    if (candidates.empty()) {
        return added;
    }

    std::vector<cv::Point2f> matched_left;
    std::vector<cv::Point2f> matched_right;
    std::vector<int32_t> matched_indices;
    track_points(left, right, candidates, parameters.feature,
                 &matched_left, &matched_right, &matched_indices,
                 debug_geometry, "stereo_" + std::to_string(frame_id),
                 true);

    int32_t rejected_epipolar = 0;
    int32_t rejected_depth = 0;
    const std::size_t match_count = std::min(matched_right.size(),
                                             matched_indices.size());
    for (std::size_t j = 0; j < match_count; ++j) {
        const cv::Point2f& pixel_left = matched_left[j];
        const cv::Point2f& pixel_right = matched_right[j];
        const double dy = std::abs(static_cast<double>(pixel_left.y) -
                                   static_cast<double>(pixel_right.y));
        const double disparity = static_cast<double>(pixel_left.x) -
                                 static_cast<double>(pixel_right.x);
        if (dy > stereo.max_epipolar_error ||
            disparity < stereo.min_disparity ||
            disparity > stereo.max_disparity) {
            ++rejected_epipolar;
            continue;
        }
        const double z = camera.fx * baseline / disparity;
        if (z < stereo.min_depth || z > stereo.max_depth) {
            ++rejected_depth;
            continue;
        }
        const cv::Point3f camera_point(
            static_cast<float>((static_cast<double>(pixel_left.x) -
                                camera.cx) * z / camera.fx),
            static_cast<float>((static_cast<double>(pixel_left.y) -
                                camera.cy) * z / camera.fy),
            static_cast<float>(z));
        const cv::Point3f world_point = apply_pose_inverse(pose,
                                                           camera_point);
        MapPoint& point = (*map_points)[owners[
            static_cast<std::size_t>(matched_indices[j])]];
        point.position = world_point;
        point.has_position = true;
        point.has_anchor = false;
        point.last_seen_frame = frame_id;
        point.last_reprojection_error = 0.0;
        // The right-image match is a second view, so the point starts with
        // the same track support a mono two-view triangulation would have.
        point.track_length = std::max(point.track_length + 1, 2);
        point.candidate = point.track_length <
                          parameters.mapping.candidate_min_track_length;
        all_map_points->push_back(world_point);
        if (archive != nullptr && point.id >= 0) {
            archive_observation(archive, {frame_id, point.id, pixel_left});
            archive_stereo_observation(
                archive, {frame_id, point.id, pixel_left,
                          static_cast<float>(pixel_right.x)});
            archive->positions[point.id] = world_point;
            archive->last_seen[point.id] = frame_id;
        }
        ++added;
    }
    if (debug_geometry) {
        std::cout << "stereo_triangulation frame=" << frame_id
                  << " candidates=" << candidates.size()
                  << " matched=" << match_count
                  << " added=" << added
                  << " rejected_epipolar=" << rejected_epipolar
                  << " rejected_depth=" << rejected_depth << std::endl;
    }
    return added;
}

bool initialize_stereo_tracking(const cv::Mat& left,
                                const cv::Mat& right,
                                const Pose& initial_pose,
                                const CameraIntrinsics& camera,
                                double baseline,
                                const MvoParameters& parameters,
                                int32_t frame_id,
                                bool debug_geometry,
                                MapArchive* archive,
                                TrackState* state) {
    bool ok = false;
    std::vector<cv::Point2f> initial_points;
    std::vector<MapPoint> map_points;
    std::vector<cv::Point3f> new_map_points;
    int32_t added = 0;
    if (detect_initial_points(left, parameters.feature, &initial_points)) {
        map_points.reserve(initial_points.size());
        for (const cv::Point2f& point : initial_points) {
            map_points.push_back(make_pending_map_point(
                point, initial_pose, frame_id, parameters.mapping));
        }
        added = triangulate_stereo_map_points(
            left, right, initial_points, initial_pose, camera, baseline,
            parameters, frame_id, debug_geometry, &map_points,
            &new_map_points, archive);
        ok = added >= parameters.stereo.min_init_points;
    }
    if (ok) {
        state->prev_image = left;
        state->prev_pose = initial_pose;
        state->last_pose = initial_pose;
        state->prev_points = initial_points;
        state->map_points = map_points;
        state->all_map_points.insert(state->all_map_points.end(),
                                     new_map_points.begin(),
                                     new_map_points.end());
        ++state->keyframes;
    }
    std::cout << "stereo_init frame=" << frame_id
              << " features=" << initial_points.size()
              << " map_points=" << added
              << " min_required=" << parameters.stereo.min_init_points
              << " ok=" << ok << std::endl;
    return ok;
}

bool run_stereo_local_ba(const CameraIntrinsics& camera,
                         double baseline,
                         const StereoParameters& parameters,
                         int32_t frame_id,
                         bool debug_geometry,
                         MapArchive* archive,
                         StereoBackend* backend,
                         TrackState* state) {
    bool accepted = false;
    const int32_t total = static_cast<int32_t>(backend->trajectory.size());
    const int32_t window = std::min(parameters.local_ba_window, total);
    if (window < 2) {
        return accepted;
    }
    const int32_t base = total - window;
    std::unordered_map<int32_t, int32_t> frame_row;
    std::vector<Pose> row_poses;
    frame_row.reserve(static_cast<std::size_t>(window));
    row_poses.reserve(static_cast<std::size_t>(window));
    for (int32_t r = 0; r < window; ++r) {
        const StereoFramePose& entry =
            backend->trajectory[static_cast<std::size_t>(base + r)];
        frame_row[entry.frame_id] = r;
        row_poses.push_back(entry.pose);
    }

    StereoBaProblem problem;
    if (!build_stereo_ba_problem(*archive, frame_row, row_poses, camera,
                                 parameters.local_ba_min_observations,
                                 parameters.local_ba_min_camera_observations,
                                 parameters.local_ba_max_points,
                                 parameters.local_ba_loss_scale, false,
                                 parameters.local_ba_stereo_rows != 0,
                                 &problem)) {
        if (debug_geometry) {
            std::cout << "stereo_local_ba_skipped frame=" << frame_id
                      << " cams=" << problem.camera_rows.size() << "/"
                      << window
                      << " points=" << problem.point_positions.size()
                      << " observations="
                      << problem.observation_rows.size() << std::endl;
        }
        return accepted;
    }

    std::vector<Pose> active_poses;
    active_poses.reserve(problem.camera_rows.size());
    for (const int32_t row : problem.camera_rows) {
        active_poses.push_back(row_poses[static_cast<std::size_t>(row)]);
    }
    std::vector<Pose> optimized;
    cvlib::optimize::OptimizeReport report = {};
    int32_t status = 0;
    accepted = solve_stereo_ba(camera, active_poses, baseline,
                               parameters.ba_jacobian_mode,
                               parameters.local_ba_max_iterations,
                               parameters.local_ba_loss_scale,
                               parameters.max_rotation_error, &problem,
                               &optimized, &report, &status);
    if (accepted) {
        for (std::size_t c = 0; c < problem.camera_rows.size(); ++c) {
            const int32_t row = problem.camera_rows[c];
            backend->trajectory[static_cast<std::size_t>(base + row)].pose =
                optimized[c];
            // Tracking continues from the refined current-frame estimate.
            if (row == window - 1) {
                state->last_pose = optimized[c];
                state->prev_pose = state->last_pose;
            }
        }
        std::unordered_map<int32_t, cv::Point3f> refined;
        refined.reserve(problem.point_ids.size());
        for (std::size_t p = 0; p < problem.point_ids.size(); ++p) {
            refined[problem.point_ids[p]] = problem.point_positions[p];
            archive->positions[problem.point_ids[p]] =
                problem.point_positions[p];
        }
        for (MapPoint& point : state->map_points) {
            if (point.has_position) {
                const auto it = refined.find(point.id);
                if (it != refined.end()) {
                    point.position = it->second;
                }
            }
        }
        ++backend->local_ba_accepted;
    }
    ++backend->local_ba_runs;
    std::cout << "stereo_local_ba frame=" << frame_id
              << " cams=" << problem.camera_rows.size() << "/" << window
              << " points=" << problem.point_positions.size()
              << " observations=" << problem.observation_rows.size()
              << " stereo_observations="
              << problem.stereo_observation_rows.size()
              << " status=" << status
              << " accepted=" << accepted
              << " cost=" << report.initial_cost << "->"
              << report.final_cost
              << " iterations=" << report.iterations
              << " term=" << report.termination << std::endl;
    return accepted;
}

bool run_stereo_full_ba(const CameraIntrinsics& camera,
                        double baseline,
                        const StereoParameters& parameters,
                        bool debug_geometry,
                        const MapArchive& archive,
                        StereoBackend* backend) {
    bool accepted = false;
    backend->optimized = backend->trajectory;
    const int32_t total = static_cast<int32_t>(backend->trajectory.size());
    if (total < 2) {
        return accepted;
    }

    std::set<int32_t> cam_set;
    cam_set.insert(0);
    cam_set.insert(total - 1);
    const int32_t stride = std::max(
        1, (total + parameters.full_ba_max_cameras - 1) /
               parameters.full_ba_max_cameras);
    for (int32_t i = 0; i < total; i += stride) {
        cam_set.insert(i);
    }
    const std::vector<int32_t> cams(cam_set.begin(), cam_set.end());
    const int32_t cam_count = static_cast<int32_t>(cams.size());
    std::unordered_map<int32_t, int32_t> frame_row;
    std::vector<Pose> row_poses;
    frame_row.reserve(cams.size());
    row_poses.reserve(cams.size());
    for (int32_t c = 0; c < cam_count; ++c) {
        const StereoFramePose& entry =
            backend->trajectory[static_cast<std::size_t>(
                cams[static_cast<std::size_t>(c)])];
        frame_row[entry.frame_id] = c;
        row_poses.push_back(entry.pose);
    }

    StereoBaProblem problem;
    cvlib::optimize::OptimizeReport report = {};
    int32_t status = 0;
    std::vector<Pose> optimized;
    if (build_stereo_ba_problem(archive, frame_row, row_poses, camera,
                                parameters.full_ba_min_observations,
                                parameters.full_ba_min_camera_observations,
                                parameters.full_ba_max_points,
                                parameters.full_ba_loss_scale, true,
                                parameters.full_ba_stereo_rows != 0,
                                &problem)) {
        std::vector<Pose> active_poses;
        active_poses.reserve(problem.camera_rows.size());
        for (const int32_t row : problem.camera_rows) {
            active_poses.push_back(
                row_poses[static_cast<std::size_t>(row)]);
        }
        accepted = solve_stereo_ba(camera, active_poses, baseline,
                                   parameters.ba_jacobian_mode,
                                   parameters.full_ba_max_iterations,
                                   parameters.full_ba_loss_scale,
                                   parameters.max_rotation_error, &problem,
                                   &optimized, &report, &status);
    } else if (debug_geometry) {
        std::cout << "stereo_full_ba_skipped cams="
                  << problem.camera_rows.size() << "/" << cam_count
                  << " points=" << problem.point_positions.size()
                  << " observations=" << problem.observation_rows.size()
                  << std::endl;
    }

    if (accepted) {
        // Transport unselected frames on the relative chain of the nearest
        // optimized camera at or before them: T'' = (T * T_s^-1) * T_s''.
        std::vector<int32_t> anchor_traj_indices;
        anchor_traj_indices.reserve(problem.camera_rows.size());
        for (const int32_t row : problem.camera_rows) {
            anchor_traj_indices.push_back(
                cams[static_cast<std::size_t>(row)]);
        }
        std::size_t selected = 0;
        for (int32_t i = 0; i < total; ++i) {
            while (selected + 1U < anchor_traj_indices.size() &&
                   anchor_traj_indices[selected + 1U] <= i) {
                ++selected;
            }
            const Pose& anchor_before =
                backend->trajectory[static_cast<std::size_t>(
                    anchor_traj_indices[selected])].pose;
            const Pose& anchor_after = optimized[selected];
            const Pose relative = compose_reference_relative_pose(
                invert_pose(anchor_before),
                backend->trajectory[static_cast<std::size_t>(i)].pose);
            backend->optimized[static_cast<std::size_t>(i)].pose =
                compose_reference_relative_pose(anchor_after, relative);
        }
        backend->full_ba_accepted = true;
    }
    std::cout << "stereo_full_ba cams=" << problem.camera_rows.size()
              << "/" << cam_count
              << " points=" << problem.point_positions.size()
              << " observations=" << problem.observation_rows.size()
              << " stereo_observations="
              << problem.stereo_observation_rows.size()
              << " status=" << status
              << " accepted=" << accepted
              << " cost=" << report.initial_cost << "->"
              << report.final_cost
              << " iterations=" << report.iterations
              << " term=" << report.termination << std::endl;
    return accepted;
}

}  // namespace mvo
