#include "converter.h"

#include "constants.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mvo {

cvlib::Matrix make_camera_matrix(const CameraIntrinsics& camera) {
    cvlib::Matrix k = cvlib::matrix_create(3, 3);
    cvlib::matrix_set(&k, 0, 0, camera.fx);
    cvlib::matrix_set(&k, 0, 1, 0.0);
    cvlib::matrix_set(&k, 0, 2, camera.cx);
    cvlib::matrix_set(&k, 1, 0, 0.0);
    cvlib::matrix_set(&k, 1, 1, camera.fy);
    cvlib::matrix_set(&k, 1, 2, camera.cy);
    cvlib::matrix_set(&k, 2, 0, 0.0);
    cvlib::matrix_set(&k, 2, 1, 0.0);
    cvlib::matrix_set(&k, 2, 2, 1.0);
    return k;
}

cvlib::Matrix points2f_to_matrix(const std::vector<cv::Point2f>& points) {
    cvlib::Matrix mat = cvlib::matrix_create(
        static_cast<int32_t>(points.size()), 2);
    for (int32_t i = 0; i < static_cast<int32_t>(points.size()); ++i) {
        cvlib::matrix_set(&mat, i, 0, static_cast<double>(points[i].x));
        cvlib::matrix_set(&mat, i, 1, static_cast<double>(points[i].y));
    }
    return mat;
}

cvlib::Matrix points2f_to_normalized_matrix(
    const std::vector<cv::Point2f>& points,
    const CameraIntrinsics& camera) {
    cvlib::Matrix mat = cvlib::matrix_create(
        static_cast<int32_t>(points.size()), 2);
    for (int32_t i = 0; i < static_cast<int32_t>(points.size()); ++i) {
        const double x = (static_cast<double>(points[i].x) - camera.cx) /
                         camera.fx;
        const double y = (static_cast<double>(points[i].y) - camera.cy) /
                         camera.fy;
        cvlib::matrix_set(&mat, i, 0, x);
        cvlib::matrix_set(&mat, i, 1, y);
    }
    return mat;
}

cvlib::Matrix points3f_to_matrix(const std::vector<cv::Point3f>& points) {
    cvlib::Matrix mat = cvlib::matrix_create(
        static_cast<int32_t>(points.size()), 3);
    for (int32_t i = 0; i < static_cast<int32_t>(points.size()); ++i) {
        cvlib::matrix_set(&mat, i, 0, static_cast<double>(points[i].x));
        cvlib::matrix_set(&mat, i, 1, static_cast<double>(points[i].y));
        cvlib::matrix_set(&mat, i, 2, static_cast<double>(points[i].z));
    }
    return mat;
}

cvlib::Matrix pose_to_projection(const Pose& pose) {
    cvlib::Matrix p = cvlib::matrix_create(3, 4);
    for (int32_t r = 0; r < 3; ++r) {
        for (int32_t c = 0; c < 3; ++c) {
            cvlib::matrix_set(&p, r, c, pose.r[r * 3 + c]);
        }
        cvlib::matrix_set(&p, r, 3, pose.t[r]);
    }
    return p;
}

void copy_cvlib_pose(const cvlib::Matrix* r, const cvlib::Vector* t,
                     Pose* pose) {
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            pose->r[row * 3 + col] = cvlib::matrix_get(r, row, col);
        }
        pose->t[row] = t->data[row];
    }
}

cvlib::Matrix pose_rotation_to_matrix(const Pose& pose) {
    cvlib::Matrix r = cvlib::matrix_create(3, 3);
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            cvlib::matrix_set(&r, row, col, pose.r[row * 3 + col]);
        }
    }
    return r;
}

cvlib::Vector pose_translation_to_vector(const Pose& pose) {
    cvlib::Vector t = cvlib::vector_create(3);
    for (int32_t i = 0; i < 3; ++i) {
        t.data[i] = pose.t[i];
    }
    return t;
}

double depth_in_pose(const cv::Point3f& point, const Pose& pose) {
    const double depth = pose.r[6] * static_cast<double>(point.x) +
                         pose.r[7] * static_cast<double>(point.y) +
                         pose.r[8] * static_cast<double>(point.z) +
                         pose.t[2];
    return depth;
}

namespace {

bool parallax_deg_between_centers(const cv::Point3f& point,
                                  const cv::Point3f& c0,
                                  const cv::Point3f& c1,
                                  double* angle_deg) {
    bool ok = false;
    const double v0x = static_cast<double>(point.x - c0.x);
    const double v0y = static_cast<double>(point.y - c0.y);
    const double v0z = static_cast<double>(point.z - c0.z);
    const double v1x = static_cast<double>(point.x - c1.x);
    const double v1y = static_cast<double>(point.y - c1.y);
    const double v1z = static_cast<double>(point.z - c1.z);
    const double n0 = std::sqrt(v0x * v0x + v0y * v0y + v0z * v0z);
    const double n1 = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    if (n0 > 1.0e-9 && n1 > 1.0e-9) {
        double cos_angle = (v0x * v1x + v0y * v1y + v0z * v1z) / (n0 * n1);
        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
        *angle_deg = std::acos(cos_angle) * 180.0 / kPi;
        ok = true;
    }
    return ok;
}

}  // namespace

cv::Point3f camera_center_from_pose(const Pose& pose) {
    const float x = static_cast<float>(
        -(pose.r[0] * pose.t[0] + pose.r[3] * pose.t[1] +
          pose.r[6] * pose.t[2]));
    const float y = static_cast<float>(
        -(pose.r[1] * pose.t[0] + pose.r[4] * pose.t[1] +
          pose.r[7] * pose.t[2]));
    const float z = static_cast<float>(
        -(pose.r[2] * pose.t[0] + pose.r[5] * pose.t[1] +
          pose.r[8] * pose.t[2]));
    return cv::Point3f(x, y, z);
}

double median_parallax_deg(const std::vector<cv::Point3f>& map_points,
                           const Pose& pose0,
                           const Pose& pose1) {
    double median = 0.0;
    std::vector<double> angles;
    const cv::Point3f c0 = camera_center_from_pose(pose0);
    const cv::Point3f c1 = camera_center_from_pose(pose1);
    angles.reserve(map_points.size());
    for (const cv::Point3f& point : map_points) {
        double angle_deg = 0.0;
        if (parallax_deg_between_centers(point, c0, c1, &angle_deg)) {
            angles.push_back(angle_deg);
        }
    }
    if (!angles.empty()) {
        std::sort(angles.begin(), angles.end());
        median = angles[angles.size() / 2];
    }
    return median;
}

double parallax_deg_for_point(const cv::Point3f& point,
                              const Pose& pose0,
                              const Pose& pose1) {
    double angle_deg = 0.0;
    const cv::Point3f c0 = camera_center_from_pose(pose0);
    const cv::Point3f c1 = camera_center_from_pose(pose1);
    parallax_deg_between_centers(point, c0, c1, &angle_deg);
    return angle_deg;
}

double rotation_orthonormality_error(const Pose& pose) {
    double max_error = 0.0;
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            double value = 0.0;
            for (int32_t k = 0; k < 3; ++k) {
                value += pose.r[k * 3 + row] * pose.r[k * 3 + col];
            }
            const double target = row == col ? 1.0 : 0.0;
            max_error = std::max(max_error, std::abs(value - target));
        }
    }
    return max_error;
}

double rotation_determinant(const Pose& pose) {
    return pose.r[0] * (pose.r[4] * pose.r[8] - pose.r[5] * pose.r[7]) -
           pose.r[1] * (pose.r[3] * pose.r[8] - pose.r[5] * pose.r[6]) +
           pose.r[2] * (pose.r[3] * pose.r[7] - pose.r[4] * pose.r[6]);
}

double reprojection_residual(const cv::Point3f& point,
                             const cv::Point2f& observation,
                             const Pose& pose,
                             const CameraIntrinsics& camera) {
    double residual = std::numeric_limits<double>::infinity();
    const double x = pose.r[0] * static_cast<double>(point.x) +
                     pose.r[1] * static_cast<double>(point.y) +
                     pose.r[2] * static_cast<double>(point.z) + pose.t[0];
    const double y = pose.r[3] * static_cast<double>(point.x) +
                     pose.r[4] * static_cast<double>(point.y) +
                     pose.r[5] * static_cast<double>(point.z) + pose.t[1];
    const double z = depth_in_pose(point, pose);
    if (z > 1.0e-9) {
        const double u = camera.fx * x / z + camera.cx;
        const double v = camera.fy * y / z + camera.cy;
        const double du = u - static_cast<double>(observation.x);
        const double dv = v - static_cast<double>(observation.y);
        residual = std::sqrt(du * du + dv * dv);
    }
    return residual;
}

ReprojectionStats compute_reprojection_stats(
    const std::vector<cv::Point3f>& map_points,
    const std::vector<cv::Point2f>& image_points,
    const Pose& pose,
    const CameraIntrinsics& camera) {
    ReprojectionStats stats;
    std::vector<double> residuals;
    const std::size_t n = std::min(map_points.size(), image_points.size());
    residuals.reserve(n);
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double residual = reprojection_residual(
            map_points[i], image_points[i], pose, camera);
        if (std::isfinite(residual)) {
            residuals.push_back(residual);
            sum_sq += residual * residual;
        }
    }
    if (!residuals.empty()) {
        std::sort(residuals.begin(), residuals.end());
        stats.valid = static_cast<int32_t>(residuals.size());
        stats.rmse = std::sqrt(sum_sq / static_cast<double>(residuals.size()));
        stats.median = residuals[residuals.size() / 2];
        stats.p90 = residuals[static_cast<std::size_t>(
            std::min(static_cast<int32_t>(residuals.size()) - 1,
                     static_cast<int32_t>(residuals.size() * 9 / 10)))];
        stats.max = residuals.back();
    }
    return stats;
}

void filter_by_reprojection(
    const std::vector<cv::Point3f>& map_points,
    const std::vector<cv::Point2f>& image_points,
    const Pose& pose,
    const CameraIntrinsics& camera,
    double threshold,
    std::vector<cv::Point3f>* filtered_map_points,
    std::vector<cv::Point2f>* filtered_image_points) {
    filtered_map_points->clear();
    filtered_image_points->clear();
    const std::size_t n = std::min(map_points.size(), image_points.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double residual = reprojection_residual(
            map_points[i], image_points[i], pose, camera);
        if (std::isfinite(residual) && residual <= threshold) {
            filtered_map_points->push_back(map_points[i]);
            filtered_image_points->push_back(image_points[i]);
        }
    }
}

std::vector<cv::Mat> descriptor_rows(const cv::Mat& descriptors) {
    std::vector<cv::Mat> rows;
    if (!descriptors.empty()) {
        rows.reserve(static_cast<std::size_t>(descriptors.rows));
        for (int32_t i = 0; i < descriptors.rows; ++i) {
            rows.push_back(descriptors.row(i));
        }
    }
    return rows;
}

}  // namespace mvo
