#include "visualization.h"

#include "converter.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace mvo {
namespace {

bool visualizer_requested(const AppConfig& config) {
    return config.rerun_spawn || !config.rerun_save_path.empty();
}

#if MVO_HAS_RERUN
struct Point3fHash {
    std::size_t operator()(const cv::Point3f& point) const {
        const std::hash<float> hasher;
        std::size_t seed = hasher(point.x + 0.0F);
        seed ^= hasher(point.y + 0.0F) + 0x9e3779b9U + (seed << 6) +
                (seed >> 2);
        seed ^= hasher(point.z + 0.0F) + 0x9e3779b9U + (seed << 6) +
                (seed >> 2);
        return seed;
    }
};

struct Point3fEqual {
    bool operator()(const cv::Point3f& a, const cv::Point3f& b) const {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
};

void collect_previous_map_points(
    const std::vector<cv::Point3f>& all_map_points,
    const std::vector<cv::Point3f>& current_map_points,
    std::vector<cv::Point3f>* previous_map_points) {
    previous_map_points->clear();
    previous_map_points->reserve(all_map_points.size());
    const std::unordered_set<cv::Point3f, Point3fHash, Point3fEqual>
        current_set(current_map_points.begin(), current_map_points.end());
    for (const cv::Point3f& point : all_map_points) {
        if (current_set.find(point) == current_set.end()) {
            previous_map_points->push_back(point);
        }
    }
}

std::vector<uint8_t> image_bytes(const cv::Mat& image) {
    cv::Mat continuous = image;
    if (!image.isContinuous()) {
        continuous = image.clone();
    }
    std::vector<uint8_t> bytes(
        continuous.data,
        continuous.data + continuous.total() * continuous.elemSize());
    return bytes;
}

std::vector<rerun::Position2D> points2d_for_rerun(
    const std::vector<cv::Point2f>& points) {
    std::vector<rerun::Position2D> rerun_points;
    rerun_points.reserve(points.size());
    for (const cv::Point2f& point : points) {
        rerun_points.emplace_back(point.x, point.y);
    }
    return rerun_points;
}

std::vector<rerun::Position3D> points3d_for_rerun(
    const std::vector<cv::Point3f>& points) {
    std::vector<rerun::Position3D> rerun_points;
    rerun_points.reserve(points.size());
    for (const cv::Point3f& point : points) {
        rerun_points.emplace_back(point.x, point.y, point.z);
    }
    return rerun_points;
}
#endif

}  // namespace

bool initialize_visualizer(const AppConfig& config, Visualizer* visualizer) {
    bool ok = true;
    visualizer->enabled = false;
    visualizer->parameters = config.parameters.visualization;
    visualizer->trajectory.clear();
    visualizer->loop_edges.clear();
    if (visualizer_requested(config)) {
#if MVO_HAS_RERUN
        visualizer->rec = std::make_unique<rerun::RecordingStream>("mvo");
        if (!config.rerun_save_path.empty()) {
            visualizer->rec->save(config.rerun_save_path).exit_on_failure();
        } else {
            visualizer->rec->spawn().exit_on_failure();
        }
        visualizer->enabled = true;
        std::cout << "rerun=enabled mode="
                  << (config.rerun_save_path.empty() ? "spawn" : "save")
                  << " path=" << config.rerun_save_path << std::endl;
#else
        std::cout << "rerun=unavailable build_without_rerun_sdk"
                  << std::endl;
        ok = false;
#endif
    }
    return ok;
}

void log_visualization(Visualizer* visualizer, int32_t frame_id,
                       const cv::Mat& image,
                       const std::vector<cv::Point2f>& tracked_points,
                       const std::vector<cv::Point3f>& current_map_points,
                       const std::vector<cv::Point3f>& all_map_points,
                       const Pose& pose) {
    const cv::Point3f camera_center = camera_center_from_pose(pose);
    if (visualizer->enabled) {
        visualizer->trajectory.push_back(camera_center);
#if MVO_HAS_RERUN
        visualizer->rec->set_time_sequence("frame", frame_id);
        if (!image.empty()) {
            const std::vector<uint8_t> bytes = image_bytes(image);
            visualizer->rec->log(
                "camera/image",
                rerun::Image::from_grayscale8(
                    bytes, {static_cast<uint32_t>(image.cols),
                            static_cast<uint32_t>(image.rows)}));
        }
        if (!tracked_points.empty()) {
            visualizer->rec->log("camera/klt_tracks",
                                 rerun::Points2D(
                                     points2d_for_rerun(tracked_points))
                                     .with_radii(
                                         rerun::Radius::ui_points(
                                             visualizer->parameters
                                                 .klt_track_radius))
                                     .with_colors(rerun::Color(0, 255, 255)));
        }
        std::vector<cv::Point3f> previous_map_points;
        collect_previous_map_points(all_map_points, current_map_points,
                                    &previous_map_points);
        if (!previous_map_points.empty()) {
            visualizer->rec->log("world/previous_map_points",
                                 rerun::Points3D(
                                     points3d_for_rerun(previous_map_points))
                                     .with_radii(
                                         visualizer->parameters
                                             .previous_map_point_radius)
                                     .with_colors(rerun::Color(0, 0, 0)));
        }
        if (!current_map_points.empty()) {
            visualizer->rec->log("world/current_3d_keypoints",
                                 rerun::Points3D(
                                     points3d_for_rerun(current_map_points))
                                     .with_radii(
                                         visualizer->parameters
                                             .current_map_point_radius)
                                     .with_colors(rerun::Color(255, 0, 0)));
        }
        if (!visualizer->trajectory.empty()) {
            visualizer->rec->log("world/camera_trajectory",
                                 rerun::Points3D(
                                     points3d_for_rerun(
                                         visualizer->trajectory))
                                     .with_radii(
                                         visualizer->parameters
                                             .trajectory_radius)
                                     .with_colors(rerun::Color(0, 128, 255)));
        }
#else
        (void)frame_id;
        (void)image;
        (void)tracked_points;
        (void)current_map_points;
        (void)all_map_points;
#endif
    }
}

void log_loop_edge(Visualizer* visualizer, int32_t frame_id,
                   const cv::Point3f& match_center,
                   const cv::Point3f& query_center) {
    if (visualizer->enabled) {
        visualizer->loop_edges.push_back({match_center, query_center});
#if MVO_HAS_RERUN
        std::vector<rerun::components::LineStrip3D> strips;
        strips.reserve(visualizer->loop_edges.size());
        for (const std::array<cv::Point3f, 2>& edge : visualizer->loop_edges) {
            strips.emplace_back(std::vector<rerun::datatypes::Vec3D>{
                rerun::datatypes::Vec3D(edge[0].x, edge[0].y, edge[0].z),
                rerun::datatypes::Vec3D(edge[1].x, edge[1].y, edge[1].z)});
        }
        visualizer->rec->set_time_sequence("frame", frame_id);
        visualizer->rec->log("world/loop_closures",
                             rerun::LineStrips3D(strips)
                                 .with_radii(
                                     visualizer->parameters.loop_edge_radius)
                                 .with_colors(rerun::Color(0, 200, 0)));
#else
        (void)frame_id;
#endif
    }
}

void log_optimized_trajectory(Visualizer* visualizer, int32_t frame_id,
                              const std::vector<cv::Point3f>& centers) {
    if (visualizer->enabled && !centers.empty()) {
#if MVO_HAS_RERUN
        visualizer->rec->set_time_sequence("frame", frame_id);
        visualizer->rec->log("world/optimized_trajectory",
                             rerun::Points3D(points3d_for_rerun(centers))
                                 .with_radii(
                                     visualizer->parameters
                                         .optimized_trajectory_radius)
                                 .with_colors(rerun::Color(255, 165, 0)));
#else
        (void)frame_id;
#endif
    }
}

void flush_visualizer(Visualizer* visualizer) {
#if MVO_HAS_RERUN
    if (visualizer->enabled) {
        visualizer->rec->flush_blocking(5.0f).exit_on_failure();
    }
#else
    (void)visualizer;
#endif
}

}  // namespace mvo
