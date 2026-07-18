#pragma once

#include "config.h"

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#ifndef MVO_HAS_RERUN
#define MVO_HAS_RERUN 0
#endif

#if MVO_HAS_RERUN
#include <rerun.hpp>
#endif

namespace mvo {

struct Visualizer {
    bool enabled = false;
    VisualizationParameters parameters;
    std::vector<cv::Point3f> trajectory;
    std::vector<std::array<cv::Point3f, 2>> loop_edges;
#if MVO_HAS_RERUN
    std::unique_ptr<rerun::RecordingStream> rec;
#endif
};

bool initialize_visualizer(const AppConfig& config, Visualizer* visualizer);
void log_visualization(Visualizer* visualizer, int32_t frame_id,
                       const cv::Mat& image,
                       const std::vector<cv::Point2f>& current_points,
                       const std::vector<cv::Point3f>& current_map_points,
                       const std::vector<cv::Point3f>& all_map_points,
                       const Pose& pose);
void log_loop_edge(Visualizer* visualizer, int32_t frame_id,
                   const cv::Point3f& match_center,
                   const cv::Point3f& query_center);
void log_optimized_trajectory(Visualizer* visualizer, int32_t frame_id,
                              const std::vector<cv::Point3f>& centers);
void flush_visualizer(Visualizer* visualizer);

}  // namespace mvo
