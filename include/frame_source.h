#pragma once

#include "config.h"

#include <opencv2/videoio.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mvo {

struct FrameSource {
    InputType input_type = InputType::kImageSequence;
    SensorMode sensor_mode = SensorMode::kMonocular;
    std::vector<std::string> images;
    std::vector<std::string> right_images;
    cv::VideoCapture video;
    int32_t next_index = 0;
    int32_t total_frames = 0;
    bool opened = false;
};

bool open_frame_source(const AppConfig& config, FrameSource* source);
bool read_next_frame(FrameSource* source, cv::Mat* gray);
bool read_next_stereo_frame(FrameSource* source, cv::Mat* left,
                            cv::Mat* right);

}  // namespace mvo
