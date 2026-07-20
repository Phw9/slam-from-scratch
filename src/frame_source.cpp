#include "frame_source.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>

namespace mvo {
namespace {

std::vector<std::string> list_images(const std::string& image_dir) {
    std::vector<std::string> images;
    const std::filesystem::path dir(image_dir);
    if (std::filesystem::exists(dir)) {
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                images.push_back(entry.path().string());
            }
        }
        std::sort(images.begin(), images.end());
    }
    return images;
}

bool convert_to_grayscale(const cv::Mat& image, cv::Mat* gray) {
    bool ok = false;
    if (!image.empty()) {
        if (image.channels() == 1) {
            *gray = image;
        } else {
            cv::cvtColor(image, *gray, cv::COLOR_BGR2GRAY);
        }
        ok = !gray->empty();
    }
    return ok;
}

}  // namespace

bool open_frame_source(const AppConfig& config, FrameSource* source) {
    bool ok = false;
    source->input_type = config.input_type;
    source->sensor_mode = config.sensor_mode;
    source->next_index = 0;
    source->total_frames = 0;
    source->opened = false;

    if (config.sensor_mode == SensorMode::kStereo) {
        // Stereo consumes rectified left/right image pairs; video input has
        // no second stream, so only image sequences are supported.
        if (config.input_type == InputType::kImageSequence) {
            source->images = list_images(config.input_path);
            source->right_images = list_images(config.right_input_path);
            source->total_frames = static_cast<int32_t>(
                std::min(source->images.size(),
                         source->right_images.size()));
            ok = source->total_frames >= 2;
        }
    } else if (config.input_type == InputType::kImageSequence) {
        source->images = list_images(config.input_path);
        source->total_frames = static_cast<int32_t>(source->images.size());
        ok = source->total_frames >= 2;
    } else if (config.input_type == InputType::kVideo) {
        source->video.open(config.input_path);
        ok = source->video.isOpened();
        if (ok) {
            const double frame_count =
                source->video.get(cv::CAP_PROP_FRAME_COUNT);
            if (frame_count > 0.0) {
                source->total_frames = static_cast<int32_t>(frame_count);
            } else {
                source->total_frames = config.max_frames;
            }
        }
    }

    source->opened = ok;
    return ok;
}

bool read_next_frame(FrameSource* source, cv::Mat* gray) {
    bool ok = false;
    if (source->opened) {
        cv::Mat image;
        if (source->input_type == InputType::kImageSequence) {
            if (source->next_index <
                static_cast<int32_t>(source->images.size())) {
                image = cv::imread(
                    source->images[static_cast<std::size_t>(
                        source->next_index)],
                    cv::IMREAD_GRAYSCALE);
                ++source->next_index;
            }
        } else if (source->video.read(image)) {
            ++source->next_index;
        }
        ok = convert_to_grayscale(image, gray);
    }
    return ok;
}

bool read_next_stereo_frame(FrameSource* source, cv::Mat* left,
                            cv::Mat* right) {
    bool ok = false;
    if (source->opened && source->next_index < source->total_frames) {
        const std::size_t index =
            static_cast<std::size_t>(source->next_index);
        const cv::Mat left_image = cv::imread(source->images[index],
                                              cv::IMREAD_GRAYSCALE);
        const cv::Mat right_image = cv::imread(source->right_images[index],
                                               cv::IMREAD_GRAYSCALE);
        ++source->next_index;
        ok = convert_to_grayscale(left_image, left) &&
             convert_to_grayscale(right_image, right);
    }
    return ok;
}

}  // namespace mvo
