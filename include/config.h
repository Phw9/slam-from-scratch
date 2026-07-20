#pragma once

#include "parameters.h"
#include "types.h"

#include <cstdint>
#include <string>

namespace mvo {

struct AppConfig {
    InputType input_type = InputType::kImageSequence;
    SensorMode sensor_mode = SensorMode::kMonocular;
    std::string input_path = "image/image_0";
    std::string right_input_path = "image/image_1";
    std::string parameter_dir = "configs/parameters";
    std::string calib_path = "image/calib.txt";
    std::string vocabulary = "image/KITTI_00_phphww_voc.yml.gz";
    CameraIntrinsics camera;
    double stereo_baseline = 0.0;
    int32_t max_frames = 20;
    bool run_ba = true;
    bool debug_geometry = false;
    bool rerun_spawn = true;
    std::string rerun_save_path;
    MvoParameters parameters;
};

void set_identity_pose(Pose* pose);
std::string input_type_name(InputType input_type);
std::string sensor_mode_name(SensorMode sensor_mode);
AppConfig parse_args(int argc, char** argv);
bool load_kitti_calibration(const std::string& path,
                            CameraIntrinsics* camera);
bool load_kitti_stereo_baseline(const std::string& path,
                                double* baseline);

}  // namespace mvo
