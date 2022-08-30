#pragma once

#include <iostream>
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"

namespace mvo
{
    class KeyFrame
    {
    public:
        KeyFrame();
        bool CalcWorldPoints(const cv::Mat& pose1,
                            const cv::Mat& pose2,
                            const std::vector<cv::Vec2f>& pts1,
                            const std::vector<cv::Vec2f>& pts2);
        bool ScalingPoints();

    public:
        cv::Mat mworldMapPoints;
    };
}