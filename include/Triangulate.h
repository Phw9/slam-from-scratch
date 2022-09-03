#pragma once

#include <iostream>
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"

namespace mvo
{
    class Triangulate
    {
    public:
        Triangulate();
        bool CalcWorldPoints(const cv::Mat& pose1,
                            const cv::Mat& pose2,
                            const std::vector<cv::Point2f>& pts1,
                            const std::vector<cv::Point2f>& pts2);
        bool ScalingPoints();
        void MatToPoints3f();
    public:
        cv::Mat mworldMapPoints;
        std::vector<cv::Point3f> mworldMapPointsV;
    };
}