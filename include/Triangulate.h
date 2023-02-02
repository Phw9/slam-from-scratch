#pragma once

#include <iostream>
#include "opencv2/calib3d.hpp"
#include "Feature.h"

namespace mvo
{
    class Triangulate
    {
    public:
        Triangulate();
        
        bool CalcWorldPoints(const cv::Mat& pose1,
                            const cv::Mat& pose2,
                            const mvo::Feature& pts1,
                            const mvo::Feature& pts2);
        bool ManageMapPoints(const std::vector<uchar>& mstatus);                           
        bool ManageMapPoints(std::vector<mvo::Feature>& feature);
        bool ScalingPoints();
        void MatToPoints3f();

        cv::Mat mworldMapPoints;
        std::vector<cv::Point3f> mworldMapPointsV;
        std::vector<std::vector<uchar>> mvdesc;
        cv::Vec3d mrvec;
        cv::Vec3d mtvec;
    };
}

bool ManageMapPoints(const std::vector<uchar>& mstatus, std::vector<cv::Point3f>& map);
bool ManageMinusLocal(std::vector<mvo::Feature>& localTrackPoints, const std::vector<int>& id);
bool ManageMinusZ(mvo::Triangulate& map, cv::Mat& R, std::vector<int>& id);
void ManageInlier(std::vector<mvo::Feature>& features2d, std::vector<cv::Point3f>& mapPoints3d, const cv::Mat& inlier);
bool ManageMinusZSFM(mvo::Triangulate& map, cv::Mat& R, std::vector<int>& id);


static double data1[] = {7.070912000000e+02, 0, 6.018873000000e+02,
                0, 7.070912000000e+02, 1.831104000000e+02,
                0, 0, 1};
static cv::Mat intrinsic(cv::Size(3, 3), CV_64FC1, data1);