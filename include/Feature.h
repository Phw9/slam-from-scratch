#pragma once

#include <vector>
#include "eigen3/Eigen/Dense"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/video.hpp"

namespace mvo
{
    class Feature
    {
    public:
        Feature();
        
        bool CornerFAST(const cv::Mat& src);
        bool GoodFeaturesToTrack(const cv::Mat& src);
        bool OpticalFlowPyrLK(const cv::Mat& src1, const cv::Mat& src2, mvo::Feature& next);

        std::vector<uchar> mstatus;
        std::vector<float> merr;
        std::vector<cv::Point2f> mfeatures;
        cv::Mat mdesc;
        std::vector<cv::KeyPoint> mfastKeyPoints;
    };
}//namespace mvo

void ManageTrackPoints(const mvo::Feature& present, mvo::Feature& before);
void ManageTrackPoints(const mvo::Feature& present, mvo::Feature& before, std::vector<cv::Point3d>& mapPoints);