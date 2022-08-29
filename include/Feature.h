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
        bool OpticalFlowPyrLK(const cv::Mat& src1, const cv::Mat& src2, std::vector<cv::Vec2f>& pts1);

        std::vector<uchar> mstatus;
        std::vector<float> merr;
        std::vector<cv::Vec2f> mfeatures;
        cv::Mat mdesc;
        std::vector<cv::KeyPoint> mfastKeyPoints;
    };
}//namespace mvo

bool ManageTrackPoints(const mvo::Feature& present, mvo::Feature before);