#pragma once

#include "Feature.h"
#include "opencv2/calib3d.hpp"

namespace mvo
{
    class StrctureFromMotion
    {
    public:
        StrctureFromMotion();

        bool CreateEssentialMatrix(const std::vector<cv::Point2f>& pts1, 
                                    const std::vector<cv::Point2f>& pts2, 
                                    const cv::InputArray& K);
        bool GetEssentialRt(const cv::InputArray& E, const cv::InputArray& K, 
                            const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);
        bool CombineRt();
        bool GetRTvec();
    public:
        cv::Mat mEssential;
        cv::Mat mRotation;
        cv::Mat mTranslation;
        cv::Mat mCombineRt;
        cv::Vec3d mrvec;
        cv::Vec3d mtvec;
    };

    class PoseEstimation
    {
    public:
        PoseEstimation();

    public:
        bool solvePnP(const std::vector<cv::Point3f>& objectPoints,
                    const std::vector<cv::Point2f>& imagePoints,
                    const cv::Mat cameraIntrinsic,
                    cv::OutputArray rvec,
                    cv::OutputArray tvec);

    public:
        cv::Vec3f rvec;
        cv::Vec3f tvec;
    };
} //namespace mvo

