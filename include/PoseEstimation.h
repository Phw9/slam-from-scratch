#pragma once

#include "Feature.h"
#include "opencv2/calib3d.hpp"

namespace mvo
{
    //2-view SFM
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
        void GetRTvec();
    public:
        cv::Mat mEssential;
        cv::Mat mRotation;
        cv::Mat mTranslation;
        cv::Mat mCombineRt;
        cv::Vec3d mrvec;
        cv::Vec3d mtvec;
    };

    // solvePnP
    class PoseEstimation
    {
    public:
        PoseEstimation();

    public:
        void solvePnP(const std::vector<cv::Point3f>& objectPoints,
                    const std::vector<cv::Point2f>& imagePoints,
                    const cv::Mat& cameraIntrinsic);
        void GetRMatTPose();
        bool CombineRt();

    public:
        cv::Mat mRotation;
        cv::Mat mTranslation;
        cv::Mat mCombineRt;
        cv::Vec3d mrvec;
        cv::Vec3d mtvec;
        cv::Mat mtvecBef;
        cv::Mat minlier;
    };

    cv::Mat MultiplyMat(const cv::Mat& R1, const cv::Mat& R2);
    double RotationAngle(const cv::Mat& R1, const cv::Mat& R2);
} //namespace mvo

