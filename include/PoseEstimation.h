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
        bool solvePnP(const std::vector<cv::Point3f>& objectPoints,
                    const std::vector<cv::Point2f>& imagePoints,
                    const cv::Mat& cameraIntrinsic);
        void GetRMatTPose();
        bool CombineRt();

    public:
        cv::Mat mRotation;
        cv::Mat mTranslation;
        cv::Mat mCombineRt;
        cv::Vec3d mrvec;
        cv::Mat mtvecBef;
        cv::Vec3d mtvec;
    };

    cv::Mat MultiplyMat(const cv::Mat& R1, const cv::Mat& R2)
    {
        float data[] = {0,0,0,1};
        cv::Mat rowVec(cv::Size(1,4), CV_32F, data);
        cv::Mat temp1 = R1;
        cv::Mat temp2 = R2;
        temp1.push_back(R1);
        temp2.push_back(R2);
        std::cout << "temp1: " << temp1 << std::endl;
        std::cout << "temp2: " << temp2 << std::endl;
        std::cout << "t2*t1: " << temp2*temp1 << std::endl;
        return temp2*temp1;
    }
} //namespace mvo

