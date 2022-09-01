#include <iostream>
#include "../include/Feature.h"
#include "../include/PoseEstimation.h"

#include "opencv2/core.hpp"

mvo::StrctureFromMotion::StrctureFromMotion() 
: mEssential{cv::Mat()}, mRotation{cv::Mat()}, mTranslation{cv::Mat()}, mCombineRt{cv::Mat()}{}

bool mvo::StrctureFromMotion::CreateEssentialMatrix(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const cv::InputArray& K)
{
    mEssential = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, cv::noArray());
    if (mEssential.empty())
    {
        std::cerr << "Can't find essential matrix" << std::endl;
        return false;
    }
    return true;
}

bool mvo::StrctureFromMotion::GetEssentialRt(const cv::InputArray& E, const cv::InputArray& K, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2)
{
    cv::recoverPose(E, pts1, pts2, K, mRotation, mTranslation);
    if (mRotation.empty() || mTranslation.empty())
    {
        std::cerr << "Can't get Essential Rt" << std::endl;
        return false;
    }
    return true;
}

bool mvo::StrctureFromMotion::CombineRt()
{
    cv::Mat temp = cv::Mat();
    temp = mRotation.t();
    temp.push_back(mTranslation.t());
    mCombineRt = temp.t();
    if (!(mCombineRt.rows == 3 && mCombineRt.cols == 4))
    {
        std::cerr << "failed Combine Rotation Translation Matrix" << std::endl;
        return false;
    }
    return true;
}

void mvo::StrctureFromMotion::GetRTvec()
{
    cv::Rodrigues(mRotation, mrvec);
    for (int j = 0; j < mTranslation.rows; j++) {
		for (int i = 0; i < mTranslation.cols; i++) {
			mtvec[j] = mTranslation.at<double>(j, i);
		}
	}
}

mvo::PoseEstimation::PoseEstimation(): 
            mRotation{cv::Mat()}, mTranslation{cv::Mat()}, mCombineRt{cv::Mat()}{};

bool mvo::PoseEstimation::solvePnP(const std::vector<cv::Point3d>& objectPoints,
                    const std::vector<cv::Point2f>& imagePoints,
                    const cv::Mat& cameraIntrinsic)
{
    if(!cv::solvePnPRansac(objectPoints, imagePoints, cameraIntrinsic, cv::Mat(), mrvec, mTranslation))
    {
        std::cerr <<"Can't solve PnP" << std::endl;
    }
    return true;
}

void mvo::PoseEstimation::GetRTMat()
{
    cv::Rodrigues(mrvec, mRotation);
    cv::Mat temp = -mRotation.inv()*mTranslation;

    for (int j = 0; j < mTranslation.rows; j++) {
		for (int i = 0; i < mTranslation.cols; i++) {
			mtvec[j] = mTranslation.at<double>(j, i);
		}
	}
}

bool mvo::PoseEstimation::CombineRt()
{
    cv::Mat temp = cv::Mat();
    temp = mRotation.t();
    temp.push_back(mTranslation.t());
    mCombineRt = temp.t();
    if (!(mCombineRt.rows == 3 && mCombineRt.cols == 4))
    {
        std::cerr << "failed Combine Rotation Translation Matrix" << std::endl;
        return false;
    }
    return true;
}