#include <iostream>
#include "Feature.h"
#include "PoseEstimation.h"

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
    for (int j = 0; j < mTranslation.rows; j++)
    {
		for (int i = 0; i < mTranslation.cols; i++)
        {
			mtvec[j] = mTranslation.at<double>(j, i);
		}
	}
}

mvo::PoseEstimation::PoseEstimation(): 
            mRotation{cv::Mat()}, mTranslation{cv::Mat()}, mCombineRt{cv::Mat()}{};

bool mvo::PoseEstimation::solvePnP(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    const cv::Mat& cameraIntrinsic)
{   
    // std::vector<cv::Point3f> v;
    // cv::Point3f temp;
    // for(int i = 0; i<objectPoints.size(); i++)
    // {
    //    temp.x= (float)objectPoints.at(i).x;
    //    temp.y= (float)objectPoints.at(i).y;
    //    temp.z= (float)objectPoints.at(i).z;
    //    v.emplace_back(std::move(temp));
    // }
    
    if(!cv::solvePnP(objectPoints, imagePoints, cameraIntrinsic, cv::Mat(), mrvec, mTranslation))
    {
        std::cerr <<"Can't solve PnP" << std::endl;
        return false;
    }

    // cv::Mat inlier;
    // if(!cv::solvePnPRansac(objectPoints, imagePoints, cameraIntrinsic, cv::Mat(), mrvec, mTranslation, false, 100, 3.0F, 0.99, inlier, cv::SOLVEPNP_ITERATIVE))
    // {
    //     std::cerr <<"Can't solve PnP" << std::endl;
    //     return false;
    // }
    return true;
}

void mvo::PoseEstimation::GetRMatTPose()
{
    cv::Rodrigues(mrvec, mRotation);
    cv::Mat temp = -mRotation.inv()*mTranslation;

    for (int j = 0; j < mTranslation.rows; j++)
    {
		for (int i = 0; i < mTranslation.cols; i++)
        {
			mtvec[j] = temp.at<double>(j, i);
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

cv::Mat mvo::MultiplyMat(const cv::Mat& R1, const cv::Mat& R2)
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