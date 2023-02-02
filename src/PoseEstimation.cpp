#include <iostream>
#include "Feature.h"
#include "PoseEstimation.h"
#include "Config.h"

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

    cv::recoverPose(mEssential, pts1, pts2, K, mRotation, mTranslation);
    if (mRotation.empty() || mTranslation.empty())
    {
        std::cerr << "Can't get Essential Rt" << std::endl;
        return false;
    }

    cv::Mat temp = cv::Mat();
    temp = mRotation.t();
    temp.push_back(mTranslation.t());
    mCombineRt = temp.t();
    if (!(mCombineRt.rows == 3 && mCombineRt.cols == 4))
    {
        std::cerr << "failed Combine Rotation Translation Matrix" << std::endl;
        return false;
    }

    cv::Rodrigues(mRotation, mrvec);
    for (int j = 0; j < mTranslation.rows; j++)
    {
		for (int i = 0; i < mTranslation.cols; i++)
        {
			mtvec[j] = mTranslation.at<double>(j, i);
		}
	}
    // std::cout << "mCombineRT: " << mCombineRt << std::endl;
    return true;
}

bool mvo::StrctureFromMotion::CreateHomographyMatrix(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const cv::Mat& K)
{
    mHomography = cv::findHomography(pts1, pts2, 0, 3.0, cv::noArray(), 2000, 0.995);
    if (mHomography.empty())
    {
        std::cerr << "Can't find Homography matrix" << std::endl;
        return false;
    }
    cv::Mat dd;
    std::cout << "mHomography: " << mHomography << std::endl;
    std::cout << "K: " << K << std::endl;
    cv::decomposeHomographyMat(mHomography, K, mRotation, mTranslation, cv::noArray());
    std::cout << "dd: " << dd << std::endl;

    if (mRotation.empty() || mTranslation.empty())
    {
        std::cerr << "Can't get Essential Rt" << std::endl;
        return false;
    }

    cv::Mat temp = cv::Mat();
    temp = mRotation.t();
    temp.push_back(mTranslation.t());
    mCombineRt = temp.t();
    if (!(mCombineRt.rows == 3 && mCombineRt.cols == 4))
    {
        std::cerr << "failed Combine Rotation Translation Matrix" << std::endl;
        return false;
    }
    std::cout << mCombineRt << std::endl;
    cv::Rodrigues(mRotation, mrvec);
    for (int j = 0; j < mTranslation.rows; j++)
    {
		for (int i = 0; i < mTranslation.cols; i++)
        {
			mtvec[j] = mTranslation.at<double>(j, i);
		}
	}
    // std::cout << "mCombineRT: " << mCombineRt << std::endl;
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
            mRotation{cv::Mat()}, mtvec{cv::Mat()}, mCombineRt{cv::Mat()}{};

void mvo::PoseEstimation::solvePnP(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    const cv::Mat& cameraIntrinsic)
{
    // for(auto& n:objectPoints)
    // {
    //     std::cout << n;
    // }
    if(!cv::solvePnPRansac(objectPoints, imagePoints, cameraIntrinsic, cv::Mat(), mrvec, mtvec, false, 100, 3.0F, 0.99, minlier, cv::SOLVEPNP_ITERATIVE))
    {
        std::cerr <<"Can't solve PnP" << std::endl;    
    }
    // std::cout << minlier << std::endl;
    std::cout << "inlier.rows: " << minlier.rows << std::endl;
    
    cv::Rodrigues(mrvec, mRotation);
    cv::Mat temp = -mRotation.inv();
    temp = temp*mtvec;

    for (int j = 0; j < mtvec.rows; j++)
    {
		for (int i = 0; i < mtvec.cols; i++)
        {
			mtranslation[j] = temp.at<double>(j, i);
		}
	}

    cv::Mat tempm = cv::Mat();
    tempm = mRotation.t();
    tempm.push_back(mtvec.t());
    mCombineRt = tempm.t();
    if (!(mCombineRt.rows == 3 && mCombineRt.cols == 4))
    {
        std::cerr << "failed Combine Rotation Translation Matrix" << std::endl;
    }
}

void mvo::PoseEstimation::GetRMatTPose()
{
    cv::Rodrigues(mrvec, mRotation);
    cv::Mat temp = -mRotation.inv();
    temp = temp*mtvec;

    for (int j = 0; j < mtvec.rows; j++)
    {
		for (int i = 0; i < mtvec.cols; i++)
        {
			mtranslation[j] = temp.at<double>(j, i);
		}
	}
}

bool mvo::PoseEstimation::CombineRt()
{
    cv::Mat temp = cv::Mat();
    temp = mRotation.t();
    temp.push_back(mtvec.t());
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
        double data[] = {0,0,0,1};
        cv::Mat rowVec(cv::Size(4,1), CV_64F, data);
        cv::Mat temp1 = R1;
        cv::Mat temp2 = R2;
        cv::Mat temp3;
        temp1.push_back(rowVec);
        temp2.push_back(rowVec);
        temp3= temp1*temp2;
        temp3.pop_back();
        return temp3;
}

double mvo::RotationAngle(const cv::Vec3d& R1, const cv::Vec3d& R2)
{
    cv::Vec3d tempV;
    cv::Mat temp; cv::Vec<double,1> v;
    cv::Mat tempR1;
    cv::Mat tempR2;
    cv::Rodrigues(R1, tempR1);
    cv::Rodrigues(R2, tempR2);
    tempR2 = tempR2.inv();
    temp = tempR1 * tempR2;
    cv::Rodrigues(temp, tempV);
    v = tempV.t() * tempV;
    double theta;
    theta = sqrt(v(0));
    return theta;
}