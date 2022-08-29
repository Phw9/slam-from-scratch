#include <iostream>
#include "../include/Feature.h"

mvo::Feature::Feature() : mdesc{cv::Mat()}
{
    mfastKeyPoints.clear();
    mfeatures.clear();
    mstatus.clear();
    merr.clear();
    mfeatures.clear();
}



bool mvo::Feature::CornerFAST(const cv::Mat& src)
{
    // threshold는 120이고 비최대 억제를 수행한다
    cv::FAST(src, mfastKeyPoints, 120, true);
    if (mfastKeyPoints.empty())
    {
        std::cerr << "Failed FAST detection" << std::endl;
        return false;
    }
    return true;
}

bool mvo::Feature::GoodFeaturesToTrack(const cv::Mat& src)
{
    cv::goodFeaturesToTrack(src, mfeatures, 1000, 0.01, 10);
    if(mfeatures.empty())
    {
        std::cerr << "failed to goodFeaturesToTrack" << std::endl;
        return false;
    }
    else if(mfeatures.size() < 100)
    {
        std::cerr << "tracker size small than 100" << std::endl;
        return false;
    }
    return true;
}

bool mvo::Feature::OpticalFlowPyrLK(const cv::Mat& src1, const cv::Mat& src2, std::vector<cv::Vec2f>& pts1)
{
    cv::Size winSize = cv::Size(21,21);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
    
    cv::calcOpticalFlowPyrLK(src1, src2, pts1, mfeatures, mstatus, merr, winSize, 3, termcrit, 0, 0.0001);

    int indexCorrection = 0;
    for(int i = 0; i < mstatus.size(); i++)
    {
        cv::Vec2f pt = mfeatures.at(i - indexCorrection);
        if((mstatus.at(i) == 0) || (pt[0] < 0) || (pt[1] < 0))
        {
            if(((pt[0] < 0) || (pt[1] < 0)))
            {
                mstatus.at(i) = 0;
            }
            pts1.erase(pts1.begin() + (i - indexCorrection));           // time complexity
            mfeatures.erase(mfeatures.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
    if(mfeatures.empty())
    {
        std::cerr << "failed calcOpticalFlowPyrLK" << std::endl;
        return false;
    }
    return true;
}

bool ManageTrackPoints(const mvo::Feature& present, mvo::Feature before)
{
    int indexCorrection = 0;
    for(int i = 0; i < present.mstatus.size(); i++)
    {
        cv::Vec2f pt = before.mfeatures.at(i-indexCorrection);
        if(present.mstatus.at(i) == 0)
        {
            before.mfeatures.erase(before.mfeatures.begin() + (i-indexCorrection));
            indexCorrection++;
        }
    }
}