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

bool mvo::Feature::OpticalFlowPyrLK(const cv::Mat& src1, const cv::Mat& src2, mvo::Feature& next)
{
    cv::Size winSize = cv::Size(21,21);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
    
    cv::calcOpticalFlowPyrLK(src1, src2, mfeatures, next.mfeatures, next.mstatus, next.merr,
                             winSize, 3, termcrit, 0, 0.0001);

    int indexCorrection = 0;
    for(int i = 0; i < next.mstatus.size(); i++)
    {
        cv::Point2f pt = next.mfeatures.at(i - indexCorrection);
        if((next.mstatus.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
        {
            if(((pt.x < 0) || (pt.y < 0)))
            {
                next.mstatus.at(i) = 0;
            }
            mfeatures.erase(mfeatures.begin() + (i - indexCorrection));           // time complexity
            next.mfeatures.erase(next.mfeatures.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
    if(next.mfeatures.empty())
    {
        std::cerr << "failed calcOpticalFlowPyrLK" << std::endl;
        return false;
    }
    return true;
}

void ManageTrackPoints(const mvo::Feature& present, mvo::Feature& before)
{
    int indexCorrection = 0;
    for(int i = 0; i < present.mstatus.size(); i++)
    {
        if(present.mstatus.at(i) == 0)
        {
            before.mfeatures.erase(before.mfeatures.begin() + (i-indexCorrection));
            indexCorrection++;
        }
    }
}
void ManageTrackPoints(const mvo::Feature& present, mvo::Feature& before, std::vector<cv::Point3d>& mapPoints)
{
    int indexCorrection = 0;
    
    if(mapPoints.size() == before.mfeatures.size())
    {
        for(int i = 0; i < present.mstatus.size(); i++)
        {
            if(present.mstatus.at(i) == 0)
            {
                mapPoints.erase(mapPoints.begin() + (i - indexCorrection));
                indexCorrection++;
            }
        }
    }
    for(int i = 0; i < present.mstatus.size(); i++)
    {
        if(present.mstatus.at(i) == 0)
        {
            before.mfeatures.erase(before.mfeatures.begin() + (i-indexCorrection));
            indexCorrection++;
        }
    }
}