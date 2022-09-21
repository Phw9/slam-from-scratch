#include <iostream>
#include "Feature.h"


mvo::Feature::Feature()
{
    mfeatures.clear();
    mstatus.clear();
    merr.clear();
}



// bool mvo::Feature::CornerFAST(const cv::Mat& src)
// {
//     // threshold는 120이고 비최대 억제를 수행한다
//     cv::FAST(src, mfastKeyPoints, 120, true);
//     if (mfastKeyPoints.empty())
//     {
//         std::cerr << "Failed FAST detection" << std::endl;
//         return false;
//     }
//     return true;
// }

bool mvo::Feature::GoodFeaturesToTrack(const cv::Mat& src)
{
    cv::goodFeaturesToTrack(src, mfeatures, MAXCORNER, QUALITYLEVEL, MINDISTANCE);
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
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.11);
    
    cv::calcOpticalFlowPyrLK(src1, src2, mfeatures, next.mfeatures, next.mstatus, next.merr,
                             winSize, 3, termcrit, 0, 0.0001);

    int indexCorrection = 0;
    for(int i = 0; i < next.mstatus.size(); i++)
    {
        cv::Point2f pt = next.mfeatures.at(i - indexCorrection);

        if((next.mstatus.at(i) == 0) || 
            (pt.x < 0 || pt.x > (float)1241.0) || 
            (pt.y < 0 || pt.y > (float)376.0) ||
            next.merr.at(i) > ERRORSIZE)
        {
            if((pt.x < 0 || pt.x > (float)1241.0) || 
                (pt.y < 0 || pt.y > (float)376.0) ||
                next.merr.at(i) > ERRORSIZE)
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
 
bool ManageMapPoints(const std::vector<uchar>& mstatus, std::vector<cv::Point3f>& map)
{
    int indexCorrection = 0;
    
    if(map.size() == mstatus.size())
    {
        for(int i = 0; i < mstatus.size(); i++)
        {
            if(mstatus.at(i) == 0)
            {
                map.erase(map.begin() + (i - indexCorrection));
                indexCorrection++;
            }
        }
        // for(int i = 0; i < map.size(); i++)
        // {
        //     std::cout << map.at(i).x << " " << map.at(i).y << " " << map.at(i).z << std::endl;
        // }
        return true;
    }
    return false;
}
bool ManageMinusZ(mvo::Triangulate& map, cv::Mat& R, std::vector<int>& id)
{
    double tempd = 0;
    for(int i = 0; i < map.mworldMapPointsV.size(); i++)
    {
        tempd = (R.at<double>(2,0)*map.mworldMapPointsV.at(i).x) +
                (R.at<double>(2,1)*map.mworldMapPointsV.at(i).y) +
                (R.at<double>(2,2)*map.mworldMapPointsV.at(i).z);
        if(tempd < 0)
        {
            id.emplace_back(i);
            map.mworldMapPointsV.erase(map.mworldMapPointsV.begin()+i);
        }
    }
    if(map.mworldMapPointsV.size() == 0) return false;

    return true;
}

bool ManageMinusLocal(std::vector<mvo::Feature>& localTrackPoints, const std::vector<int>& id)
{
    for(int i = 0; i<localTrackPoints.size(); i++)
    {
        for(int j = 0; j<id.size(); j++)
        {
            localTrackPoints.at(i).mfeatures.erase(localTrackPoints.at(i).mfeatures.begin()+id.at(j));
        }
    }
    
    if(localTrackPoints.at(0).mfeatures.size() == localTrackPoints.at(localTrackPoints.size()-1).mfeatures.size()) return true;
    
    return false;
}

void ManageInlier(std::vector<mvo::Feature>& features2d, std::vector<cv::Point3f>& mapPoints3d, const cv::Mat& inlier)
{
    int interval;
    std::vector<int> idv;
    int ilast = inlier.at<int>(inlier.rows-1,0);
    int flast = features2d.at(0).mfeatures.size() - 1;
    idv.reserve(3000);
    idv.emplace_back(inlier.at<int>(0,0)-1);
    for(int i = 0; i < inlier.rows; i++)
    {
        idv.emplace_back(inlier.at<int>(i,0));
    }

    for(int i=0; i<idv.size()-1; i++)
    {
        interval = idv.at(i+1) - idv.at(i)-1;
        for(int j = i; j<interval+i; j++)
        {
            for(int k = 0; k < features2d.size(); k++)
            {
                features2d.at(k).mfeatures.erase(features2d.at(k).mfeatures.begin()+j);
            }
            mapPoints3d.erase(mapPoints3d.begin()+j);  
        }

        if(i == idv.size()-2 && flast-ilast != 0)
        {
            for(int l = 0; l < features2d.size(); l++)
            {
                for(int h = 0; h < flast-ilast; h++)
                {
                    features2d.at(l).mfeatures.pop_back();
                    features2d.at(l).mfeatures.pop_back();
                }
            }
            for(int h = 0; h < flast-ilast; h++)
            {
                mapPoints3d.pop_back();
                mapPoints3d.pop_back();
            }
            break;
        }
    }
}