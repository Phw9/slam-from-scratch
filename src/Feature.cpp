#include <iostream>
#include "Feature.h"
#include "opencv2/core/hal/interface.h"


mvo::Feature::Feature()
{
    mfeatures.clear();
    mstatus.clear();
    merr.clear();
    mvdesc.clear();
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
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create();

    std::vector<cv::KeyPoint> kp;
    VecToKeyPoint(mfeatures, kp);
    // std::cout << "before mfeatures: " << mfeatures.size() << std::endl;
    // std::cout << "before kp size: " << kp.size() << std::endl;
    brief->compute(src, kp, mdesc);
    mfeatures.clear();
    KeyPointToVec(kp, mfeatures);
    mvdesc.clear();
    MatToVec(mdesc, mvdesc);
    // std::cout << "desc mat size: " << desc1.size() << std::endl;
    // std::cout << "after mfeatures: " << mfeatures.size() << std::endl;
    // std::cout << "after kp size: " << kp.size() << std::endl;
    // std::cout << "mvdesc: " << mvdesc.size() << std::endl;
    

    return true;
}

bool mvo::Feature::OpticalFlowPyrLK(const cv::Mat& src1, const cv::Mat& src2, mvo::Feature& next)
{
    next.mfeatures.clear(); next.mstatus.clear(); next.merr.clear(); next.mvdesc.clear();
    cv::Size winSize = cv::Size(21,21);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 0.01);
    cv::calcOpticalFlowPyrLK(src1, src2, mfeatures, next.mfeatures, next.mstatus, next.merr,
                             winSize, 3, termcrit, 0, 0.001);



    int indexCorrection = 0;
    int N = next.mstatus.size();
    for(int i = 0; i < N; i++)
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
            mvdesc.erase(mvdesc.begin() + (i - indexCorrection));           // time complexity
            next.mfeatures.erase(next.mfeatures.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create();
    std::vector<cv::KeyPoint> kp2;
    std::vector<uchar> idx;
    VecToKeyPoint(next.mfeatures, kp2);
    brief->compute(src2, kp2, next.mdesc);
    next.mdelete = FindDeletePoints(kp2, next.mfeatures);
    DeletePoints(next.mdelete, mvdesc, mfeatures);
    next.mfeatures.clear(); next.mvdesc.clear();
    KeyPointToVec(kp2, next.mfeatures);
    MatToVec(next.mdesc, next.mvdesc);
    
    if(next.mfeatures.empty())
    {
        std::cerr << "failed calcOpticalFlowPyrLK" << std::endl;
        return false;
    }
    if(next.mvdesc.empty())
    {
        std::cerr << "failed calcOpticalFlowPyrLK descriptor" << std::endl;
        return false;
    }

    return true;
}

void ManageTrackPoints(const mvo::Feature& present, mvo::Feature& before)
{
    int indexCorrection = 0;
    int N = present.mstatus.size();
    for(int i = 0; i < N; i++)
    {
        if(present.mstatus.at(i) == 0)
        {
            before.mfeatures.erase(before.mfeatures.begin() + (i-indexCorrection));
            before.mvdesc.erase(before.mvdesc.begin() + (i-indexCorrection));
            indexCorrection++;
        }
    }
    indexCorrection = 0;
    int M = present.mdelete.size();
    for(int i = 0; i < M; i++)
    {
        if(present.mdelete.at(i) == 0)
        {
            before.mfeatures.erase(before.mfeatures.begin() + (i-indexCorrection));
            before.mvdesc.erase(before.mvdesc.begin() + (i-indexCorrection));
            indexCorrection++;
        }
    }
}
 
bool KeyPointToVec(const std::vector<cv::KeyPoint>& kp, std::vector<cv::Point2f>& features2d)
{
    cv::Point2f temp;
    int N = kp.size();
    for(int i = 0; i < N; i++)
    {
        temp.x = kp.at(i).pt.x;
        temp.y = kp.at(i).pt.y;
        features2d.emplace_back(std::move(temp));
    }

    if(features2d.size() != kp.size()) return false;

    return true;
}

bool VecToKeyPoint(const std::vector<cv::Point2f>& features2d, std::vector<cv::KeyPoint>& kp)
{
    cv::KeyPoint temp;
    int N = features2d.size();
    for(int i = 0; i < N; i++)
    {
        temp.pt.x = features2d.at(i).x;
        temp.pt.y = features2d.at(i).y;
        kp.emplace_back(std::move(temp));
    }

    if(features2d.size() != kp.size()) return false;

    return true;
}

bool MatToVec(const cv::Mat& m, std::vector<std::vector<DTYPE>>& v)
{
    std::vector<DTYPE> temp;
    for(int j=0; j<m.rows ; j++)
    {
        for(int i=0; i<m.cols; i++)
        {
            temp.emplace_back(m.at<uchar>(j, i));
        }
        v.emplace_back(std::move(temp));
        temp.clear();
    }
    int N = v.size();
    if(m.rows != N) return false;

    return true;
}

bool VecToMat(const std::vector<std::vector<uchar>>& v, cv::Mat& m)
{
    int M=v.size(); int N = v.at(0).size();
    cv::Mat temp(cv::Size(N, M), CV_8UC1);

    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            temp.at<uchar>(i,j) = v.at(i).at(j);
        }
    }
    m=temp.clone();
    return true;
}

std::vector<uchar> FindDeletePoints(std::vector<cv::KeyPoint>& kp, std::vector<cv::Point2f>& mfeatures)
{
    // kp < mfeatures
    std::vector<uchar> index;
    int k = 0;
    int M = mfeatures.size();
    int P = kp.size();
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < P; j++)
        {
            if(mfeatures.at(i).x == kp.at(j).pt.x && mfeatures.at(i).y == kp.at(j).pt.y)
            {
                index.emplace_back(1);
                k = 1;
                break;
            }
        }
        if(k == 0)
        {
            index.emplace_back(0);
        }
        k = 0;
    }
    int t = 0;
    int f = 0;
    int N = index.size();
    for(int i = 0; i < N; i++)
    {
        if(index.at(i) == 0) f++;
        else t++;
    }
    return index;
}

void DeletePoints(std::vector<uchar>& idx, std::vector<std::vector<uchar>>& mvdesc, std::vector<cv::Point2f>& mfeatures)
{
    if(idx.size() != mfeatures.size())
    {
        std::cerr <<"different number of DeletePoints" << std::endl;
        return;
    }

    int indexCorrection = 0;
    int N = idx.size();
    for(int i = 0; i < N; i++)
    {
        if(idx.at(i) == 0)
        {
            mfeatures.erase(mfeatures.begin() + (i-indexCorrection));
            mvdesc.erase(mvdesc.begin() + (i-indexCorrection));
            indexCorrection++;
        }
    }
}