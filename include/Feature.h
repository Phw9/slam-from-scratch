#pragma once

#include <vector>
#include <chrono>
#include "eigen3/Eigen/Dense"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/video.hpp"
#include "opencv2/core.hpp"


#define DTYPE uchar
#define ERRORSIZE 20 // 20
#define MAXCORNER 1500 // window = 1000 ubuntu = 1500
#define QUALITYLEVEL 0.001 // 0.001
#define MINDISTANCE 15 // window = 9 ubuntu = 15


namespace mvo
{
    class Feature
    {
    public:
        Feature();
        ~Feature()=default;
        
        bool CornerFAST(const cv::Mat& src);
        bool GoodFeaturesToTrack(const cv::Mat& src);
        bool OpticalFlowPyrLK(const cv::Mat& src1, const cv::Mat& src2, mvo::Feature& next);
        std::vector<uchar> mstatus;
        std::vector<uchar> mdelete;
        std::vector<float> merr;
        std::vector<cv::Point2f> mfeatures;
        std::vector<std::vector<DTYPE>> mvdesc;
        cv::Mat mdesc;
    };
}//namespace mvo

void ManageTrackPoints(const mvo::Feature& present, mvo::Feature& before);
bool KeyPointToVec(const std::vector<cv::KeyPoint>& kp, std::vector<cv::Point2f>& features2d);
bool VecToKeyPoint(const std::vector<cv::Point2f>& features2d, std::vector<cv::KeyPoint>& kp);
bool MatToVec(const cv::Mat& m, std::vector<std::vector<DTYPE>>& v);
bool VecToMat(const std::vector<std::vector<uchar>>& v, cv::Mat& m);
bool VecToMat(const std::vector<std::vector<DTYPE>>& v, cv::Mat& m);
std::vector<uchar> FindDeletePoints(std::vector<cv::KeyPoint>& kp, 
                                    std::vector<cv::Point2f>& mfeatures);
void DeletePoints(std::vector<uchar>& idx, std::vector<std::vector<uchar>>& mvdesc, std::vector<cv::Point2f>& mfeatures);