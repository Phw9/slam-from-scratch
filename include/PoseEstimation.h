#pragma once

#include "Feature.h"
#include "opencv2/calib3d.hpp"

namespace mvo
{
    class StrctureFromMotion
    {
    public:
        StrctureFromMotion();

        bool CreateEssentialMatrix(const std::vector<cv::Vec2f>& pts1, const std::vector<cv::Vec2f>& pts2, const cv::InputArray& K);
        bool GetEssentialRt(const cv::InputArray& E, const cv::InputArray& K, const std::vector<cv::Vec2f>& pts1, const std::vector<cv::Vec2f>& pts2);
        bool CombineRt();
    public:
        cv::Mat mEssential;
        cv::Mat mRotation;
        cv::Mat mTranslation;
        cv::Mat mCombineRt;
    };
}//namespace mvo

