#pragma once

#include <vector>
#include "opencv2/core.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>
#include "thirdparty/g2o/config.h"
#include "thirdparty/g2o/g2o/types/se3quat.h"
#include "thirdparty/g2o/g2o/types/sim3.h"

namespace mvo
{
    class Converter
    {
    public:
    Converter();
    ~Converter()=default;

    std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);
    g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
    cv::Mat toCvMat(const g2o::SE3Quat &SE3);
    cv::Mat toCvMat(const g2o::Sim3 &Sim3);
    cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    cv::Mat toCvMat(const Eigen::Matrix3d &m);
    cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
    cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
    Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
    Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
    Eigen::Matrix<double,2,1> toVector2d(const cv::Mat &cvVector);
    Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
    std::vector<float> toQuaternion(const cv::Mat &M);
    };
} // 
