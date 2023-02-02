#pragma once

#include "opencv2/core.hpp"
#include "ceres/loss_function.h"
#include "Triangulate.h"
#include "MapData.h"

#define LOCAL 10 // 10
#define LOSS 1 // 1 = Trivial, 2 = Cauchy

static double camX = 6.018873000000e+02;
static double camY = 1.831104000000e+02;
static double focald = 7.070912000000e+02;
static double reprojerr = 2.0;


namespace mvo
{
    struct SnavelyReprojectionError
    {
        SnavelyReprojectionError(double observed_x, double observed_y, Eigen::Vector4d worldHomoGen4d, 
                                 double focal, double ppx, double ppy);
        ~SnavelyReprojectionError() = default;

        template <typename T> bool operator()(const T *const rvec_eig,
                                              const T *const tvec_eig,
                                              T *residuals) const;

        double observed_x;
        double observed_y;
        const Eigen::Vector4d worldHomoGen4d;
        double focal;
        double ppx;
        double ppy;
    };

    struct SnavelyReprojectionErrorLocal
    {
        SnavelyReprojectionErrorLocal(double observed_x, double observed_y,
                                      double focal, double ppx, double ppy);
        ~SnavelyReprojectionErrorLocal() = default;

        template <typename T>
        bool operator()(const T *const rvec_eig,
                        const T *const tvec_eig,
                        const T *const point_3d_homo_eig,
                        T *residuals) const;

        double observed_x;
        double observed_y;
        double focal;
        double ppx;
        double ppy;
    };

    struct SnavelyReprojectionErrorLocalPoseFixed
    {
        SnavelyReprojectionErrorLocalPoseFixed(double observed_x, double observed_y, 
                                                  double focal, double ppx, double ppy, 
                                            Eigen::Vector3d rvec_eig, Eigen::Vector3d tvec_eig);
        ~SnavelyReprojectionErrorLocalPoseFixed() = default;

        template <typename T>
        bool operator()(const T *const worldPoint,
                        T *residuals) const;

        double observed_x;
        double observed_y;
        double focal;
        double ppx;
        double ppy;
        Eigen::Vector3d rvec_eig;
        Eigen::Vector3d tvec_eig;
    };

    class BundleAdjustment
    {
    public:
    BundleAdjustment()=default;
    BundleAdjustment(mvo::Feature& ft, mvo::PoseEstimation& rt,
                     mvo::Triangulate& tri);               
    BundleAdjustment(mvo::MapData& data);
    ~BundleAdjustment() = default;

    bool MotionOnlyBA();
    bool LocalBA(int gD, std::vector<mvo::MapData>& map, mvo::Covisibilgraph& cov);
    bool FullBA();

    mvo::Feature ft;
    mvo::PoseEstimation rt;
    mvo::Triangulate tri;
    mvo::MapData data;
    };
}

bool Projection(const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                const cv::Point2f& pts, const cv::Point3f& world);