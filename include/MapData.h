#pragma once

#include <algorithm>
#include "ceres/ceres.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "Triangulate.h"
#include "PoseEstimation.h"

#include "DBoW2/DBoW2.h"

#define DISTANCEDESC 20 // 20
#define REPROJECTERROR 2 // 2

namespace mvo
{
    class MapData
    {
    public:
        MapData();
        ~MapData()=default;
        
        bool GetSFMPose(const mvo::StrctureFromMotion& sfm);
        bool GetPnPPose(const mvo::PoseEstimation& pe);
        bool Get2DPoints(const mvo::Feature& feature);
        bool Get3DPoints(const mvo::Triangulate& tr);

        std::vector<cv::Point2f> mpoint2D;
        std::vector<cv::Point3f> mpoint3D;
        cv::Mat mglobalRTMat;
        cv::Mat mglobalRMat;
        cv::Vec3d mglobalrvec;
        cv::Vec3d mglobaltvec;
        cv::Vec3d mglobalTranslation;
        cv::Mat mdesc;
        std::vector<std::vector<DTYPE>> mvdesc;
        std::vector<cv::Mat> mvecdesc;
        std::vector<int> mindex;
        cv::Mat minlier;
    };

    class Covisibilgraph
    {
    public:
        Covisibilgraph()=default;
        Covisibilgraph(const std::vector<mvo::MapData>& v,
                        const double& focal, const double& ppx, 
                                            const double& ppy);
        ~Covisibilgraph()=default;
        
        template <typename T> void MakeEdgeProj(int gD, std::vector<mvo::MapData>& mapdata);
        void MakeEdgeDesc(int gD, mvo::Feature& before, mvo::Triangulate& mapPoints);
        void CullingNode(int gD);
        bool Projection(const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                        const cv::Point2f& pts, const cv::Point3f& world);

        const std::vector<mvo::MapData>& mglobalMapData;
        // vector < pair <(gD-1)index,(gD)index>
        std::vector<std::vector<std::pair<int, int>>> mgraphreproj;
        std::vector<std::vector<std::vector<std::pair<int, int>>>> mglobalgraph;
        std::vector<std::vector<std::pair<int, int>>> mgraph;
        const double& mfocal;
        const double& mppx;
        const double& mppy;
    };

    void PushData(std::vector<mvo::MapData>& v, const mvo::MapData& md);
    void GetLocalPose(std::vector<cv::Mat>& v, const cv::Mat& m);

}

