#pragma once

#include <deque>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "eigen3/Eigen/Dense"
#include "Triangulate.h"
#include "MapData.h"

#include <pangolin/display/display.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/image/image_utils.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/viewport.h>
#include <pangolin/utils/range.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void FileRead(std::deque<std::string>& v, std::ifstream &fin);
void MakeTextFile(std::ofstream& fout, const int& imageNum);
void GTPoseRead(std::vector<cv::Vec3f>& v, std::ifstream& fin);



namespace Viewer
{

    class MyVisualize
	{
        private:
            int window_width;
            int window_height;

        public:
            float window_ratio;

            //생성자
            MyVisualize(int width,int height);
            void initialize();
            void active_cam();

            // pts1: GT Pose, pts2: Pose, pts3: 3D Points, pts4: FOV of 3D Points
            void DrawPoint(const std::vector<cv::Vec3d>& tvec, 
                            const std::vector<cv::Vec3f>& gtPose,
                            const std::vector<mvo::Triangulate>& allOfPoints, 
                            const std::vector<cv::Point3f>& fovPoints);
            void DrawPoint(const std::vector<mvo::MapData>& v, const std::vector<cv::Vec3f>& gtPose);
            // circle is before, rectangle is after
            cv::Mat DrawFeatures(cv::Mat& src, std::vector<cv::Point2f>& beforePoints, std::vector<cv::Point2f>& afterPoints);
    };
}//namespace Viewer

namespace mvo
{
    class Initializer
    {
    public:
        Initializer();
        // Fix the reference frame
        Initializer(std::vector<cv::KeyPoint> refKeys1, std::vector<cv::KeyPoint> refKeys2);
        ~Initializer()=default;

        float CheckHomography(const std::vector<cv::Point2f>& refKeys1, const std::vector<cv::Point2f>& refKeys2, float sigma);
        float CheckFundamental(const std::vector<cv::Point2f>& refKeys1, const std::vector<cv::Point2f>& refKeys2, float sigma);
        void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

        cv::Mat mEssential;
        cv::Mat mFundamental;
        cv::Mat mHomography;
        cv::Mat mInvHomography;
        cv::Vec3d mrvec;
        cv::Vec3d mtvec;
        // Keypoints from Reference Frame (Frame 1)
        std::vector<cv::KeyPoint> mvKeys1;

        // Keypoints from Current Frame (Frame 2)
        std::vector<cv::KeyPoint> mvKeys2;

        // Calibration
        cv::Mat mK;

        // Standard Deviation and Variance
        float mSigma, mSigma2;

        // Ransac max iterations
        int mMaxIterations;
    };
}