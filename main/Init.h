#pragma once

#include <deque>
#include <fstream>
#include <string>
#include "Config.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <pangolin/display/display.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/image/image_utils.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/viewport.h>
#include <pangolin/utils/range.h>

#include <functional>
#include <stdio.h>
#include <iostream>



void FileRead(std::deque<std::string>& v, std::ifstream &fin);
void MakeTextFile(std::ofstream& fout, const int& imageNum);
void GTPoseRead(std::vector<cv::Vec3f>& v, std::ifstream& fin);


namespace Viewer
{
    class my_visualize
	{
        private:
            int window_width;
            int window_height;

        public:
            float window_ratio;

            //생성자
            my_visualize(int width,int height);
            void initialize();
            void active_cam();

            // pts1: GT Pose, pts2: Pose, pts3: 3D Points, pts4: FOV of 3D Points
            void draw_point(std::vector<cv::Mat>& pose, std::vector<cv::Vec3f>& gtPose, std::vector<cv::Mat>& allOfPoints, cv::Mat fovPoints);

            // circle is before, rectangle is after
            cv::Mat cv_draw_features(cv::Mat src, std::vector<cv::Point2f>& beforePoints, std::vector<cv::Point2f> afterPoints);
    };
}