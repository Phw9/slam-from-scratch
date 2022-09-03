#pragma once

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

#define IMAGENUM 4540
#define ESSENTIALFRAME 6
#define WINDOWWIDTH 1024
#define WINDOWHEIGHT 768
#define VIEWPOINTF 2000.0
#define VIEWPOINTX 0.0
#define VIEWPOINTY -10.0
#define VIEWPOINTZ -0.1

#define NUMOFPOINTS 200

float cameraX = 6.018873000000e+02;
float cameraY = 1.831104000000e+02;
float focalLength = 7.070912000000e+02;
float data[] = {focalLength, 0, cameraX,
                0, focalLength, cameraY,
                0, 0, 1};
static cv::Mat intrinsicK(cv::Size(3, 3), CV_32FC1, data);

// cv::Mat mm = cv::Mat::zeros(cv::Size(3,3), CV_64F);
// cv::Mat mv = cv::Mat(cv::Size(3,1),CV_64F,1);
// m.push_back(mv);
// mm = mm.t();
