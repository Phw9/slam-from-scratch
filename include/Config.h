#pragma once

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

#define IMAGENUM 4540
#define ESSENTIALFRAME 6
#define WINDOWWIDTH 1024
#define WINDOWHEIGHT 768
#define VIEWPOINTF 20.0
#define VIEWPOINTX 0.0
#define VIEWPOINTY -40.0
#define VIEWPOINTZ -0.1

#define NUMOFPOINTS 200 // 200

#define ANGULARVELOCITY 0.1 // 0.1
#define NUMOFINLIER 200    // 200
#define INLIERRATIO 0.70    //0.70 
#define BUNDLE 1 // 1 = BA, 2 = off

int initialNum = 1; // 1 = auto, 2 = default
float inlierRatio = 1000.0f;
double angularVelocity = 0.0f;

float camXf = 6.018873000000e+02; double camXd = 6.018873000000e+02;
float cameraYf = 1.831104000000e+02; double camYd = 1.831104000000e+02;
float focalLengthf = 7.070912000000e+02; double focalLengthd = 7.070912000000e+02;
float fdata[] = {focalLengthf, 0, camXf,
                0, focalLengthf, cameraYf,
                0, 0, 1};
double data[] = {focalLengthd, 0, camXd,
                0, focalLengthd, camYd,
                0, 0, 1};
static cv::Mat intrinsicKf(cv::Size(3, 3), CV_32FC1, fdata);
static cv::Mat intrinsicKd(cv::Size(3, 3), CV_64FC1, data);

static cv::Mat m1 = cv::Mat::eye(cv::Size(4,3), CV_64F);
static cv::Mat m2 = cv::Mat::eye(cv::Size(4,3), CV_64F);
