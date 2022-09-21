#pragma once

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

#define IMAGENUM 4540
#define ESSENTIALFRAME 5
#define WINDOWWIDTH 1024
#define WINDOWHEIGHT 768
#define VIEWPOINTF 20.0
#define VIEWPOINTX 0.0
#define VIEWPOINTY -50.0
#define VIEWPOINTZ -0.1

#define NUMOFPOINTS 200

#define ANGULARVELOCITY 0.055
#define NUMOFINLIER 150
#define INLIERRATIO 0.48
#define MINLOCAL 2

float cameraXf = 6.018873000000e+02; double cameraXd = 6.018873000000e+02;
float cameraYf = 1.831104000000e+02; double cameraYd = 1.831104000000e+02;
float focalLengthf = 7.070912000000e+02; double focalLengthd = 7.070912000000e+02;
float fdata[] = {focalLengthf, 0, cameraXf,
                0, focalLengthf, cameraYf,
                0, 0, 1};
double data[] = {focalLengthd, 0, cameraXd,
                0, focalLengthd, cameraYd,
                0, 0, 1};
static cv::Mat intrinsicKf(cv::Size(3, 3), CV_32FC1, fdata);
static cv::Mat intrinsicKd(cv::Size(3, 3), CV_64FC1, data);
