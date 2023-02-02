#pragma once
/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

// DBoW2 // defines OrbVocabulary and OrbDatabase
#include "MapData.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

namespace mvo
{
    void testDatabase(const std::vector<std::vector<cv::Mat>> &globaldesc, std::vector<mvo::MapData> map, OrbDatabase& db);
    void wait();
// ----------------------------------------------------------------------------
    void LoopDetectCompute(const cv::Mat& img, std::vector<std::vector<cv::Mat>>& globaldesc, OrbDatabase& db);

// ----------------------------------------------------------------------------
    void VocCreation(const std::vector<std::vector<cv::Mat>>& features);

// ----------------------------------------------------------------------------
    void testDatabase(const std::vector<std::vector<cv::Mat>> &globaldesc, std::vector<mvo::MapData> map, OrbDatabase& db);

    // ----------------------------------------------------------------------------
};
void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out);

