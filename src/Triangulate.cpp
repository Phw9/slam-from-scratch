#include "Triangulate.h"

mvo::Triangulate::Triangulate(): mworldMapPoints{cv::Mat()}
{
    mworldMapPointsV.clear();
};

bool mvo::Triangulate::CalcWorldPoints(const cv::Mat& pose1,
                                        const cv::Mat& pose2,
                                        const std::vector<cv::Point2f>& pts1,
                                        const std::vector<cv::Point2f>& pts2)
{
    cv::triangulatePoints(pose1, pose2, pts1, pts2, mworldMapPoints);
    
    if(mworldMapPoints.empty())
    {
        std::cerr << "failed triagulatePoints" << std::endl;
        return false;
    }

    return true;
}

bool mvo::Triangulate::ScalingPoints()
{
    for(int i = 0; i < mworldMapPoints.cols; i++)
    {
        for(int j = 0; j < mworldMapPoints.rows; j++)
        {
            mworldMapPoints.at<float>(j,i) = mworldMapPoints.at<float>(j,i) / mworldMapPoints.at<float>(mworldMapPoints.rows-1,i);
        }
    }
    if(mworldMapPoints.at<float>(mworldMapPoints.rows-1,0) != 1.0f) return false;
    return true;
}

void mvo::Triangulate::MatToPoints3f()
{
    cv::Point3f temp;
    for (int i = 0; i < mworldMapPoints.cols; i++) 
    {
		temp.x = mworldMapPoints.at<float>(0, i);
        temp.y = mworldMapPoints.at<float>(1, i);
        temp.z = mworldMapPoints.at<float>(2, i);
        mworldMapPointsV.emplace_back(std::move(temp));
        // std::cout << temp.x << " " << temp.y << " " << temp.z << std::endl;
	}
}