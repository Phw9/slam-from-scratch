#include "Config.h"
#include "Feature.h"
#include "Init.h"
#include "PoseEstimation.h"
#include "opencv2/imgcodecs.hpp"



int main()
{
    std::ofstream rawData ("../main/image.txt", rawData.out | rawData.trunc);
	std::ifstream read ("../main/image.txt", read.in);
	std::ifstream readGTtvec ("../image/GTpose.txt", readGTtvec.in);
    
	std::vector<Eigen::Vector3d> tvecOfGT;


	if(!read.is_open())
	{
		std::cerr << "file can't read image" << std::endl;
		return 0;
	}
	std::deque<std::string> readImageName;
	int imageCurNum = 0;
	MakeTextFile(rawData, IMAGENUM);
	FileRead(readImageName, read);
	GTPoseRead(tvecOfGT, readGTtvec);

	mvo::Feature detector;
	mvo::Feature trackerA1, trackerA2, trackerB1, trackerB2;
	std::vector<mvo::Feature> localATrackPoints;
	int lATP = 0;
	std::vector<mvo::Feature> localBTrackPoints;
	int lBTP = 0;

	
	mvo::StrctureFromMotion getEssential;
	
	std::vector<Eigen::Vector3f> globalRvec;
	std::vector<Eigen::Vector3f> globalTvec;
	int gRTA = 0;
	int gRTB = 0;

	while(imageCurNum < ESSENTIALFRAME+10)
	{
		cv::Mat img;
		img = cv::imread(readImageName.at(imageCurNum), cv::ImreadModes::IMREAD_UNCHANGED);
		if (img.empty())
		{
			std::cerr << "frame upload failed" << std::endl;
		}
		imageCurNum++;
		if(imageCurNum == 1)
		{
			if(!trackerA1.GoodFeaturesToTrack(img))
			{
				std::cout << "new tracker" << std::endl;
			}
		}
		else if(imageCurNum == ESSENTIALFRAME + 10)
		{
			// getEssential.CreateEssentialMatrix(localATrackPoints[gRTA].mfeatures, localATrackPoints[gRTA+lATP-1].mfeatures, intrinsicK);
			// getEssential.GetEssentialRt(getEssential.mEssential, intrinsicK,
			// 							localATrackPoints[gRTA].mfeatures, localATrackPoints[gRTA+lATP-1].mfeatures);
			// std::cout << "getEssential.mTranslation: " << getEssential.mTranslation << std:: endl;

		}else
		{
			localATrackPoints[gRTA+lATP].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-2), cv::ImreadModes::IMREAD_UNCHANGED), img,
													localATrackPoints[gRTA+lATP+1].mfeatures);
			// for(int i = 0; i < lATP; i++)
			// {
			// 	ManageTrackPoints(localATrackPoints[gRTA+lATP],localATrackPoints[gRTA+i]);
			// 	std::cout << "i: " << i << " ";
			// }
			// std::cout << std::endl;
			// std::cout << "lATP: " << lATP << std::endl;
			// lATP++;
		}
	}// 2view SFM, get Essential
	lATP = 0;
	localATrackPoints.clear();

    return 0;
}