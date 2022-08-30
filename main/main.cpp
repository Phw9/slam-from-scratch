#include "Config.h"
#include "Feature.h"
#include "Init.h"
#include "PoseEstimation.h"
#include "KeyFrame.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


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
	cv::Mat img;
	int imageCurNum = 0;
	MakeTextFile(rawData, IMAGENUM);
	FileRead(readImageName, read);
	GTPoseRead(tvecOfGT, readGTtvec);

	mvo::Feature detector;
	mvo::Feature trackerA1, trackerA2, trackerB1, trackerB2;
	std::vector<mvo::Feature> localTrackPointsA;
	localTrackPointsA.reserve(500);
	int lTPA = 0;
	std::vector<mvo::Feature> localTrackPointsB;
	localTrackPointsB.reserve(500);
	int lTPB = 0;

	mvo::StrctureFromMotion getEssential1, getEssential2;
	mvo::PoseEstimation getPose;

	// mvo::KeyFrame keyA, keyB;
	// std::vector<mvo::KeyFrame> globalLandMark;
	// int gLM = 0;

	std::vector<Eigen::Vector3f> globalRvec;
	std::vector<Eigen::Vector3f> globalTvec;
	int gRTA = 0;
	int gRTB = 0;
	

	// Viewer::my_visualize pangolinViewer=Viewer::my_visualize(WINDOWWIDTH, WINDOWHEIGHT);
    // pangolinViewer.initialize();
    // pangolin::OpenGlRenderState s_cam(
    // pangolin::ProjectionMatrix(WINDOWWIDTH, WINDOWHEIGHT, VIEWPOINTF, VIEWPOINTF, 512, 389, 0.1, 1000),
    // pangolin::ModelViewLookAt(VIEWPOINTX, VIEWPOINTY, VIEWPOINTZ, 0, 0, 0, 0.0, -1.0, 0.0));
    // pangolin::View &d_cam = pangolin::CreateDisplay()
    //                             .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -pangolinViewer.window_ratio)
    //                             .SetHandler(new pangolin::Handler3D(s_cam));

	while(true)
	{
		while(imageCurNum < ESSENTIALFRAME)
		{
			
			img = cv::imread(readImageName.at(imageCurNum), 
								cv::ImreadModes::IMREAD_UNCHANGED);
			std::cout << imageCurNum << std::endl;
			if (img.empty())
			{
				std::cerr << "frame upload failed" << std::endl;
			}

			if(imageCurNum == 0)	// feature extract
			{
				if(!trackerA1.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker" << std::endl;
				}
				std::cout << imageCurNum << std::endl;
				localTrackPointsA.emplace_back(std::move(trackerA1));
			}	
			else if(imageCurNum == ESSENTIALFRAME-1)	// 2-viewSFM
			{
				std::cout << imageCurNum << std::endl;
				getEssential1.CreateEssentialMatrix(localTrackPointsA[gRTA+lTPA].mfeatures, localTrackPointsA[gRTA].mfeatures, intrinsicK);
				getEssential1.GetEssentialRt(getEssential1.mEssential, intrinsicK,
											localTrackPointsA[gRTA].mfeatures, localTrackPointsA[gRTA+lTPA-1].mfeatures);
				gRTA = imageCurNum;											
			}
			else	// tracking
			{
				std::cout << imageCurNum << std::endl;
				localTrackPointsA[gRTA+lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA1);
				localTrackPointsA.emplace_back(std::move(trackerA1));
				
				// cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
				// img = pangolinViewer.cv_draw_features(img, localTrackPointsA.at(gRTA+lTPA).mfeatures, localTrackPointsA.at(gRTA+lTPA+1).mfeatures);
				
				for(int i = 0; i < lTPA; i++)
				{
					ManageTrackPoints(localTrackPointsA.at(gRTA+lTPA+1), localTrackPointsA.at(gRTA+i));
					std::cout << "i: " << i << " ";
				}
				std::cout << std::endl;
				std::cout << "lTPA: " << lTPA << std::endl;
				lTPA++;
			}
			imageCurNum++;
			cv::imshow("img", img);
			if(cv::waitKey(0) == 27) break; // ESC key
		} // 2view SFM, get Essential

		// while(ESSENTIALFRAME < imageCurNum < 2*ESSENTIALFRAME)
		// {
		// 	img = cv::imread(readImageName.at(imageCurNum), 
		// 						cv::ImreadModes::IMREAD_UNCHANGED);
			
		// 	if (img.empty())
		// 	{
		// 		std::cerr << "frame upload failed" << std::endl;
		// 	}

		// 	if(imageCurNum == 0)	// feature extract
		// 	{
		// 		if(!trackerA1.GoodFeaturesToTrack(img))
		// 		{	
		// 			std::cout << "new tracker" << std::endl;
		// 		}
		// 		localTrackPointsA.emplace_back(std::move(trackerA1));
		// 	}	
		// 	else if(imageCurNum == ESSENTIALFRAME-1)	// 2-viewSFM
		// 	{
		// 		getEssential2.CreateEssentialMatrix(localTrackPointsA[gRTA+lTPA].mfeatures, localTrackPointsA[gRTA].mfeatures, intrinsicK);
		// 		getEssential2.GetEssentialRt(getEssential1.mEssential, intrinsicK,
		// 									localTrackPointsA[gRTA].mfeatures, localTrackPointsA[gRTA+lTPA-1].mfeatures);
		// 		gRTA = imageCurNum;											
		// 	}
		// 	else	// tracking
		// 	{
		// 		localTrackPointsA[gRTA+lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
		// 											cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA1);
		// 		localTrackPointsA.emplace_back(std::move(trackerA1));
				
		// 		cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		// 		img = pangolinViewer.cv_draw_features(img, localTrackPointsA.at(gRTA+lTPA).mfeatures, localTrackPointsA.at(gRTA+lTPA+1).mfeatures);
				
		// 		for(int i = 0; i < lTPA; i++)
		// 		{
		// 			ManageTrackPoints(localTrackPointsA.at(gRTA+lTPA+1), localTrackPointsA.at(gRTA+i));
		// 			std::cout << "i: " << i << " ";
		// 		}
		// 		std::cout << std::endl;
		// 		std::cout << "lTPA: " << lTPA << std::endl;
		// 		lTPA++;
		// 	}
		// 	imageCurNum++;
		// 	cv::imshow("img", img);
		// 	if(cv::waitKey(0) == 27) break;	// ESC key
		// }	// 2view SFM, get Essential


		// lTPA = 0;
		// localTrackPointsA.clear();
		// gRTA = imageCurNum;





		cv::Mat img;
		img = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);
		
		imageCurNum++;
		cv::imshow("img", img);
		if(cv::waitKey(10) == 27) break; // ESC key
	}

    return 0;
}

