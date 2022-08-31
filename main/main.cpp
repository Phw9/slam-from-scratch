#include "Config.h"
#include "Feature.h"
#include "Init.h"
#include "PoseEstimation.h"
#include "Triangulate.h"
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
	int imageRealFrame = 0;

	MakeTextFile(rawData, IMAGENUM);
	FileRead(readImageName, read);
	GTPoseRead(tvecOfGT, readGTtvec);

	mvo::Feature detector;
	mvo::Feature trackerA, trackerB;
	std::vector<mvo::Feature> localTrackPointsA;
	localTrackPointsA.reserve(500);
	int lTPA = 0;
	std::vector<mvo::Feature> localTrackPointsB;
	localTrackPointsB.reserve(500);
	int lTPB = 0;
	mvo::StrctureFromMotion getEssential1, getEssential2;
	
	mvo::PoseEstimation getPose;
	mvo::Triangulate keyA, keyB;
	std::vector<mvo::Triangulate> globalLandMark;
	int gLM = 0;
	std::vector<int> gKF;

	std::vector<cv::Mat> globalRTMat;
	std::vector<cv::Vec3d> globalRVec;
	std::vector<cv::Vec3d> globalTVec;
	int gP = 0;
	
	Viewer::my_visualize pangolinViewer=Viewer::my_visualize(WINDOWWIDTH, WINDOWHEIGHT);
    pangolinViewer.initialize();
    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(WINDOWWIDTH, WINDOWHEIGHT, VIEWPOINTF, VIEWPOINTF, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(VIEWPOINTX, VIEWPOINTY, VIEWPOINTZ, 0, 0, 0, 0.0, -1.0, 0.0));
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -pangolinViewer.window_ratio)
                                .SetHandler(new pangolin::Handler3D(s_cam));

	while(true)
	{
		while(imageRealFrame < 2*ESSENTIALFRAME)
		{
			
			img = cv::imread(readImageName.at(imageCurNum), 
								cv::ImreadModes::IMREAD_UNCHANGED);
			if (img.empty())
			{
				std::cerr << "frame upload failed" << std::endl;
			}

			if(imageRealFrame == 0)	// feature extract
			{
				if(!trackerA.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker A" << std::endl;
				}
				std::cout << imageCurNum << std::endl;
				localTrackPointsA.emplace_back(std::move(trackerA));
			}	
			else if(imageRealFrame == ESSENTIALFRAME-1)	// 2-viewSFM(1)
			{
				imageCurNum--;
				getEssential1.CreateEssentialMatrix(localTrackPointsA[0].mfeatures, localTrackPointsA[lTPA].mfeatures, intrinsicK);
				getEssential1.GetEssentialRt(getEssential1.mEssential, intrinsicK,
											localTrackPointsA[0].mfeatures, 
											localTrackPointsA[lTPA-1].mfeatures);
				
				getEssential1.GetRTvec();
				getEssential1.CombineRt();
				globalRTMat.emplace_back(std::move(getEssential1.mCombineRt));
				globalRVec.emplace_back(std::move(getEssential1.mrvec));
				globalTVec.emplace_back(std::move(getEssential1.mtvec));
				gP++;
				gKF.emplace_back(imageCurNum);
				std::cout << globalRTMat[gP-1] << std::endl;
				std::cout << globalRVec[gP-1] << std::endl;
				std::cout << globalTVec[gP-1] << std::endl;
				std::cout << "lTPA: " << lTPA << " imagenum: " << imageCurNum << std::endl;
				std::cout << "gKF: " << gKF[gP-1] << std::endl;
			}else if(imageRealFrame == 2*ESSENTIALFRAME-1)	// 2-viewSFM(2)
			{
				imageCurNum--;
				// 2-view SFM
				getEssential1.CreateEssentialMatrix(localTrackPointsA[ESSENTIALFRAME-2].mfeatures, localTrackPointsA[lTPA].mfeatures, intrinsicK);
				getEssential1.GetEssentialRt(getEssential1.mEssential, intrinsicK,
											localTrackPointsA[ESSENTIALFRAME-2].mfeatures, 
											localTrackPointsA[lTPA-1].mfeatures);
				getEssential1.GetRTvec();
				getEssential1.CombineRt();
				globalRTMat.emplace_back(std::move(getEssential1.mCombineRt));
				globalRVec.emplace_back(std::move(getEssential1.mrvec));
				globalTVec.emplace_back(std::move(getEssential1.mtvec));
				gP++;
				gKF.emplace_back(imageCurNum);
				
				// Triangulate Landmark
				keyA.CalcWorldPoints(globalRTMat[gP-2],globalRTMat[gP-1],
							localTrackPointsA[gKF[0]].mfeatures, localTrackPointsA[gKF[1]].mfeatures);
				keyA.ScalingPoints();
				globalLandMark.emplace_back(std::move(keyA));
				gLM++;

				// create Track B
				if(!trackerB.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker B" << std::endl;
				}
				localTrackPointsB.emplace_back(std::move(trackerB));

				std::cout << "TrackB size: " << localTrackPointsB[lTPB].mfeatures.size() << std::endl;
				std::cout << "LandMark size : " << globalLandMark[gLM-1].mworldMapPoints.size() << std::endl;
				std::cout << "local points size: " << localTrackPointsA[lTPA].mfeatures.size() << std::endl;

				std::cout << globalRTMat[gP-1] << std::endl;
				std::cout << globalRVec[gP-1] << std::endl;
				std::cout << globalTVec[gP-1] << std::endl;
				std::cout << "lTPA: " << lTPA << " imagenum: " << imageCurNum << std::endl;
				std::cout << "gKF: " << gKF[gP-1] << std::endl;
				// std::cout << globalLandMark[gLM-1].mworldMapPoints << std::endl;
			}
			else	// tracking
			{
				localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA);
				localTrackPointsA.emplace_back(std::move(trackerA));
				
				cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
				img = pangolinViewer.cv_draw_features(img, localTrackPointsA.at(lTPA).mfeatures, 
													localTrackPointsA.at(lTPA+1).mfeatures);
				
				for(int i = 0; i < lTPA; i++)
				{
					ManageTrackPoints(localTrackPointsA.at(lTPA+1), localTrackPointsA.at(i));
					std::cout << "i: " << i << " ";
				}
				std::cout << std::endl;
				std::cout << "lTPA: " << lTPA << std::endl;
				lTPA++;
			}
			imageCurNum++;
			imageRealFrame++;
			cv::imshow("img", img);
			if(cv::waitKey(0) == 27) break; // ESC key
		} // 2view SFM, Track(A,B) make, generate LandMark = current localFeature size


		// start SolvePnP

		img = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);
		
		// if(keyframe gen){ generate MapPoints, Pose, new Track}

		// if num of Feature is less than NUMOFPOINTS, GFTT
		if(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS)
		{
			if(!trackerA.GoodFeaturesToTrack(img))
			{	
				std::cout << "new tracker A" << std::endl;
			}
			std::cout << imageCurNum << std::endl;
			localTrackPointsA.clear();
			localTrackPointsA.emplace_back(std::move(trackerA));
			lTPA = 0;
		}
		if(localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS)
		{
			if(!trackerB.GoodFeaturesToTrack(img))
			{	
				std::cout << "new tracker A" << std::endl;
			}
			std::cout << imageCurNum << std::endl;
			localTrackPointsB.clear();
			localTrackPointsB.emplace_back(std::move(trackerB));
			lTPB = 0;
		}
		// solvePnP how to match number of globalLandMark, localTrackpoints
		// getPose.solvePnP(globalLandMark[gLM-1].mworldMapPoints, localTrackPointsA[lTPA].mfeatures, intrinsicK);
		// getPose.solvePnP(globalLandMark[gLM-1].mworldMapPoints, localTrackPointsB[lTPB].mfeatures, intrinsicK);


		// tracking

		localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA);
				localTrackPointsA.emplace_back(std::move(trackerA));
		for(int i = 0; i < lTPA; i++)
		{
			ManageTrackPoints(localTrackPointsA.at(lTPA+1), localTrackPointsA.at(i));
			std::cout << "i: " << i << " ";
		}
		std::cout << std::endl;
		std::cout << "lTPA: " << lTPA << std::endl;
		std::cout << "lTPA size : " << localTrackPointsA[lTPA].mfeatures.size() << std::endl;
		lTPA++;
		
		
		localTrackPointsB[lTPB].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerB);
		localTrackPointsB.emplace_back(std::move(trackerB));
		for(int i = 0; i < lTPB; i++)
		{
			ManageTrackPoints(localTrackPointsB.at(lTPB+1), localTrackPointsB.at(i));
			std::cout << "i: " << i << " ";
		}
		std::cout << std::endl;
		std::cout << "lTPB: " << lTPB << std::endl;
		std::cout << "lTPB size : " << localTrackPointsB[lTPB].mfeatures.size() << std::endl << std::endl;
		lTPB++;


		// draw tracking points
		if(localTrackPointsA[lTPA].mfeatures.size() > localTrackPointsB[lTPB].mfeatures.size())
		{
		cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		img = pangolinViewer.cv_draw_features(img, localTrackPointsA.at(lTPA-1).mfeatures, 
											localTrackPointsA.at(lTPA).mfeatures);
		}
		else
		{
		cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		img = pangolinViewer.cv_draw_features(img, localTrackPointsB.at(lTPB-1).mfeatures, 
											localTrackPointsB.at(lTPB).mfeatures);		
		}

		imageCurNum++;
		imageRealFrame++;
		cv::imshow("img", img);
		if(cv::waitKey(0) == 27) break; // ESC key
	}

    return 0;
}

