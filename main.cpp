#include "Config.h"
#include "Feature.h"
#include "Init.h"
#include "PoseEstimation.h"
#include "Triangulate.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


int main()
{
    std::ofstream rawData ("./image.txt", rawData.out | rawData.trunc);
	std::ifstream read ("./image.txt", read.in);
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
	int realFrame = 0;

	MakeTextFile(rawData, IMAGENUM);
	FileRead(readImageName, read);
	GTPoseRead(tvecOfGT, readGTtvec);

	mvo::Feature detector;
	mvo::Feature trackerA, trackerB;
	std::vector<mvo::Feature> localTrackPointsA;
	localTrackPointsA.reserve(6000);
	int lTPA = 0;
	std::vector<mvo::Feature> localTrackPointsB;
	localTrackPointsB.reserve(6000);
	int lTPB = 0;

	mvo::StrctureFromMotion getEssential;
	
	mvo::PoseEstimation getPose;
	mvo::Triangulate mapPointsA, mapPointsB;
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
		while(realFrame < 2*ESSENTIALFRAME)
		{
			img = cv::imread(readImageName.at(imageCurNum), 
								cv::ImreadModes::IMREAD_UNCHANGED);
			if (img.empty())
			{
				std::cerr << "frame upload failed" << std::endl;
			}

			if(realFrame == 0)	// feature extract
			{
				if(!trackerA.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker A" << std::endl;
				}
				localTrackPointsA.emplace_back(std::move(trackerA));
			}	
			else if(realFrame == ESSENTIALFRAME-1)	// 2-viewSFM(1)
			{
				imageCurNum--;
				getEssential.CreateEssentialMatrix(localTrackPointsA[0].mfeatures, localTrackPointsA[lTPA].mfeatures, intrinsicK);
				getEssential.GetEssentialRt(getEssential.mEssential, intrinsicK,
											localTrackPointsA[0].mfeatures, 
											localTrackPointsA[lTPA].mfeatures);
				getEssential.GetRTvec();
				getEssential.CombineRt();
				globalRTMat.emplace_back(std::move(getEssential.mCombineRt));
				globalRVec.emplace_back(std::move(getEssential.mrvec));
				globalTVec.emplace_back(std::move(getEssential.mtvec));
				gP++;
				gKF.emplace_back(imageCurNum);
				std::cout << globalRTMat[gP-1] << std::endl;
				std::cout << globalRVec[gP-1] << std::endl;
				std::cout << globalTVec[gP-1] << std::endl;
				std::cout << "lTPA: " << lTPA << " imagenum: " << imageCurNum << "realFrame: " << realFrame << std::endl;
				std::cout << "gKF: " << gKF[gP-1] << std::endl;
			}
			else if(realFrame == 2*ESSENTIALFRAME-1)	// 2-viewSFM(2)
			{
				// 2-view SFM
				imageCurNum--;
				getEssential.CreateEssentialMatrix(localTrackPointsA[ESSENTIALFRAME-2].mfeatures, localTrackPointsA[lTPA].mfeatures, intrinsicK);
				getEssential.GetEssentialRt(getEssential.mEssential, intrinsicK,
											localTrackPointsA[ESSENTIALFRAME-2].mfeatures, 
											localTrackPointsA[lTPA].mfeatures);
				getEssential.GetRTvec();
				getEssential.CombineRt();
				globalRTMat.emplace_back(std::move(getEssential.mCombineRt));
				globalRVec.emplace_back(std::move(getEssential.mrvec));
				globalTVec.emplace_back(std::move(getEssential.mtvec));
				gP++;
				gKF.emplace_back(imageCurNum);
				
				cv::Mat mm = cv::Mat::zeros(cv::Size(3,3), CV_64F);
				cv::Mat mv = cv::Mat(cv::Size(3,1),CV_64F,1);
				mm.push_back(mv);
				mm = mm.t();
				std::cout << "mm: " << mm << std::endl;

				// Triangulate Landmark
				mapPointsA.CalcWorldPoints(globalRTMat[gP-2], globalRTMat[gP-1],
							localTrackPointsA[gKF[0]].mfeatures, localTrackPointsA[gKF[1]].mfeatures);
				mapPointsA.ScalingPoints();
				mapPointsA.MatToPoints3d();
				globalLandMark.emplace_back(mapPointsA);
				gLM++;

				// create Track B
				if(!trackerB.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker B" << std::endl;
				}
				localTrackPointsB.emplace_back(std::move(trackerB));
			}
			else	// tracking
			{
				localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA);
				localTrackPointsA.emplace_back(std::move(trackerA));
				lTPA++;
				std::cout << lTPA << std::endl;
				cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
				img = pangolinViewer.cv_draw_features(img, localTrackPointsA.at(lTPA-1).mfeatures, 
													localTrackPointsA.at(lTPA).mfeatures);
				for(int i = 0; i < lTPA-1; i++)
				{
					ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i), mapPointsA.mworldMapPointsV);
					std::cout << i << "size : " << localTrackPointsA.at(i).mfeatures.size() << " ";
				}													
			}
			imageCurNum++;
			realFrame++;
			cv::imshow("img", img);
			if(cv::waitKey(0) == 27) break; // ESC key
		} // 2view SFM, Track(A,B) make, generate LandMark = current localFeature size

		// Start

		img = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);
		
		// if(keyframe gen){ generate Pose, MapPoints, new Track}

		// if num of Feature is less than NUMOFPOINTS, GFTT
		if(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS)
		{
			getPose.solvePnP(mapPointsA.mworldMapPointsV, 
							localTrackPointsA[lTPA].mfeatures, intrinsicK);
			getPose.GetRMatTPose();
			getPose.CombineRt();
			globalRTMat.emplace_back(std::move(getPose.mCombineRt));
			globalRVec.emplace_back(std::move(getPose.mrvec));
			globalTVec.emplace_back(std::move(getPose.mtvec));
			gP++;

			// triangulate
			mapPointsA.CalcWorldPoints(globalRTMat.at(gP-2), globalRTMat.at(gP-1), 
									localTrackPointsB.at(0).mfeatures, localTrackPointsB.at(lTPB).mfeatures);
			mapPointsA.ScalingPoints();
			mapPointsA.MatToPoints3d();
			globalLandMark.emplace_back(mapPointsA);
			gLM++;
			std::cout << "hello" << std::endl;
			localTrackPointsA.clear();
			std::cout << "hello" << std::endl;
			if(!trackerA.GoodFeaturesToTrack(img))
			{	
				std::cout << "new tracker A" << std::endl;
			}
			std::cout << "hello" << std::endl;
			std::cout << "localPointsA: " <<localTrackPointsA.at(lTPA).mfeatures.size() <<std::endl;
			std::cout << "local A size : " << localTrackPointsA.size() << std::endl;
			localTrackPointsA.emplace_back(std::move(trackerA));
			lTPA = 1;
			imageCurNum++;
			realFrame++;
		}
		if(localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS)
		{
			getPose.solvePnP(mapPointsB.mworldMapPointsV, 
							localTrackPointsB[lTPB-1].mfeatures, intrinsicK);
			getPose.GetRMatTPose();
			getPose.CombineRt();
			globalRTMat.emplace_back(std::move(getPose.mCombineRt));
			globalRVec.emplace_back(std::move(getPose.mrvec));
			globalTVec.emplace_back(std::move(getPose.mtvec));
			gP++;

			// triangulate
			mapPointsB.CalcWorldPoints(globalRTMat.at(gP-2), globalRTMat.at(gP-1), 
									localTrackPointsA.at(0).mfeatures, localTrackPointsA.at(lTPA).mfeatures);
			mapPointsB.ScalingPoints();
			mapPointsB.MatToPoints3d();
			globalLandMark.emplace_back(mapPointsB);
			gLM++;

			if(!trackerB.GoodFeaturesToTrack(img))
			{	
				std::cout << "new tracker A" << std::endl;
			}
			localTrackPointsB.emplace_back(std::move(trackerB));
			lTPB++;
			imageCurNum++;
			realFrame++;
		}

		// tracking

		localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA);
				localTrackPointsA.emplace_back(std::move(trackerA));
		lTPA++;
		for(int i = 0; i < lTPA-1; i++)
		{
			ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i), mapPointsA.mworldMapPointsV);
		}
		std::cout << std::endl;
		std::cout << "mapPoints.size A: " << mapPointsA.mworldMapPointsV.size() << std::endl;
		std::cout << "present local points size A: " << localTrackPointsA[lTPA].mfeatures.size() << std::endl;
		std::cout << "before local points size A: " << localTrackPointsA[lTPA-1].mfeatures.size() << std::endl;

		std::cout << std::endl;
		std::cout << "lTPA: " << lTPA << std::endl;
		std::cout << "lTPA feature size : " << localTrackPointsA[lTPA].mfeatures.size() << std::endl << std::endl;
		
		
		localTrackPointsB[lTPB].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerB);
		localTrackPointsB.emplace_back(std::move(trackerB));
		lTPB++;
		for(int i = 0; i < lTPB-1; i++)
		{
			ManageTrackPoints(localTrackPointsB.at(lTPB), localTrackPointsB.at(i), mapPointsB.mworldMapPointsV);
		}
		std::cout << "mapPoints.size B = " << mapPointsB.mworldMapPointsV.size() << std::endl;
		std::cout << "present local points size B: " << localTrackPointsB[lTPB].mfeatures.size() << std::endl;
		std::cout << "before local points size B: " << localTrackPointsB[lTPB-1].mfeatures.size() << std::endl;

		std::cout << std::endl;
		std::cout << "lTPB: " << lTPB << std::endl;
		std::cout << "lTPB feature size : " << localTrackPointsB[lTPB].mfeatures.size() << std::endl << std::endl;


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
		std::cout << "imageCurnum: " << imageCurNum;
		std::cout << " frame: " << realFrame <<std::endl;
		std::cout << "lTPA: " << lTPA;
		std::cout << " lTPB: " << lTPB << std::endl;
		imageCurNum++;
		realFrame++;
		cv::imshow("img", img);
		if(cv::waitKey(0) == 27) break; // ESC key
	}

    return 0;
}

