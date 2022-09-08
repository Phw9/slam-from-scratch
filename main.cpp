#include "Config.h"
#include "Feature.h"
#include "Init.h"
#include "PoseEstimation.h"
#include "Triangulate.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


int main(int argc, char** argv)
{
    std::ofstream rawData ("./image.txt", rawData.out | rawData.trunc);
	std::ifstream read ("./image.txt", read.in);
	std::ifstream readGTtvec ("../image/GTpose.txt", readGTtvec.in);
    
	std::vector<cv::Vec3f> readtvecOfGT;
	std::vector<cv::Vec3f> tvecOfGT;
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
	GTPoseRead(readtvecOfGT, readGTtvec);
	

	mvo::Feature detector;
	mvo::Feature trackerA1, trackerA2, trackerB1, trackerB2;
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
	
	Viewer::MyVisualize pangolinViewer=Viewer::MyVisualize(WINDOWWIDTH, WINDOWHEIGHT);
    pangolinViewer.initialize();
    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(WINDOWWIDTH, WINDOWHEIGHT, VIEWPOINTF, VIEWPOINTF, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(VIEWPOINTX, VIEWPOINTY, VIEWPOINTZ, 0, 0, 0, 0.0, -1.0, 0.0));
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -pangolinViewer.window_ratio)
                                .SetHandler(new pangolin::Handler3D(s_cam));

	while(true)
	{
		while(realFrame < ESSENTIALFRAME)
		{
			img = cv::imread(readImageName.at(imageCurNum), 
								cv::ImreadModes::IMREAD_UNCHANGED);
			if (img.empty())
			{
				std::cerr << "frame upload failed" << std::endl;
			}
			
			if(realFrame == 0)	// feature extract
			{
				if(!trackerA1.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker A" << std::endl;
				}
				localTrackPointsA.emplace_back(std::move(trackerA1));
			}
			else if(realFrame == ESSENTIALFRAME-1)	// 2-viewSFM(2)
			{
				imageCurNum--;
				cv::Mat m1 = cv::Mat::eye(cv::Size(4,3), CV_64F);
				cv::Mat m2 = cv::Mat::eye(cv::Size(4,3), CV_64F);
				for(int i = 0; i<gP; i++)
				{
					std::cout << std::endl << "globalRTMat: " << std::endl << globalRTMat[i] << std::endl;
					m2 = mvo::MultiplyMat(m2,globalRTMat[i]);
					std::cout << "m2: " << i << std::endl << m2 << std::endl;
				}

				// Triangulate Landmark
				if(!mapPointsA.CalcWorldPoints(intrinsicKd*m1, intrinsicKd*m2,
							localTrackPointsA[0].mfeatures, localTrackPointsA[lTPA].mfeatures))
				{
					std::cout << "fail scaling" << std::endl;
				}

				if(!mapPointsA.ScalingPoints())
				{
					std::cout << "fail scaling" << std::endl;
				}
				mapPointsA.MatToPoints3f();
				globalLandMark.emplace_back(mapPointsA);
				gLM++;
				// create Track B
				if(!trackerB1.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker B" << std::endl;
				}
				localTrackPointsB.emplace_back(std::move(trackerB1));
				std::cout << "lTPA size: " << localTrackPointsA.size() << std::endl;
				std::cout << "lTPA: " << lTPA << std::endl;
			}
			else	// tracking
			{
				localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
														cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA1);
				localTrackPointsA.emplace_back(std::move(trackerA1));
				lTPA++;
				getEssential.CreateEssentialMatrix(localTrackPointsA[lTPA-1].mfeatures, localTrackPointsA[lTPA].mfeatures, intrinsicKf);
				getEssential.GetEssentialRt(getEssential.mEssential, intrinsicKf,
											localTrackPointsA[lTPA-1].mfeatures, 
											localTrackPointsA[lTPA].mfeatures);
				getEssential.GetRTvec(); getEssential.CombineRt();
				globalRTMat.emplace_back(std::move(getEssential.mCombineRt));
				globalRVec.emplace_back(std::move(getEssential.mrvec));
				globalTVec.emplace_back(std::move(getEssential.mtvec));
				gP++;
				std::cout << "lTPA: " << lTPA << std::endl;

				for(int i = 0; i < lTPA-1; i++)
				{
					ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i), mapPointsA.mworldMapPointsV);
					std::cout << "size" << i <<": " << localTrackPointsA.at(i).mfeatures.size() << " ";
				}
				cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
				img = pangolinViewer.DrawFeatures(img, localTrackPointsA.at(lTPA-1).mfeatures, 
													localTrackPointsA.at(lTPA).mfeatures);							
			}
			std::cout << std::endl;
			tvecOfGT.emplace_back(readtvecOfGT.at(imageCurNum));
			std::cout << "tvecOfGT: " << tvecOfGT.at(imageCurNum) << std::endl;
			imageCurNum++;
			realFrame++;
			
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			d_cam.Activate(s_cam);
			pangolinViewer.DrawPoint(globalTVec, tvecOfGT, globalLandMark, mapPointsA.mworldMapPointsV);
			pangolin::FinishFrame();
			
			cv::imshow("img", img);
			if(cv::waitKey(0) == 27) break; // ESC key
		} // 2view SFM, Track(A,B) make, generate LandMark = current localFeature size

		// Start

		img = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);
		
		// if(keyframe gen){ generate Pose, MapPoints, new Track}

		// if num of Feature is less than NUMOFPOINTS, GFTT
		if(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS || localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS)
		{
			if(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS)
			{
				// triangulate
				mapPointsB.CalcWorldPoints(globalRTMat.at(gP-2), globalRTMat.at(gP-1), 
										localTrackPointsB.at(0).mfeatures, localTrackPointsB.at(lTPB).mfeatures);
				mapPointsB.ScalingPoints(); mapPointsB.MatToPoints3f();
				globalLandMark.emplace_back(mapPointsB);
				gLM++;

				localTrackPointsA.clear();
				lTPA = 1;
				std::cout << "local A size : " << localTrackPointsA.size() << std::endl;
				std::cout << "localTrackPointsA.mfeature size : " << localTrackPointsA.at(0).mfeatures.size() <<std::endl;
				std::cout << "localTrackPointsB.size(): " << localTrackPointsB.size() << std::endl;
				std::cout << "localTrackPointsB.at(lTPB).feature size(): " << localTrackPointsB.at(lTPB).mfeatures.size() << std::endl;
			}

			if(localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS)
			{
				std::cout << "hello1B" <<std::endl;

				// triangulate
				mapPointsA.CalcWorldPoints(globalRTMat.at(gP-2), globalRTMat.at(gP-1), 
										localTrackPointsA.at(0).mfeatures, localTrackPointsA.at(lTPA).mfeatures);
				mapPointsA.ScalingPoints(); mapPointsA.MatToPoints3f();
				globalLandMark.emplace_back(mapPointsA);
				gLM++;
				std::vector<mvo::Feature> localTrackPointsB;
				mvo::Feature trackerB1;
				std::cout << "hello2B" << std::endl;
				if(!trackerB1.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker A" << std::endl;
				}
				std::cout << "local B size : " << localTrackPointsB.size() << std::endl;
				localTrackPointsA.emplace_back(std::move(trackerB1));
				lTPB = 1;
			}
			imageCurNum--;
		}

		// tracking
		localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA1);
		localTrackPointsA.emplace_back(std::move(trackerA1));
		lTPA++;	
		std::cout << "localTrackPointsA size: " << localTrackPointsA.size() << std::endl;
		std::cout << "localTrackPointsA.mfeature size : " << localTrackPointsA.at(0).mfeatures.size() <<std::endl;
		for(int i = 0; i < lTPA-1; i++)
		{
			ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i), mapPointsA.mworldMapPointsV);
		}
		std::cout << "ddd:" << std::endl;
		localTrackPointsA.clear();

		
		if(mapPointsA.mworldMapPointsV.size() == localTrackPointsA.at(lTPA).mfeatures.size())
		{
			std::cout << "before inlier lTPA size: " << localTrackPointsA.at(lTPA).mfeatures.size() << std::endl;
			std::cout << "before inlier mappoints size: " << mapPointsA.mworldMapPointsV.size() << std::endl;
			getPose.solvePnP(mapPointsA.mworldMapPointsV, localTrackPointsA.at(lTPA).mfeatures, intrinsicKd);
			ManageInlier(localTrackPointsA, mapPointsA.mworldMapPointsV, getPose.minlier);
			std::cout << "after inlier lTPA size: " << localTrackPointsA.at(lTPA).mfeatures.size() << std::endl;
			std::cout << "after inlier mappoints size: " << mapPointsA.mworldMapPointsV.size() << std::endl;	

			getPose.GetRMatTPose(); getPose.CombineRt();
			globalRTMat.emplace_back(std::move(getPose.mCombineRt));
			globalRVec.emplace_back(std::move(getPose.mrvec));
			globalTVec.emplace_back(std::move(getPose.mtvec));
			gP++;
			std::cout << std::endl;
			std::cout << "mrvec: " << globalRVec.at(gP-1) << std::endl;
			std::cout << "mtvec: " << globalTVec.at(gP-1) << std::endl;
			std::cout << "mapPoints.size A: " << mapPointsA.mworldMapPointsV.size() << std::endl;
			std::cout << "present local points size A: " << localTrackPointsA[lTPA].mfeatures.size() << std::endl;
			std::cout << "before local points size A: " << localTrackPointsA[lTPA-1].mfeatures.size() << std::endl;
		}

		std::cout << std::endl;
		std::cout << "lTPA: " << lTPA << std::endl;
		std::cout << "lTPA feature size : " << localTrackPointsA[lTPA].mfeatures.size() << std::endl << std::endl;
		
		
		localTrackPointsB[lTPB].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerB1);
		localTrackPointsB.emplace_back(std::move(trackerB1));
		lTPB++;

		for(int i = 0; i < lTPB-1; i++)
		{		
			ManageTrackPoints(localTrackPointsB.at(lTPB), localTrackPointsB.at(i), mapPointsB.mworldMapPointsV);
		}
		if(mapPointsB.mworldMapPointsV.size() == localTrackPointsB.at(lTPB).mfeatures.size())
		{
			getPose.solvePnP(mapPointsB.mworldMapPointsV, localTrackPointsA.at(lTPB).mfeatures, intrinsicKd);
			std::cout << "before inlier lTPB size: " << localTrackPointsB.at(lTPB).mfeatures.size() << std::endl;
			std::cout << "before inlier mappoints size: " << mapPointsB.mworldMapPointsV.size() << std::endl;
			ManageInlier(localTrackPointsB, mapPointsB.mworldMapPointsV, getPose.minlier);
			std::cout << "after inlier lTPB size: " << localTrackPointsB.at(lTPB).mfeatures.size() << std::endl;
			std::cout << "after inlier mappoints size: " << mapPointsB.mworldMapPointsV.size() << std::endl;		
			getPose.GetRMatTPose(); getPose.CombineRt();
			globalRTMat.emplace_back(std::move(getPose.mCombineRt));
			globalRVec.emplace_back(std::move(getPose.mrvec));
			globalTVec.emplace_back(std::move(getPose.mtvec));
			gP++;
			std::cout << std::endl;
			std::cout << "mrvec: " << globalRVec.at(gP-1) << std::endl;
			std::cout << "mtvec: " << globalTVec.at(gP-1) << std::endl;
			std::cout << "mapPoints.size B: " << mapPointsB.mworldMapPointsV.size() << std::endl;
			std::cout << "present local points size B: " << localTrackPointsA[lTPB].mfeatures.size() << std::endl;
			std::cout << "before local points size B: " << localTrackPointsA[lTPB-1].mfeatures.size() << std::endl;
		}

		std::cout << std::endl;
		std::cout << "lTPB: " << lTPB << std::endl;
		std::cout << "lTPB feature size : " << localTrackPointsB[lTPB].mfeatures.size() << std::endl << std::endl;

		// draw tracking points
		if(localTrackPointsA[lTPA].mfeatures.size() > localTrackPointsB[lTPB].mfeatures.size())
		{
		cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		img = pangolinViewer.DrawFeatures(img, localTrackPointsA.at(lTPA-1).mfeatures, 
											localTrackPointsA.at(lTPA).mfeatures);
		}
		else
		{
		cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		img = pangolinViewer.DrawFeatures(img, localTrackPointsB.at(lTPB-1).mfeatures, 
											localTrackPointsB.at(lTPB).mfeatures);		
		}
		tvecOfGT.emplace_back(readtvecOfGT.at(imageCurNum));
		std::cout << std::endl;
		std::cout << "mtvec: " << globalTVec.at(gP-1) << std::endl;
		std::cout << "tvecOfGT: " << tvecOfGT.at(imageCurNum) << std::endl;
		imageCurNum++;
		realFrame++;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);
		if(mapPointsA.mworldMapPointsV.size() > mapPointsB.mworldMapPointsV.size())
		{
			pangolinViewer.DrawPoint(globalTVec, tvecOfGT, globalLandMark, mapPointsA.mworldMapPointsV);
		}
		else
		{
			pangolinViewer.DrawPoint(globalTVec, tvecOfGT, globalLandMark, mapPointsB.mworldMapPointsV);
		}
    	pangolin::FinishFrame();
		cv::imshow("img", img);
		if(cv::waitKey(0) == 27) break; // ESC key
	}

    return 0;
}

