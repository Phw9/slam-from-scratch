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
	localTrackPointsA.reserve(300);
	int lTPA = 0;
	std::vector<mvo::Feature> localTrackPointsB;
	localTrackPointsB.reserve(300);
	int lTPB = 0;
	std::vector<uchar> stats;

	mvo::StrctureFromMotion getEssential;
	mvo::PoseEstimation getPose;

	std::vector<int> mapStats;
	cv::Mat cam2World;
	mvo::Triangulate mapPointsA, mapPointsB;
	std::vector<cv::Point3f> localMapPointsA, localMapPointsB;
	std::vector<mvo::Triangulate> globalLandMark;
	int gLM = 0;
	int gKF = 0;

	cv::Vec<double,1> angularVelocity;
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
					std::cerr << "new tracker A" << std::endl;
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
					std::cerr << "fail scaling" << std::endl;
				}

				if(!mapPointsA.ScalingPoints())
				{
					std::cerr << "fail scaling" << std::endl;
				}
				mapPointsA.MatToPoints3f();	mapStats.clear();
				m2 = m2.t(), m2.pop_back();
				m2 = m2.t(), cam2World=m2;
				if(!ManageMinusZ(mapPointsA, cam2World, mapStats))
				{
					std::cerr << "failed ManageMinusZ A" << std::endl;
				}
				if(!ManageMinusLocal(localTrackPointsA, mapStats))
				{
					std::cerr << "failed Minus local" << std::endl;
				}
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
				gKF++;
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
					ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i));
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
		// while(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS || localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS)
		while((localTrackPointsA.size()>3 && localTrackPointsB.size()>3) && 
				(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS || 
				localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS || 
				angularVelocity[0] > ANGULARVELOCITY ||
				(getPose.minlier.rows < NUMOFINLIER && gP>ESSENTIALFRAME)))
		{
			imageCurNum--;
			img = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);
			if(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS || mapPointsA.mworldMapPointsV.size() == localTrackPointsA.at(lTPA).mfeatures.size())
			{
				std::cout << "ㅇㅇㅇNewTrack Aㅇㅇㅇ" << std::endl;
				// triangulate
				mapPointsB.mworldMapPointsV.clear();
				mapPointsB.CalcWorldPoints(intrinsicKd*globalRTMat.at(gP-lTPB), intrinsicKd*globalRTMat.at(gP-1), 
										localTrackPointsB.at(0).mfeatures, localTrackPointsB.at(lTPB).mfeatures);
				if(!mapPointsB.ScalingPoints())
				{
					std::cerr << "failed scaling" << std::endl;
				}
				mapPointsB.MatToPoints3f(); mapStats.clear(); cam2World=getPose.mRotation;
				std::cout << "mapPointsB.mworldMapPointsV.size: " <<mapPointsB.mworldMapPointsV.size() << std::endl;
				if(!ManageMinusZ(mapPointsB, cam2World, mapStats))
				{
					std::cerr << "failed ManageMinusZ B" << std::endl;
				}
				if(!ManageMinusLocal(localTrackPointsB, mapStats))
				{
					std::cerr << "failed Minus local B" << std::endl;
				}
				globalLandMark.emplace_back(mapPointsB);
				gLM++;

				localTrackPointsA.clear();
				if(!trackerA1.GoodFeaturesToTrack(img))
				{
					std::cerr << "new tracker A" << std::endl;
				}
				localTrackPointsA.emplace_back(std::move(trackerA1));
				lTPA = 0;
				std::cout << "localTrackPointsA size : " << localTrackPointsA.size() << std::endl;
				std::cout << "localTrackPointsA.mfeature size : " << localTrackPointsA.at(0).mfeatures.size() <<std::endl;
				std::cout << "localTrackPointsB size : " << localTrackPointsB.size() << std::endl;
				std::cout << "localTrackPointsB.mfeature size : " << localTrackPointsB.at(lTPB).mfeatures.size() << std::endl;
				imageCurNum++;
				img = cv::imread(readImageName.at(imageCurNum), 
							cv::ImreadModes::IMREAD_UNCHANGED);
				gKF++;
				break;
			}

			if(localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS || mapPointsB.mworldMapPointsV.size() == localTrackPointsB.at(lTPB).mfeatures.size())
			{
				std::cout << "ㅇㅇㅇNewTrack Bㅇㅇㅇ" << std::endl;
				// triangulate
				mapPointsA.mworldMapPointsV.clear();
				mapPointsA.CalcWorldPoints(intrinsicKd*globalRTMat.at(gP-lTPA), intrinsicKd*globalRTMat.at(gP-1), 
										localTrackPointsA.at(0).mfeatures, localTrackPointsA.at(lTPA).mfeatures);
				if(!mapPointsA.ScalingPoints())
				{
					std::cerr << "failed scaling" << std::endl;
				}
				mapPointsA.MatToPoints3f(); mapStats.clear(); cam2World = getPose.mRotation;
				if(!ManageMinusZ(mapPointsA, cam2World, mapStats))
				{
					std::cerr << "failed ManageMinusZ A" << std::endl;
				}
				if(!ManageMinusLocal(localTrackPointsA, mapStats))
				{
					std::cerr << "failed Minus local" << std::endl;
				}
				globalLandMark.emplace_back(mapPointsA);
				gLM++;

				localTrackPointsB.clear();
				if(!trackerB1.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker B" << std::endl;
				}
				localTrackPointsB.emplace_back(std::move(trackerB1));
				lTPB = 0;
				std::cout << "localTrackPointsB size : " << localTrackPointsB.size() << std::endl;
				std::cout << "localTrackPointsB.mfeature size : " << localTrackPointsB.at(0).mfeatures.size() <<std::endl;
				std::cout << "localTrackPointsA.size : " << localTrackPointsA.size() << std::endl;
				std::cout << "localTrackPointsA.mfeature0 size : " << localTrackPointsA.at(0).mfeatures.size() << std::endl;
				std::cout << "localTrackPointsA.mfeature size : " << localTrackPointsA.at(lTPA).mfeatures.size() << std::endl;
				imageCurNum++;
				img = cv::imread(readImageName.at(imageCurNum), 
							cv::ImreadModes::IMREAD_UNCHANGED);
				gKF++;
				break;
			}
			// imageCurNum++;
			// img = cv::imread(readImageName.at(imageCurNum), 
			// 			cv::ImreadModes::IMREAD_UNCHANGED);
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
			ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i));
		}
		stats = localTrackPointsA.at(lTPA).mstatus;
		std::cout << "statssize: " << stats.size() << std::endl;
		std::cout << "before mapPointsA.mworldMapPointsV: " << mapPointsA.mworldMapPointsV.size() << std::endl;
		if(!ManageMapPoints(stats, mapPointsA.mworldMapPointsV))
		{
			std::cerr << "failed manage MapPointsA" << std::endl;
		}
		std::cout << "after mapPointsA.mworldMapPointsV: " << mapPointsA.mworldMapPointsV.size() << std::endl;

		
		if(mapPointsA.mworldMapPointsV.size() == localTrackPointsA.at(lTPA).mfeatures.size())
		{
			std::cout << "ㅁㅁㅁㅁㅁTrackA SolvePnPㅁㅁㅁㅁㅁㅁ" << std::endl;
			getPose.solvePnP(mapPointsA.mworldMapPointsV, localTrackPointsA.at(lTPA).mfeatures, intrinsicKd);
			// std::cout << "before inlier lTPA size: " << localTrackPointsA.at(lTPA).mfeatures.size() << std::endl;
			// std::cout << "before inlier mappoints size: " << mapPointsA.mworldMapPointsV.size() << std::endl;
			// ManageInlier(localTrackPointsA, mapPointsA.mworldMapPointsV, getPose.minlier);
			// std::cout << "after inlier lTPA size: " << localTrackPointsA.at(lTPA).mfeatures.size() << std::endl;
			// std::cout << "after inlier mappoints size: " << mapPointsA.mworldMapPointsV.size() << std::endl;	
			getPose.GetRMatTPose(); getPose.CombineRt();
			globalRTMat.emplace_back(std::move(getPose.mCombineRt));
			globalRVec.emplace_back(std::move(getPose.mrvec));
			globalTVec.emplace_back(std::move(getPose.mtvec));
			gP++;
			std::cout << std::endl;
			

			angularVelocity = globalRVec.at(gP-1).t() * globalRVec.at(gP-1);

			

			std::cout << "mrvec: " << globalRVec.at(gP-1) << "  angularV: " << angularVelocity << std::endl;
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
			ManageTrackPoints(localTrackPointsB.at(lTPB), localTrackPointsB.at(i));
		}
		stats = localTrackPointsB.at(lTPB).mstatus;
		std::cout << "statssize: " << stats.size() << std::endl;
		std::cout << "before mapPointsB.mworldMapPointsV: " << mapPointsB.mworldMapPointsV.size() << std::endl;
		if(!ManageMapPoints(stats, mapPointsB.mworldMapPointsV))
		{
			std::cerr << "failed manage MapPointsB" << std::endl;
		}
		std::cout << "after mapPointsB.mworldMapPointsV: " << mapPointsB.mworldMapPointsV.size() << std::endl;
		std::cout << "after trackPoints: " << localTrackPointsB.at(lTPB).mfeatures.size() << std::endl;


		if(mapPointsB.mworldMapPointsV.size() == localTrackPointsB.at(lTPB).mfeatures.size())
		{
			std::cout << "ㅁㅁㅁㅁㅁTrackB SolvePnPㅁㅁㅁㅁㅁㅁ" << std::endl;
			getPose.solvePnP(mapPointsB.mworldMapPointsV, localTrackPointsB.at(lTPB).mfeatures, intrinsicKd);
			// std::cout << "before inlier lTPB size: " << localTrackPointsB.at(lTPB).mfeatures.size() << std::endl;
			// std::cout << "before inlier mappoints size: " << mapPointsB.mworldMapPointsV.size() << std::endl;
			// ManageInlier(localTrackPointsB, mapPointsB.mworldMapPointsV, getPose.minlier);
			// std::cout << "after inlier lTPB size: " << localTrackPointsB.at(lTPB).mfeatures.size() << std::endl;
			// std::cout << "after inlier mappoints size: " << mapPointsB.mworldMapPointsV.size() << std::endl;		
			getPose.GetRMatTPose(); getPose.CombineRt();
			globalRTMat.emplace_back(std::move(getPose.mCombineRt));
			globalRVec.emplace_back(std::move(getPose.mrvec));
			globalTVec.emplace_back(std::move(getPose.mtvec));
			gP++;

			angularVelocity = globalRVec.at(gP-1).t() * globalRVec.at(gP-1);

			std::cout << std::endl;
			std::cout << "mrvec: " << globalRVec.at(gP-1) << "  angularV: " << angularVelocity << std::endl;
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
		std::cout << "mtvec: " << globalTVec.at(gP-1) << "  " << globalTVec.size() << std::endl;
		std::cout << "tvecOfGT: " << tvecOfGT.at(imageCurNum) << "  gKF: " << gKF << std::endl;
		std::cout << "-------------------------" << std::endl;
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

