#include "Config.h"
#include "Init.h"
#include "BundleAdjustment.h"
#include "LoopClosure.h"
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
	cv::Mat img2;
	int imageCurNum = 0;
	int realFrame = 0;
	float RH = 100; float SH, SF;

	MakeTextFile(rawData, IMAGENUM);
	FileRead(readImageName, read);
	GTPoseRead(readtvecOfGT, readGTtvec);
	
	mvo::Feature detector;
	mvo::Feature trackerA, trackerB;
	std::vector<mvo::Feature> localTrackPointsA;
	localTrackPointsA.reserve(300);
	int lTPA = 0;
	std::vector<mvo::Feature> localTrackPointsB;
	localTrackPointsB.reserve(300);
	int lTPB = 0;
	std::vector<uchar> stats;

	mvo::StrctureFromMotion getInitialize;
	mvo::PoseEstimation getPose;
	mvo::Initializer checkScore;
	std::vector<cv::Mat> localPose;
	localPose.reserve(100);

	std::vector<int> mapStats;
	mvo::Triangulate mapPointsA, mapPointsB;
	std::vector<cv::Point3f> localMapPointsA, localMapPointsB;
	localMapPointsA.reserve(100); localMapPointsB.reserve(100);

	mvo::MapData setMapData;
	std::vector<mvo::MapData> globalMapData;
	std::vector<std::vector<cv::Mat>> globaldesc;
	globalMapData.reserve(10000);
	int gD = 0;
	int gIdx = -1;
	mvo::Covisibilgraph covGraph(globalMapData, focald, camXd, camYd);
	// load the vocabulary from disk
  	OrbVocabulary voc("../image/KITTI_00_phphww_voc.yml.gz");
	OrbDatabase db(voc, false, 0); // false = do not use direct index
	cv::Ptr<cv::ORB> orb = cv::ORB::create();
	cv::Mat orbmask;
    std::vector<cv::KeyPoint> orbkps;
    cv::Mat orbdesc;
	DBoW2::QueryResults ret;
	bool bloop = false;

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
		if(initialNum == 1)
		{
			// Automatically Initilizer Calculate RH = SH/(SH+SF)
			if(realFrame == 0)	// feature extract
			{
				img = cv::imread(readImageName.at(imageCurNum), 
									cv::ImreadModes::IMREAD_UNCHANGED);
				img2 = cv::imread(readImageName.at(imageCurNum), 
									cv::ImreadModes::IMREAD_UNCHANGED);
				if (img.empty())
				{
					std::cerr << "frame upload failed" << std::endl;
				}				
				if(!trackerA.GoodFeaturesToTrack(img))
				{	
					std::cerr << "new tracker A" << std::endl;
				}
				localTrackPointsA.emplace_back(std::move(trackerA));
				tvecOfGT.emplace_back(readtvecOfGT.at(imageCurNum));
				imageCurNum++;
				realFrame++;
			}
			while(initialNum!=0)
			{
				img = cv::imread(readImageName.at(imageCurNum), 
									cv::ImreadModes::IMREAD_UNCHANGED);
				img2 = cv::imread(readImageName.at(imageCurNum), 
									cv::ImreadModes::IMREAD_UNCHANGED);
				if (img.empty())
				{
					std::cerr << "frame upload failed" << std::endl;
				}
				localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
															cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA);
				localTrackPointsA.emplace_back(std::move(trackerA));
				lTPA++;

				std::cout << "lTPA: " << lTPA << std::endl;

				for(int i = 0; i < lTPA-1; i++)
				{
					ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i));
					// std::cout << "size" << i <<": " << localTrackPointsA.at(i).mfeatures.size() << " ";
					// std::cout << "descsize" << i <<": " << localTrackPointsA.at(i).mvdesc.size() << std::endl;
				}

				SF = checkScore.CheckFundamental(localTrackPointsA.at(0).mfeatures, localTrackPointsA.at(lTPA).mfeatures, 200.0f);
				SH = checkScore.CheckHomography(localTrackPointsA.at(0).mfeatures, localTrackPointsA.at(lTPA).mfeatures, 200.0f);
				RH = SH/(SH+SF);
				std::cout << "SF: "<< SF << " SH:  " << SH << std::endl;
				std::cout << "score: " << RH << std::endl;
				// RH>0.45
				// SH>1800 && isnan(SF) == true
				if(lTPA == 1)	// 1, 3
				{
					std::cout << "RH>0.45" << std::endl;

					getInitialize.CreateEssentialMatrix(localTrackPointsA.at(0).mfeatures, localTrackPointsA[lTPA].mfeatures, intrinsicKf);
					mvo::GetLocalPose(localPose, getInitialize.mCombineRt);
					setMapData.GetSFMPose(getInitialize);

					if(!mapPointsA.CalcWorldPoints(m1, setMapData.mglobalRTMat,
							localTrackPointsA[0], localTrackPointsA[lTPA]))
					{
						std::cerr << "fail calc" << std::endl;
					}
					if(!ManageMinusZ(mapPointsA, setMapData.mglobalRMat, mapStats))
					{
						std::cerr << "failed ManageMinusZ A" << std::endl;
					}
					
					if(!ManageMinusLocal(localTrackPointsA, mapStats))
					{
						std::cerr << "failed Minus local" << std::endl;
					}
					
					setMapData.Get2DPoints(localTrackPointsA.at(lTPA)); setMapData.Get3DPoints(mapPointsA);
					globalMapData.emplace_back(std::move(setMapData));
					mvo::LoopDetectCompute(img, globaldesc, db);

					std::cout << "statssize: " << localTrackPointsA.at(lTPA).mstatus.size()<< std::endl;
					std::cout << "mapPointsA.mworldMapPointsV: " << mapPointsA.mworldMapPointsV.size() << std::endl;
					std::cout << "localTrackPoints 0: " << localTrackPointsA.at(0).mfeatures.size() << std::endl;
					std::cout << "localtrackPoints l: " << localTrackPointsA.at(lTPA).mfeatures.size() << std::endl;

					if(!trackerB.GoodFeaturesToTrack(img))
					{	
						std::cout << "new tracker B" << std::endl;
					}
					localTrackPointsB.emplace_back(std::move(trackerB));

					initialNum = 0;
				}
				cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
				cv::cvtColor(img2, img2, cv::ColorConversionCodes::COLOR_GRAY2BGR);
				img = pangolinViewer.DrawFeatures(img, localTrackPointsA.at(lTPA-1).mfeatures, 
													localTrackPointsA.at(lTPA).mfeatures);
				

				tvecOfGT.emplace_back(readtvecOfGT.at(imageCurNum));
				std::cout << "KeyFrame(glbalMapData)size: " << globalMapData.size() << ", real:" << realFrame << ", image: " << imageCurNum <<std::endl;
				std::cout << "tvecOfGT: " << tvecOfGT.at(imageCurNum) << std::endl;
				realFrame++;
				imageCurNum++;
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				d_cam.Activate(s_cam);
				pangolinViewer.DrawPoint(globalMapData, tvecOfGT);
				pangolin::FinishFrame();
				
				cv::imshow("img", img);
				cv::imshow("useful", img2);
				if(cv::waitKey(0) == 27) break; // ESC key
			}
			std::cout << "======================  2-view end & start!!  ======================" << std::endl;
		}
		// Start
		img = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);
		img2 = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);	
		// localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS || 
		// localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS ||
		// (getPose.minlier.rows < NUMOFINLIER && getPose.minlier.rows != 0)
		while(angularVelocity > ANGULARVELOCITY ||
			localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS || 	
			localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS ||
			(getPose.minlier.rows < NUMOFINLIER && getPose.minlier.rows != 0) ||
			inlierRatio < INLIERRATIO)
		{
			imageCurNum--;
			img = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);

			if(mapPointsA.mworldMapPointsV.size() == localTrackPointsA.at(lTPA).mfeatures.size())
			{
				// triangulate
				std::cout << "ㅇㅇㅇNewTrack Aㅇㅇㅇ" << std::endl;
				
				if(BUNDLE == 1)
				{
					if(((gD % LOCAL) == 0) && gD > 8 || 
						angularVelocity > ANGULARVELOCITY)
					{
						std::cout << "localBA start" << std::endl;
						
						mvo::BundleAdjustment* localBA = new mvo::BundleAdjustment();
						if(!localBA->LocalBA(gD, globalMapData, covGraph))
							std::cerr << "local error" << std::endl;
						else delete localBA;
					}
				}
				// descriptor of mapPointsA & mapPointsB matcher
				mapPointsB.CalcWorldPoints(globalMapData.at(gD).mglobalRTMat, getPose.mCombineRt, 
										localTrackPointsB.at(0), localTrackPointsB.at(lTPB));
				if(!ManageMinusZ(mapPointsB, localPose.at(lTPB/2), mapStats))
				{
					std::cerr << "failed ManageMinusZ B" << std::endl;
				}
				if(!ManageMinusLocal(localTrackPointsB, mapStats))
				{
					std::cerr << "failed Minus local B" << std::endl;
				}
				mvo::LoopDetectCompute(img, globaldesc, db);

				localPose.clear();
				setMapData.GetPnPPose(getPose); setMapData.Get2DPoints(localTrackPointsB.at(lTPB)); setMapData.Get3DPoints(mapPointsB);
				globalMapData.emplace_back(std::move(setMapData));	// mvo::PushData(globalMapData, setMapData);
				gD++; gIdx++;

				covGraph.MakeEdgeDesc(gD, localTrackPointsA.at(lTPA), mapPointsA);

				localTrackPointsA.clear();
				if(!trackerA.GoodFeaturesToTrack(img))
				{
					std::cerr << "new tracker A" << std::endl;
				}
				localTrackPointsA.emplace_back(std::move(trackerA));
				lTPA = 0;
				std::cout << "localTrackPoints(A,B).size : (" << localTrackPointsA.size() << ", " << localTrackPointsB.size() << ")" << std::endl;
				std::cout << "localTrackPoints(A,B).mfeature.size : (" << localTrackPointsA.at(0).mfeatures.size() << ", " << localTrackPointsB.at(lTPB).mfeatures.size()<< ")" <<std::endl;
				imageCurNum++;
				img = cv::imread(readImageName.at(imageCurNum), 
							cv::ImreadModes::IMREAD_UNCHANGED);
				break;
			}

			if(mapPointsB.mworldMapPointsV.size() == localTrackPointsB.at(lTPB).mfeatures.size())
			{
				// triangulate
				std::cout << "ㅇㅇㅇNewTrack Bㅇㅇㅇ" << std::endl;
				if(BUNDLE == 1)
				{
					if(((gD % LOCAL) == 0) && gD > 8 ||
						angularVelocity > ANGULARVELOCITY)
					{
						std::cout << "localBA start" << std::endl;
						mvo::BundleAdjustment* localBA = new mvo::BundleAdjustment();
						if(!localBA->LocalBA(gD, globalMapData, covGraph)) 
							std::cerr << "local error" << std::endl;
						else delete localBA;
					}
				}

				mapPointsA.CalcWorldPoints(globalMapData.at(gD).mglobalRTMat, getPose.mCombineRt, 
										localTrackPointsA.at(0), localTrackPointsA.at(lTPA));

				if(!ManageMinusZ(mapPointsA, localPose.at(lTPA/2), mapStats))
				{
					std::cerr << "failed ManageMinusZ A" << std::endl;
				}
				if(!ManageMinusLocal(localTrackPointsA, mapStats))
				{
					std::cerr << "failed Minus local A" << std::endl;
				}
				mvo::LoopDetectCompute(img, globaldesc, db);

				localPose.clear();
				setMapData.GetPnPPose(getPose); setMapData.Get2DPoints(localTrackPointsA.at(lTPA)); setMapData.Get3DPoints(mapPointsA);
				globalMapData.emplace_back(std::move(setMapData));	// mvo::PushData(globalMapData, setMapData);
				gD++; gIdx++;
				
				covGraph.MakeEdgeDesc(gD, localTrackPointsB.at(lTPB), mapPointsB);

				localTrackPointsB.clear();
				if(!trackerB.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker B" << std::endl;
				}
				localTrackPointsB.emplace_back(std::move(trackerB));
				lTPB = 0;
				std::cout << "localTrackPoints(B,A).size : (" << localTrackPointsB.size()
				<< ", " << localTrackPointsA.size() << ")" << std::endl;
				std::cout << "localTrackPoints(B,A).mfeature.size : (" << localTrackPointsB.at(0).mfeatures.size()
				<< ", " << localTrackPointsA.at(lTPA).mfeatures.size()<< ")" <<std::endl;

				imageCurNum++;
				img = cv::imread(readImageName.at(imageCurNum), 
							cv::ImreadModes::IMREAD_UNCHANGED);
				break;
			}
			// imageCurNum++;
			// img = cv::imread(readImageName.at(imageCurNum), 
			// 			cv::ImreadModes::IMREAD_UNCHANGED);
		}
		// tracking A
		localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA);
		localTrackPointsA.emplace_back(std::move(trackerA));
		lTPA++;
    
		for(int i = 0; i < lTPA-1; i++)
		{
			ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i));
		}

		if(!mapPointsA.ManageMapPoints(localTrackPointsA.at(lTPA).mstatus))
		{
			std::cerr << "failed manage MapPointsA" << std::endl;
		}

		if(!mapPointsA.ManageMapPoints(localTrackPointsA.at(lTPA).mdelete))
		{
			std::cerr << "failed manage MapPointsA" << std::endl;
		}

		if(mapPointsA.mworldMapPointsV.size() == localTrackPointsA.at(lTPA).mfeatures.size())
		{
			std::cout << "ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁTrackA SolvePnPㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ" << std::endl;

			getPose.solvePnP(mapPointsA.mworldMapPointsV, localTrackPointsA.at(lTPA).mfeatures, intrinsicKd);
			mvo::GetLocalPose(localPose, getPose.mCombineRt);
			inlierRatio = (float)getPose.minlier.rows/(float)localTrackPointsA.at(lTPA).mfeatures.size();
			std::cout << "inlierRatio: " << inlierRatio << std::endl;
			
			if(BUNDLE == 1)
			{
				// std::unique_ptr<mvo::BundleAdjustment> ba = std::make_unique<mvo::BundleAdjustment>(localTrackPointsA.at(lTPA), getPose, mapPointsA));
				mvo::BundleAdjustment* ba = new mvo::BundleAdjustment(localTrackPointsA.at(lTPA), getPose, mapPointsA);
				if(!ba->MotionOnlyBA())
					std::cerr << "motion error" << std::endl;
				else delete ba;
			}
			
			if(gD>2)
			{
				angularVelocity = mvo::RotationAngle(globalMapData.at(gD-1).mglobalrvec, globalMapData.at(gD).mglobalrvec);
			}

			std::cout << std::endl;
			std::cout << "mapPoints.size A: " << mapPointsA.mworldMapPointsV.size() << std::endl;
			std::cout << "before, present local points size A: " << localTrackPointsA[lTPA-1].mfeatures.size() <<
			", "<< localTrackPointsA[lTPA].mfeatures.size() << std::endl;
		}
		std::cout << std::endl;

		// tacking B
		localTrackPointsB[lTPB].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerB);
		localTrackPointsB.emplace_back(std::move(trackerB));
		lTPB++;

		for(int i = 0; i < lTPB-1; i++)
		{
			ManageTrackPoints(localTrackPointsB.at(lTPB), localTrackPointsB.at(i));
		}

		if(!mapPointsB.ManageMapPoints(localTrackPointsB.at(lTPB).mstatus))
		{
			std::cerr << "failed manage MapPointsB" << std::endl;
		}
		if(!mapPointsB.ManageMapPoints(localTrackPointsB.at(lTPB).mdelete))
		{
			std::cerr << "failed manage delete PointsB" << std::endl;
		}		

		if(mapPointsB.mworldMapPointsV.size() == localTrackPointsB.at(lTPB).mfeatures.size())
		{
			std::cout << "ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁTrackB SolvePnPㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ" << std::endl;
			getPose.solvePnP(mapPointsB.mworldMapPointsV, localTrackPointsB.at(lTPB).mfeatures, intrinsicKf);
			mvo::GetLocalPose(localPose, getPose.mCombineRt);
			inlierRatio = (float)getPose.minlier.rows/(float)localTrackPointsB.at(lTPB).mfeatures.size();
			std::cout << "inlierRatio: " << inlierRatio << std::endl;

			if(BUNDLE == 1)
			{
				mvo::BundleAdjustment* ba = new mvo::BundleAdjustment(localTrackPointsB.at(lTPB), getPose, mapPointsB);
				if(!ba->MotionOnlyBA())
					std::cerr << "motion error" << std::endl;
				else delete ba;			
			}
			if(gD>2)
			{
				angularVelocity = mvo::RotationAngle(globalMapData.at(gD-1).mglobalrvec, globalMapData.at(gD).mglobalrvec);
			}
			std::cout << std::endl;
			std::cout << "mapPoints.size B: " << mapPointsB.mworldMapPointsV.size() << std::endl;
			std::cout << "before, present local points size B: " << localTrackPointsB[lTPB-1].mfeatures.size() <<
			", "<< localTrackPointsB[lTPB].mfeatures.size() << std::endl;
		}

		std::cout << std::endl;
		std::cout << "lTPA: " << lTPA << "                  lTPB: " << lTPB << std::endl;
		std::cout << "lTPA feature size : " << localTrackPointsA[lTPA].mfeatures.size()
		<< "  lTPB feature size : " << localTrackPointsB[lTPB].mfeatures.size() << std::endl;

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

		// Descriptor Matching Viewer : Search img2
		std::vector<cv::Point2f> aaa;
		std::vector<cv::Point2f> bbb;
		aaa.clear(); aaa.reserve(1000);
		bbb.clear(); bbb.reserve(1000);
		if(gD>=1)
		{
			int mgidx = covGraph.mglobalgraph.at(gD-1).size() - 1;
			int baba = covGraph.mglobalgraph.at(gD-1).at(mgidx).size();
			for(int i = 0; i < baba; i++)
			{
				cv::Point2f pr;
				cv::Point2f ne;
				int prev = covGraph.mglobalgraph.at(gD-1).at(mgidx).at(i).first;
				int next = covGraph.mglobalgraph.at(gD-1).at(mgidx).at(i).second;
				pr = globalMapData.at(gD-1).mpoint2D.at(prev);
				ne = globalMapData.at(gD).mpoint2D.at(next);
				aaa.push_back(pr);
				bbb.push_back(ne);
			}
			cv::cvtColor(img2, img2, cv::ColorConversionCodes::COLOR_GRAY2BGR);
			img2 = pangolinViewer.DrawFeatures(img2, aaa, bbb);
		}

		tvecOfGT.emplace_back(readtvecOfGT.at(imageCurNum));
		std::cout << std::endl;
		std::cout << "KeyFrame(glbalMapData)size: " << globalMapData.size() << ", image: " << imageCurNum <<std::endl;
		std::cout << "mpoint2D size: " << globalMapData.at(gD).mpoint2D.size()
		<< ", mpoint3D size: " << globalMapData.at(gD).mpoint3D.size()
		<< ", mvdesc size: " << globalMapData.at(gD).mvdesc.size() << std::endl;
		
		std::cout << "mrvec: " << globalMapData.at(gD).mglobalrvec<< "  angularV: " << angularVelocity << std::endl;
		std::cout << "mTranslation: " << globalMapData.at(gD).mglobalTranslation << " gD: " << gD << std::endl;
		std::cout << "tvecOfGT: " << tvecOfGT.at(imageCurNum) << "  KeyFrame: " << globalMapData.size() << " gIdx: " << covGraph.mglobalgraph.size() << " Img/KF: " << imageCurNum/(gD+1) << std::endl; 
		std::cout << "=============================================================================" << std::endl;
		imageCurNum++;
		realFrame++;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);
		pangolinViewer.DrawPoint(globalMapData, tvecOfGT);
    	pangolin::FinishFrame();
		cv::imshow("img", img);
		cv::imshow("useful", img2);
		char ch = cv::waitKey(30);
		if(ch == 27) break; // ESC key
		if(ch == 32) if(cv::waitKey(0) == 27) break;; // Spacebar key
	}	

    return 0;
}