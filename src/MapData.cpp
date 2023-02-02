#include "MapData.h"

mvo::MapData::MapData()
{
    mpoint2D.clear();
    mpoint3D.clear();
    mvdesc.clear();
    mglobalrvec.zeros();
    mglobaltvec.zeros();
}

bool mvo::MapData::GetSFMPose(const mvo::StrctureFromMotion& sfm)
{
    mglobalRTMat = sfm.mCombineRt.clone();  mglobalRMat = sfm.mRotation.clone();
    mglobalrvec = sfm.mrvec;        mglobaltvec = sfm.mtvec;
    mdesc = cv::Mat(); // orb feature 뽑아줘야함
    return true;
}

void mvo::GetLocalPose(std::vector<cv::Mat>& v, const cv::Mat& m)
{
    cv::Mat temp;
    temp = m.clone();
    v.push_back(m);
}

bool mvo::MapData::GetPnPPose(const mvo::PoseEstimation& pe)
{
    mglobalRTMat = pe.mCombineRt.clone();  mglobalRMat = pe.mRotation.clone();  minlier = pe.minlier.clone();
    mglobalrvec = pe.mrvec;  mglobaltvec = pe.mtvec;  mglobalTranslation = pe.mtranslation;
    // std::cout << "minlier: " << minlier << std::endl;
    // std::cout << "mglobalRTMat: " << mglobalRTMat <<std::endl;
    // std::cout << "pe.mCombineRT: " << pe.mCombineRt <<std::endl;
    return true;
}

bool mvo::MapData::Get2DPoints(const mvo::Feature& ft)
{   
    mpoint2D.clear();
    mpoint2D = ft.mfeatures;
    mvdesc = ft.mvdesc;
    mdesc = ft.mdesc.clone();

    if(mpoint2D.size() != ft.mfeatures.size()) return false;
    return true;
}

bool mvo::MapData::Get3DPoints(const mvo::Triangulate& tr)
{
    mpoint3D.clear();
    mpoint3D = tr.mworldMapPointsV;
    if(mpoint3D.size() != tr.mworldMapPointsV.size()) return false;
    // std::cout << "Get3DPoints: " << mpoint3D.size() << ", " << tr.mworldMapPointsV.size() << std::endl;
    // std::cout << "Get3Dmvdesc: " << mvdesc.size() << ", " << tr.mvdesc.size() << std::endl;
    // std::cout << "mappoint size: " << mpoint3D.size() << std::endl;

    return true;
}

void mvo::PushData(std::vector<mvo::MapData>& v, const mvo::MapData& md)
{
    mvo::MapData temp;

    temp.mglobalRMat = md.mglobalRMat.clone();
    temp.mglobalRTMat = md.mglobalRTMat.clone();
    temp.mglobalrvec = md.mglobalrvec;
    temp.mglobaltvec = md.mglobaltvec;
    temp.mpoint2D = md.mpoint2D;
    temp.mpoint3D = md.mpoint3D;
    temp.mvdesc = md.mvdesc;
    v.push_back(temp);
}

mvo::Covisibilgraph::Covisibilgraph(const std::vector<mvo::MapData>& v,
                                 const double& focal, const double& ppx, 
                                                      const double& ppy)
                                : mglobalMapData(v), mfocal(focal), mppx(ppx), mppy(ppy){};

template <typename T> void mvo::Covisibilgraph::MakeEdgeProj(int gD, std::vector<mvo::MapData>& mapdata)
{
    int w = gD-10;

    w++;
}

void mvo::Covisibilgraph::MakeEdgeDesc(int gD, mvo::Feature& before, mvo::Triangulate& mapPoints)
{
    std::cout << "gD: " << gD << ", size: " << mglobalMapData.size() << std::endl;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matches.reserve(2000);
    std::vector<std::pair<int,int>> temp;
    temp.reserve(2000);
    std::pair<int,int> idxMatch;
    std::vector<std::vector<std::pair<int,int>>> ttemp;
    ttemp.reserve(2000);

    if(gD < 10)
    {
        std::cout << "a: " << gD - 1 <<
                     " wk: " << mglobalMapData.at(gD).mpoint3D.size() << std::endl;
        int a = gD - 1;
        // int a = gD;
        int wk = mglobalMapData.at(gD).mpoint3D.size();
        for(int i = 0; i < a; i++)
        {
            int b = mglobalMapData.at(i).mpoint2D.size();
            for(int j = 0; j < b; j++)
            {
                for(int k = 0; k < wk; k++)
                {
                    if(this->Projection(mglobalMapData.at(i).mglobalrvec, mglobalMapData.at(i).mglobaltvec,
                                         mglobalMapData.at(i).mpoint2D.at(j), mglobalMapData.at(gD).mpoint3D.at(k)))
                    {
                        idxMatch.first = j;
                        idxMatch.second = k;
                        temp.emplace_back(std::move(idxMatch));
                        break;
                    }
                }
            }
            int tmpsz = temp.size();
            if(tmpsz == 0)
            {
                temp.reserve(10);
                ttemp.emplace_back(std::move(temp));
            }
            else
            {
                ttemp.emplace_back(std::move(temp));
            }
        }
    }
    else
    {
        // std::cout << "else a: " << gD - 1 <<
        //              " wk: " << mglobalMapData.at(gD).mpoint3D.size() << std::endl;

        int c = gD - 10;
        int wk = mglobalMapData.at(gD).mpoint3D.size();
        for(int i = 0; i < 9; i++)
        {
            int b = mglobalMapData.at(c).mpoint2D.size();
            for(int j = 0; j < b; j++)
            {
                for(int k = 0; k < wk; k++)
                {
                    if(this->Projection(mglobalMapData.at(c).mglobalrvec, mglobalMapData.at(c).mglobaltvec,
                                         mglobalMapData.at(c).mpoint2D.at(j), mglobalMapData.at(gD).mpoint3D.at(k)))
                    {
                        idxMatch.first = j;
                        idxMatch.second = k;
                        temp.emplace_back(std::move(idxMatch));
                        break;
                    }
                }
            }
            int tmpsz = temp.size();
            if(tmpsz == 0)
            {
                temp.reserve(10);
                ttemp.emplace_back(std::move(temp));
            }
            else
            {
                ttemp.emplace_back(std::move(temp));
            }
            c++;
        }
    }

    /*
    1. Matching 2D Points between before(query) and gD(train) 
    2. Correspond 3dPoints between (before==mapPoints) and (gD-1)
    3. Correcting index (before) in (gD-1)
    */
    // 1
    matcher.match(before.mdesc, mglobalMapData.at(gD).mdesc, matches);
    int max=0, min=100, sum=0;
    int maxq=0; int maxt=0;
    int inlier=0; int inlier2=0;
    int N = matches.size();

    for(int i = 0; i < N; i++)
    {
        if(max < matches.at(i).distance) max = matches.at(i).distance;
        if(maxq < matches.at(i).queryIdx) maxq = matches.at(i).queryIdx;
        if(maxt < matches.at(i).trainIdx) maxt = matches.at(i).trainIdx;
        if(min > matches.at(i).distance) min = matches.at(i).distance;
        sum += matches.at(i).distance;

        if(matches.at(i).distance < DISTANCEDESC)
        {
            inlier++;
            idxMatch.first = matches.at(i).queryIdx; idxMatch.second = matches.at(i).trainIdx;
            // std::cout << "first: " << idxMatch.first << " second: " << idxMatch.second << std::endl;
            temp.emplace_back(std::move(idxMatch));
        }
    }

    // if mapPoints.at(temp.first) == (gD-1)3dpoints
    // temp.first = (gD-1) 3dPoints idx
    int M = temp.size();
    int P = mglobalMapData.at(gD-1).mpoint3D.size();
    std::cout << "temp desc: " << temp.size() << ", " << P << " > " << mapPoints.mworldMapPointsV.size() << std::endl;
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < P; j++)
        {
            if(mapPoints.mworldMapPointsV.at(temp.at(i).first).x == mglobalMapData.at(gD-1).mpoint3D.at(j).x
            && mapPoints.mworldMapPointsV.at(temp.at(i).first).y == mglobalMapData.at(gD-1).mpoint3D.at(j).y
            && mapPoints.mworldMapPointsV.at(temp.at(i).first).z == mglobalMapData.at(gD-1).mpoint3D.at(j).z)
            {
                inlier2++;
                temp.at(i).first = j;
                break;
            }
        }
    }
    std::cout << "inlier2: " << inlier2 << std::endl;

    int indexCorrection = 0;

    for(int i = 0; i < M; i++)
    {
        if(!this->Projection(mglobalMapData.at(gD-1).mglobalrvec, mglobalMapData.at(gD-1).mglobaltvec,
                        mglobalMapData.at(gD-1).mpoint2D.at(temp.at(i-indexCorrection).first),
                        mglobalMapData.at(gD).mpoint3D.at(temp.at(i-indexCorrection).second)))
        {
            temp.erase(temp.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
    ttemp.emplace_back(std::move(temp));
    mglobalgraph.emplace_back(std::move(ttemp));
    
    // std::cout << "mglobalgraph size : " << mglobalgraph.size() << std::endl;
    int mgge = mglobalgraph.size();
    int mgges = mglobalgraph.size();
    if(mgge > 10) mgge = 10;
    int mggs = gD - mgge;
    // std::cout << "mggs : " << mggs << " mgge : " << mgge << std::endl;
    for(int i = mggs; i < mgges; i++)
    {
        std::cout << "inlier " << i << ", num of local matching(" << mglobalgraph.at(i).size() << ") : ";
        int mggis = mglobalgraph.at(i).size();
        for(int j = 0; j < mggis; j++)
        {
            std::cout << mglobalgraph.at(i).at(j).size() << " ";
        }
        std::cout << std::endl;
    }
    // std::cout << "mgraph size: " << mgraph.at(gD-1).size() << std::endl;
    std::cout << "before size: " << before.mfeatures.size() << ", gD data size: " << mglobalMapData.at(gD).mpoint3D.size() << std::endl;
    std::cout << "Query: " << maxq << ", Train: " << maxt << " Avg Distance: " << (float)sum/N << std::endl;
    std::cout << "inlier: " << inlier << ", inlier2: " << inlier2 <<std::endl;
}


bool mvo::Covisibilgraph::Projection(const cv::Vec3d& rvec, const cv::Vec3d& tvec, const cv::Point2f& pts, const cv::Point3f& world)
{
        const double theta = sqrt(rvec[0] * rvec[0] +
                             rvec[1] * rvec[1] + 
                             rvec[2] * rvec[2]);
        
        const double tvec_eig_0 = tvec[0];
        const double tvec_eig_1 = tvec[1];
        const double tvec_eig_2 = tvec[2];

        const double w1 = rvec[0] / theta;
        const double w2 = rvec[1] / theta;
        const double w3 = rvec[2] / theta;

        const double cos = ceres::cos(theta);
        const double sin = ceres::sin(theta);

        Eigen::Matrix<double, 3, 4> worldToCam;
        worldToCam << cos + w1 * w1 * (static_cast<double>(1) - cos), w1 * w2 * (static_cast<double>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<double>(1) - cos) + w2 * sin, tvec_eig_0,
            w1 * w2 * (static_cast<double>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<double>(1) - cos), w2 * w3 * (static_cast<double>(1) - cos) - w1 * sin, tvec_eig_1,
            w1 * w3 * (static_cast<double>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<double>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<double>(1) - cos), tvec_eig_2;

        Eigen::Matrix<double, 3, 1> pixel3d;
        Eigen::Vector4d worldHomoGen4d;

        worldHomoGen4d[0] = world.x;
        worldHomoGen4d[1] = world.y;
        worldHomoGen4d[2] = world.z;
        worldHomoGen4d[3] = static_cast<double>(1);


        Eigen::Matrix<double, 3, 3> Kd;
        Kd << mfocal, 0, mppx,
            0, mfocal, mppy,
            0, 0, 1;
        
        pixel3d = Kd.cast<double>() * worldToCam * worldHomoGen4d;

        double predicted_x = (pixel3d[0] / pixel3d[2]);
        double predicted_y = (pixel3d[1] / pixel3d[2]);

        Eigen::Vector2d residuals;
        residuals[0] = predicted_x - double(pts.x);
        residuals[1] = predicted_y - double(pts.y);

        // std::cout << abs(residuals[0]) << ", " << abs(residuals[1]) << std::endl;

        if(abs(residuals[0]) < REPROJECTERROR && abs(residuals[1]) < REPROJECTERROR) return true;
        else    return false;
}