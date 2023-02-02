#include "Init.h"


#define numofimage 4540

mvo::Initializer::Initializer(){};
mvo::Initializer::Initializer(std::vector<cv::KeyPoint> refKeys1, std::vector<cv::KeyPoint> refKeys2):
                                mvKeys1{refKeys1}, mvKeys2{refKeys2}{};

float mvo::Initializer::CheckHomography(const std::vector<cv::Point2f>& refKeys1, const std::vector<cv::Point2f>& refKeys2, float sigma)
{
    if(refKeys1.size() != refKeys2.size())
    {
        std::cerr << "failed CheckHomography" << std::endl;
        return -1;
    }
    const int N = refKeys1.size();

    mHomography = cv::findHomography(refKeys1, refKeys2);
    const float h11 = mHomography.at<float>(0,0);
    const float h12 = mHomography.at<float>(0,1);
    const float h13 = mHomography.at<float>(0,2);
    const float h21 = mHomography.at<float>(1,0);
    const float h22 = mHomography.at<float>(1,1);
    const float h23 = mHomography.at<float>(1,2);
    const float h31 = mHomography.at<float>(2,0);
    const float h32 = mHomography.at<float>(2,1);
    const float h33 = mHomography.at<float>(2,2);

    cv::findHomography(refKeys2, refKeys1, mInvHomography);
    const float h11inv = mInvHomography.at<float>(0,0);
    const float h12inv = mInvHomography.at<float>(0,1);
    const float h13inv = mInvHomography.at<float>(0,2);
    const float h21inv = mInvHomography.at<float>(1,0);
    const float h22inv = mInvHomography.at<float>(1,1);
    const float h23inv = mInvHomography.at<float>(1,2);
    const float h31inv = mInvHomography.at<float>(2,0);
    const float h32inv = mInvHomography.at<float>(2,1);
    const float h33inv = mInvHomography.at<float>(2,2);

    
    float score = 0.0f;
    const float th = 5.991f;

    const float invSigmaSquare = (1.0f)/(sigma*sigma);
    
    for(int i = 0; i < N; i++)
    {
        const cv::Point2f &kp1 = refKeys1.at(i); // .first
        const cv::Point2f &kp2 = refKeys2.at(i); // .second

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        const float w2in1inv = (1.0f)/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        const float chiSquare1 = squareDist1*invSigmaSquare;
        // std::cout << "SquareDist1: " << squareDist1;
        // std::cout << "   chiSquare1: " << chiSquare1 << std::endl;
        if(chiSquare1 < th)
            score += th - chiSquare1;
        // Reprojection error in second image
        // x1in2 = H21*x1

        const float w1in2inv = (1.0f)/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        const float chiSquare2 = squareDist2*invSigmaSquare;
        // std::cout << "SquareDist2: " << squareDist2;
        // std::cout << "   chiSquare2: " << chiSquare2 << std::endl;
        if(chiSquare2 < th)
            score += th - chiSquare2;
    }
    return score;
}


// input Matrix, pts1, pts2
float mvo::Initializer::CheckFundamental(const std::vector<cv::Point2f>& refKeys1, const std::vector<cv::Point2f>& refKeys2, float sigma)
{
    if(refKeys1.size() != refKeys2.size())
    {
        std::cerr << "failed CheckFundamental" << std::endl;
        return -1;
    }

    const int N = refKeys1.size();
    mFundamental = cv::findFundamentalMat(refKeys1, refKeys2);

    const float f11 = mFundamental.at<float>(0,0);
    const float f12 = mFundamental.at<float>(0,1);
    const float f13 = mFundamental.at<float>(0,2);
    const float f21 = mFundamental.at<float>(1,0);
    const float f22 = mFundamental.at<float>(1,1);
    const float f23 = mFundamental.at<float>(1,2);
    const float f31 = mFundamental.at<float>(2,0);
    const float f32 = mFundamental.at<float>(2,1);
    const float f33 = mFundamental.at<float>(2,2);

    float score = 0.0f;

    const float th = 3.841f;
    const float thScore = 5.991f;

    const float invSigmaSquare = (1.0f)/(sigma*sigma);

    for(int i = 0; i < N; i++)
    {
        const cv::Point2f &kp1 = refKeys1.at(i); //.first
        const cv::Point2f &kp2 = refKeys2.at(i); //.second

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        const float num2 = a2*u2+b2*v2+c2;
        // std::cout << "a2: " << a2 << " b2: " << b2 << std::endl;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1<th)
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        // std::cout << "a1: " << a1 << " b1: " << b1 << std::endl;
        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2 < th)
            score += thScore - chiSquare2;
    }
    return score;
}




void FileRead(std::deque<std::string>& v, std::ifstream &fin)
{
	std::string line;
	while(true)
	{
		getline(fin, line);
		if(fin.eof()) break;
		v.emplace_back(line);
	}
}
void MakeTextFile(std::ofstream& fout, const int& imageNum)
{
    for(int i = 0; i<imageNum+1; i++)
	{
		fout << "../image/image_0/";
		fout.width(6);
		fout.fill('0');
		// fout.right;
		fout << i;
		fout << ".png" << std::endl;
	}
}


void GTPoseRead(std::vector<cv::Vec3f>& v, std::ifstream& fin)
{
	char buf[100];
    cv::Vec3f temp;
	int numOfGT = 0;

	double tempX, tempY, tempZ;
	while(true)
	{	
		for(int i = 0; i< 12; i++)
		{
			if(i == 3)
			{
				fin.getline(buf, 100, ' ');
				tempX = atof(buf);
				temp[0] = tempX;
			}
            else if(i == 7)
			{
				fin.getline(buf, 100, ' ');
                tempY = atof(buf);
				temp[1] = tempY;
			}
			else if(i == 11)
			{
				fin.getline(buf, 100, '\n');
				tempZ = atof(buf);
				temp[2] = tempZ;
				break;
			}
            else
            {
                fin.getline(buf,100, ' ');
            }
		}
		numOfGT++;
		v.emplace_back(std::move(temp));
		if(numOfGT == numofimage) break;
		// if(fin.eof()) break;
	}
}


// /*
//     pangolin :: visualioze GT,RE Trajectory 
//     cv :: visualize 2D features points 
//     (current Features, Triangulated Features, attached Features)
// */


Viewer::MyVisualize::MyVisualize(int width,int height)
{
    this->window_width=width;
    this->window_height=height;
    this->window_ratio=(float)width/height;
}

void Viewer::MyVisualize::initialize()
{
    pangolin::CreateWindowAndBind("TrajectoryViewer", window_width, window_height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Viewer::MyVisualize::active_cam()
{
    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(window_width, window_height, 20, 20, 512, 389, 0.1, 100),
    pangolin::ModelViewLookAt(0, -100, -0.1, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -window_ratio)
                                .SetHandler(new pangolin::Handler3D(s_cam)); 
    d_cam.Activate(s_cam);
}

// pts1: GT Pose, pts2: Pose, pts3: 3D Points, pts4: FOV of 3D Points
void Viewer::MyVisualize::DrawPoint(const std::vector<mvo::MapData>& v,
                                    const std::vector<cv::Vec3f>& gtPose)
{
    int N = v.size();
    int M = gtPose.size();
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    if(v.size() == 0)
	{
        return;
    }
    else
	{
        //빨간색(첫번째) : 구한포즈, 파란색(두번째) : 지티포즈, 검은색(세번째): 모든 맵포인트 , 자홍색(네번째) : 현재 키프레임의 맵포인트
        glPointSize(3);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,0.0); //

        for(int i = 0; i < N; i++)
	    {
            glVertex3d(v.at(i).mglobalTranslation[0], 0, v.at(i).mglobalTranslation[2]);
            // std::cout << "tvec" << i << ": " << tvec.at(i)[0] << ", " << tvec.at(i)[1] << ", " << tvec.at(i)[2] << std::endl;
        }
        glEnd();

        glPointSize(3);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,1.0);

        for(int i = 0; i < M; i++)
	    {
            glVertex3f(gtPose.at(i)[0], 0, gtPose.at(i)[2]);
            // std::cout << "gtPose" << i << ": " << gtPose.at(i)[0] << ", " << gtPose.at(i)[1] << ", " << gtPose.at(i)[2] << std::endl;
        }
        glEnd();

        glPointSize(1);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,0.0);

        for(int i = 0; i < N; i++)
	    {
            int V = v.at(i).mpoint3D.size();
            for(int j = 0; j < V; j++)
            {
                glVertex3f(v.at(i).mpoint3D.at(j).x,
                            0,
                           v.at(i).mpoint3D.at(j).z);
            // std::cout << "allOfPoints" << i << ": " << allOfPoints.at(i).mworldMapPointsV.at(j).x << ", " << allOfPoints.at(i).mworldMapPointsV.at(j).y << ", " <<allOfPoints.at(i).mworldMapPointsV.at(j).z << std::endl;
           }
        }
        glEnd();      

        glPointSize(2);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,1.0);
        int P = v.at(v.size()-1).mpoint3D.size();
        for(int i = 0; i < P; i++)
	    {
            glVertex3f(v.at(v.size()-1).mpoint3D.at(i).x, 0, v.at(v.size()-1).mpoint3D.at(i).z);
            // std::cout << "fovPoints" << i << ": " << fovPoints.at(i).x << ", " << fovPoints.at(i).y << ", " << fovPoints.at(i).z << std::endl;
        }
        glEnd();            
    }

}
void Viewer::MyVisualize::DrawPoint(const std::vector<cv::Vec3d>& tvec, 
                const std::vector<cv::Vec3f>& gtPose,
                const std::vector<mvo::Triangulate>& allOfPoints,
                const std::vector<cv::Point3f>& fovPoints)
{
    int M = tvec.size();
    int N = gtPose.size();
    int O = allOfPoints.size();
    int P = fovPoints.size();
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    if(tvec.size() == 0 || gtPose.size() == 0)
	{
        return;
    }
    else
	{
        //빨간색(첫번째) : 구한포즈, 파란색(두번째) : 지티포즈, 검은색(세번째): 모든 맵포인트 , 자홍색(네번째) : 현재 키프레임의 맵포인트
        glPointSize(3);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,0.0); //

        for(int i = 0; i < M; i++)
	    {
            glVertex3d(tvec.at(i)[0], 0, tvec.at(i)[2]);
            // std::cout << "tvec" << i << ": " << tvec.at(i)[0] << ", " << tvec.at(i)[1] << ", " << tvec.at(i)[2] << std::endl;
        }
        glEnd();

        glPointSize(3);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,1.0);

        for(int i = 0; i < N; i++)
	    {
            glVertex3f(gtPose.at(i)[0], 0, gtPose.at(i)[2]);
            // std::cout << "gtPose" << i << ": " << gtPose.at(i)[0] << ", " << gtPose.at(i)[1] << ", " << gtPose.at(i)[2] << std::endl;
        }
        glEnd();

        glPointSize(1);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,0.0);

        for(int i = 0; i < O; i++)
	    {
            int J = allOfPoints.at(i).mworldMapPointsV.size();
            for(int j = 0; j < J; j++)
            {
                glVertex3f(allOfPoints.at(i).mworldMapPointsV.at(j).x,
                           0,
                           allOfPoints.at(i).mworldMapPointsV.at(j).z);
                // std::cout << "allOfPoints" << i << ": " << allOfPoints.at(i).mworldMapPointsV.at(j).x << ", " << allOfPoints.at(i).mworldMapPointsV.at(j).y << ", " <<allOfPoints.at(i).mworldMapPointsV.at(j).z << std::endl;
            }
        }
        glEnd();      

        glPointSize(2);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,1.0);

        for(int i = 0; i < P; i++)
	    {
            glVertex3f(fovPoints.at(i).x, 0, fovPoints.at(i).z);
            // std::cout << "fovPoints" << i << ": " << fovPoints.at(i).x << ", " << fovPoints.at(i).y << ", " << fovPoints.at(i).z << std::endl;
        }
        glEnd();            
    }
}

// circle is before, rectangle is after
cv::Mat Viewer::MyVisualize::DrawFeatures(cv::Mat& src, 
                                                std::vector<cv::Point2f>& beforePoints, 
                                                std::vector<cv::Point2f>& afterPoints)
{
    int N = beforePoints.size();
    for (int i = 0; i < N; i++)
	{
        //random color
        int rgb[3];
        rgb[0]=rand()%256;
        rgb[1]=rand()%256;
        rgb[2]=rand()%256;
        cv::line(src, beforePoints[i], afterPoints[i],cv::Scalar(rgb[0],rgb[1],rgb[2]),1,8,0);
        cv::circle(src, beforePoints[i], 5, cv::Scalar(rgb[0], rgb[1], rgb[2]), 1, 8, 0); //2d features  
        // circle(src, afterPoints[i], 6, Scalar(rgb[0], rgb[1], rgb[2]), 1, 8, 0); //3d points
        cv::rectangle(src, cv::Rect(cv::Point(afterPoints[i].x-5,afterPoints[i].y-5),
        cv::Point(afterPoints[i].x+5,afterPoints[i].y+5)), cv::Scalar(rgb[0],rgb[1],rgb[2]),1,8,0);//projection 3d points
    }
    return src;    
}
