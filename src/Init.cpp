#include "Init.h"

#define numofimage 4540

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
    pangolin::ProjectionMatrix(window_width, window_height, 20, 20, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(0, -100, -0.1, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -window_ratio)
                                .SetHandler(new pangolin::Handler3D(s_cam)); 
    d_cam.Activate(s_cam);
}

// pts1: GT Pose, pts2: Pose, pts3: 3D Points, pts4: FOV of 3D Points
void Viewer::MyVisualize::DrawPoint(const std::vector<cv::Vec3d>& tvec, 
                const std::vector<cv::Vec3f>& gtPose,
                const std::vector<mvo::Triangulate>& allOfPoints,
                const std::vector<cv::Point3f>& fovPoints)
{
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    if(tvec.size()==0 || gtPose.size()==0)
	{
        return;
    }
    else
	{
        //빨간색(첫번째) : 구한포즈, 파란색(두번째) : 지티포즈, 검은색(세번째): 모든 맵포인트 , 자홍색(네번째) : 현재 키프레임의 맵포인트
        glPointSize(3);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,0.0); //

        for(int i=0;i<tvec.size();i++)
	    {
            glVertex3d(tvec.at(i)[0], 0, tvec.at(i)[2]);
            // std::cout << "tvec" << i << ": " << tvec.at(i)[0] << ", " << tvec.at(i)[1] << ", " << tvec.at(i)[2] << std::endl;
        }
        glEnd();

        glPointSize(3);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,1.0);

        for(int i=0;i<gtPose.size();i++)
	    {
            glVertex3f(gtPose.at(i)[0], 0, gtPose.at(i)[2]);
            // std::cout << "gtPose" << i << ": " << gtPose.at(i)[0] << ", " << gtPose.at(i)[1] << ", " << gtPose.at(i)[2] << std::endl;
        }
        glEnd();

        glPointSize(1);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,0.0);

        for(int i = 0; i < allOfPoints.size(); i++)
	    {
           for(int j = 0; j < allOfPoints.at(i).mworldMapPointsV.size(); j++)
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

        for(int i = 0; i < fovPoints.size(); i++)
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
    for (int i = 0; i < beforePoints.size(); i++)
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
