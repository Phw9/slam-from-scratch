#include "Config.h"
#include "Init.h"


int main()
{
    std::ofstream rawData ("../main/image.txt", rawData.out | rawData.trunc);
	std::ifstream read ("../main/image.txt", read.in);
	std::ifstream readGTtvec ("../image/GTpose.txt", readGTtvec.in);
    cv::Mat temp(cv::Size(4,3), CV_64FC1, 0.0);
    
	

	if(!read.is_open())
	{
		std::cerr << "file can't read image" << std::endl;
		return 0;
	}
	std::deque<std::string> readImageName;
	int imageCurNum = 0;
	MakeTextFile(rawData, IMAGENUM);
	FileRead(readImageName, read);
	// GTPoseRead(tvecOfGT, readGTtvec);

    return 0;
}