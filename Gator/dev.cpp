#include <opencv2/opencv.hpp>
#include <random>
#include <complex>

#include "Gator.h"
#include "quality_metrics_OpenCV.h"

#define SHOWRES 
#define EXPORT

thread_local std::mt19937 gen{ std::random_device{}() };

template<typename T>
T random(T min, T max) {
	using dist = std::conditional_t<
		std::is_integral<T>::value,
		std::uniform_int_distribution<T>,
		std::uniform_real_distribution<T>
	>;
	return dist{ min, max }(gen);
}





int main()
{
	//Import and creation of Mats

	std::string path = "C:\\input_wframe.bmp";
	cv::Mat_<Pix> holoXSRC, holoYSRC;
	cv::Mat cmn;
	cv::Mat	src = cv::imread(path, cv::IMREAD_GRAYSCALE);
	ConvertToEdge(src, holoXSRC, holoYSRC, cmn);
	//define naming prefix
	std::string outPath = "outC";


	//Middle of image
	cv::Rect roi = cv::Rect(holoXSRC.cols / 4, holoXSRC.rows / 4, holoXSRC.cols / 2, holoXSRC.rows / 2);
	
	cv::Mat ref = Edge(src);
	AddFrame(ref);
	cv::Mat_<Pix> SRC2d = AddZeroPhase(ref);
	cv::Mat holoTest = SRC2d.clone();

	ref.convertTo(ref, CV_64F);

	//Variables definition

	const float minpx = 3.74e-6f;
	const float	minpy = 3.74e-6f;
	const float l = 632.8e-9f;
	const float mind = 200e-3f;

	const float maxP = 4e-6f;
	const float maxD = 201e-3f;
	const float iP = 5e-7f;
	const float iD = 10e-3f;
	float progress = 0;
	
	
	/*
	std::string filename;

	cv::Mat inputExp = ShowInt(holoTest);
	filename = "input";

	SaveResults(holoTest, path + filename, 1);


	ASDX(holoTest, mind, minpx, minpy, l);
	ASDY(holoTest, mind, minpx, minpy, l);
	NormAmp(holoTest);
	ASD(holoTest, -mind, minpx, minpy, l);


	cv::Mat intTest = ShowInt(holoTest);
	filename = "XtoYto2D";
	SaveResults(holoTest, path + filename, 1);

	*/
	/*
	here bitwise ops work
	cv::Mat holotemp;
	cv::bitwise_xor(holoXSRC, holoYSRC, holotemp);
	cv::imwrite("niewiem.bmp",ShowInt(holotemp));
	ShowPhase(holotemp);

*/
	/*ASDX(holoXSRC, mind, minpx, minpy, l);
	NormAmp(holoXSRC);
	ASDY(holoYSRC, mind, minpx, minpy, l);
	NormAmp(holoYSRC);


	cv::Mat Pha = ShowPhase(holoXSRC) + ShowPhase(holoYSRC);


	cv::Mat_<Pix> tester = ImportAsPhase(Pha);
	 
	ShowInt(tester);
	ShowPhase(tester);

	ASDX(tester, -mind, minpx, minpy, l);
	ShowInt(tester);
	ShowPhase(tester);


	ASDY(tester, -mind, minpx, minpy, l);
	ShowInt(tester);
	ShowPhase(tester);

*/

	LoopOfDeath(holoXSRC, holoYSRC,SRC2d, minpx, minpy, l, mind, maxP, maxD, iP, iD,ref,1);
	
	
	
	cv::destroyAllWindows();
	system("pause");
	return 0;
}


