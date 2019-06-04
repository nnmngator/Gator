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

	std::string path = "C:\\hud2k.bmp";
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
	const float mind = 600e-3f;

	//Ultra bad 2D prop, okay 1D
	//mind = 450e-3f;
	//mind = 800e-3f;
	//mind = 700e-3f;
	const float maxP = 4e-6f;
	const float maxD = 201e-3f;
	const float iP = 5e-7f;
	const float iD = 10e-3f;
	float progress = 0;

	/*
		ASDX(holoXSRC, mind,minpx,minpy,l);
		SaveResults(holoXSRC, "1stProp");

		NormAmp(holoXSRC);
		SaveResults(holoXSRC, "1stPropNorm");

		ASDX(holoXSRC, -mind,minpx,minpy,l);
		SaveResults(holoXSRC, "2ndProp");

	*/


	//FRAME WITHOUT CORNERS 
	//Frame experiments
		//SaveResults(holoXSRC, "FrameX");
		//SaveResults(holoYSRC, "FrameY");
		//SaveResults(SRC2d, "Frame2D");
		//frames in input are okay

		//THE MAIN 1D VS 2D LOGIC
		/*SaveResults(holoXSRC,"ramkaX");
		ShowInt(holoXSRC(roi));
		ASDX(holoXSRC, mind, minpx, minpy, l);
		NormAmp(holoXSRC);
		ASDX(holoXSRC, -mind, minpx, minpy, l);
		ShowInt(holoXSRC(roi));

		ShowInt(holoYSRC(roi));

		ASDY(holoYSRC, mind, minpx, minpy, l);
		NormAmp(holoYSRC);
		ASDY(holoYSRC, -mind, minpx, minpy, l);

		ShowInt(holoXSRC(roi));
		ShowInt(holoYSRC(roi));


		SaveResults(holoXSRC, "holoX", 1);
		SaveResults(holoYSRC, "holoY", 1);

		auto holo3 = AddAmplitudes(holoXSRC, holoYSRC);
		SaveResults(holo3, "holo3", 1);

		ShowInt(holo3(roi));
	*/
	ShowPhase(SRC2d);
	ASD(SRC2d, mind, minpx, minpy, l);
	ShowPhase(SRC2d);

	NormAmp(SRC2d);
	ShowPhase(SRC2d);

	ASD(SRC2d, -mind, minpx, minpy, l);

	ShowInt(SRC2d(roi));
	SaveResults(SRC2d, "odtworzenieKlasyczne");
	//	LoopOfDeath(holoXSRC, holoYSRC,SRC2d, minpx, minpy, l, mind, maxP, maxD, iP, iD,ref,1);



	cv::destroyAllWindows();
	system("pause");
	return 0;
}


