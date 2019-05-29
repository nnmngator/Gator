#include "Gator.h"
#include "quality_metrics_OpenCV.h"

#define SHOWRES

cv::Mat ShowLogInt(cv::Mat m) {

	cv::Mat res = cv::Mat(m.rows, m.cols, CV_32FC1, cv::Scalar(0));
	m.forEach<Pix>([&](auto& m_pix, const int * pos) -> void {
		res.at<float>(pos[0], pos[1]) = std::logf(std::powf(std::abs(m_pix), 2.f));
	});
	cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
	res.convertTo(res, CV_8UC1);
#ifdef SHOWRES
	cv::namedWindow("log int", cv::WINDOW_NORMAL);
	cv::resizeWindow("log int", 512, 512);
	cv::imshow("log int", res);
	cv::waitKey(0);
#endif
	return res;
}

cv::Mat ShowInt(cv::Mat m) {

	cv::Mat res = cv::Mat(m.rows, m.cols, CV_32FC1, cv::Scalar(0));
	m.forEach<Pix>([&](auto& m_pix, const int * pos) -> void {
		res.at<float>(pos[0], pos[1]) = std::powf(std::abs(m_pix), 2.f);
	});
	cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
	res.convertTo(res, CV_8UC1);

#ifdef SHOWRES
	cv::namedWindow("int", cv::WINDOW_NORMAL);
	cv::resizeWindow("int", 1024, 1024);
	cv::imshow("int", res);
	cv::waitKey(0);
#endif

	return res;
}

cv::Mat ShowPhase(cv::Mat m) {
	cv::Mat res = cv::Mat(m.rows, m.cols, CV_32FC1, cv::Scalar(0));
	m.forEach<Pix>([&](auto& m_pix, const int * pos) -> void {
		res.at<float>(pos[0], pos[1]) = std::arg(m_pix);
	});
	cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
	res.convertTo(res, CV_8UC1);
#ifdef SHOWRES
	cv::namedWindow("phase", cv::WINDOW_NORMAL);
	cv::resizeWindow("phase", 1024, 1024);
	cv::imshow("phase", res);
	cv::waitKey(0);
#endif
	return res;
}

cv::Mat_<Pix> AddZeroPhase(cv::Mat intensity) {
	cv::Mat_<Pix> holo = cv::Mat_<Pix>(intensity.rows, intensity.cols, 0.f);
	holo.forEach([&](auto& holo_pix, const int* pos) -> void {
		//float phi = random(-CV_PI, CV_PI);
		float A = intensity.at<uchar>(pos[0], pos[1]);
		float phi = 0;
		//holo_pix = Pix(A*std::cosf(phi), A*std::sinf(phi));
		holo_pix = Pix(A*(std::cosf(phi)), A*(std::sinf(phi)));
	});
	return holo;
}

cv::Mat_<Pix> ImportAsPhase(cv::Mat phase) {
	cv::Mat_<Pix> holo = cv::Mat_<Pix>(phase.rows, phase.cols, 0.f);
	holo.forEach([&](auto& holo_pix, const int* pos)->void {
		float phi = phase.at<uchar>(pos[0], pos[1]);
		holo_pix = Pix(std::cosf(phi/255.0f *2.0f *CV_PI), std::sinf(phi / 255.0f *2.0f *CV_PI));
	});
		return holo;
}


void SaveResults(cv::Mat_<Pix> holo, std::string filename, bool crop)
{
	cv::Mat phase = ShowPhase(holo);
	cv::normalize(phase, phase, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite(filename + "pha.bmp", phase);
	cv::Mat inte = ShowInt(holo);
	cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite(filename + "int.bmp", inte);

	if (crop) {
		cv::Rect roi = cv::Rect(holo.cols / 4, holo.rows / 4, holo.cols / 2, holo.rows / 2);
		cv::imwrite(filename + "phaCrop.bmp", phase(roi));
		cv::imwrite(filename + "intCrop.bmp", inte(roi));
	}
}

cv::Mat Edge(cv::Mat m) {
	cv::Mat edges = cv::Mat::zeros(m.rows, m.cols, CV_8UC1);
	m.forEach<uchar>([&](uchar& pix, const int* pos) {
		int row = pos[0];
		int col = pos[1];

		if (row == 0 || col == 0 || row == (m.rows - 1) || col == (m.cols - 1))
		{
			return;
		}
		if (pix == 0)
		{
			return;
		}
		cv::Mat roi = m(cv::Rect(col - 1, row - 1, 3, 3));
		if (cv::countNonZero(roi) == 9) {
			edges.at<uchar>(row, col) = 0;
		}
		else {
			edges.at<uchar>(row, col) = 255;
		}

	});
	return edges;
}



cv::Mat EdgeX(cv::Mat m) {
	cv::Mat edges = cv::Mat::zeros(m.rows, m.cols, CV_8UC1);
	m.forEach<uchar>([&](uchar& pix, const int* pos) {
		int row = pos[0];
		int col = pos[1];

		if (row == 0 || col == 0 || row == (m.rows - 1) || col == (m.cols - 1))
		{
			return;
		}
		if (pix == 0)
		{
			return;
		}
		cv::Mat roi = m(cv::Rect(col - 1, row, 3, 1));
		if (cv::countNonZero(roi) == 3)
		{
			edges.at<uchar>(row, col) = 0;
		}
		else {
			edges.at<uchar>(row, col) = 255;
		}

	});
	return edges;
}

cv::Mat EdGey(cv::Mat m) {
	cv::Mat edges = cv::Mat::zeros(m.rows, m.cols, CV_8UC1);
	m.forEach<uchar>([&](uchar& pix, const int* pos) {
		int row = pos[0];
		int col = pos[1];

		if (row == 0 || col == 0 || row == (m.rows - 1) || col == (m.cols - 1))
		{
			return;
		}
		if (pix == 0)
		{
			return;
		}
		cv::Mat roi = m(cv::Rect(col, row - 1, 1, 3));
		if (cv::countNonZero(roi) == 3)
		{
			edges.at<uchar>(row, col) = 0;
		}
		else {
			edges.at<uchar>(row, col) = 255;
		}

	});
	return edges;
}



void AddFrame(cv::Mat m) {
	m.forEach<uchar>([&](uchar& pix, const int* pos) {
		int row = pos[0];
		int col = pos[1];

		if (row == 0 || col == 0 || row == (m.rows - 1) || col == (m.cols - 1))
		{
			pix = 255;
		}
		else
		{
			return;
		}
	});
}

void AddFrameVertical(cv::Mat m) {
	m.forEach<uchar>([&](uchar& pix, const int* pos) {
		int row = pos[0];
		int col = pos[1];

		if (col == 0 || col == (m.cols - 1))
		{
			pix = 255;
		}
		else
		{
			return;
		}
	});
}

void AddFrameHorizontal(cv::Mat m) {
	m.forEach<uchar>([&](uchar& pix, const int* pos) {
		int row = pos[0];
		int col = pos[1];

		if (row == 0 || row == (m.rows - 1) )
		{
			pix = 255;
		}
		else
		{
			return;
		}
	});
}



void ConvertToEdge(cv::Mat src, cv::Mat_<Pix> &holoX, cv::Mat_<Pix> &holoY, cv::Mat &cmn) {
	//calculate both derivatives (frame is lost)
	cv::Mat edgesX = EdgeX(src);
	cv::Mat edgesY = EdGey(src);
	//Calculate common part of both images 
	cmn = edgesX.mul(edgesY, 1.0 / 255.0);
	//Subtract common part from Y image (avoid recalculating common parts in 2nd propagation)
	cv::subtract(edgesY, cmn, edgesY);

	//Reintroduce frame

	//Should i add frame that is propagated or one that is lost?
	AddFrameVertical(edgesY);
	AddFrameHorizontal(edgesX);

	cv::imwrite("x.bmp", edgesX);
	cv::imwrite("y.bmp", edgesY);

	//Create RE IM array
	//image imported as RE part, IM left empty(0)

	//X
	holoX = AddZeroPhase(edgesX);
	//Y
	holoY = AddZeroPhase(edgesY);
}






void FFTX(cv::Mat m) {
	cv::dft(m, m, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);
}

void IFFTX(cv::Mat m) {
	cv::idft(m, m, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS | cv::DFT_SCALE);
}

void FFT(cv::Mat m) {
	cv::dft(m, m, cv::DFT_COMPLEX_OUTPUT);
}

void IFFT(cv::Mat m) {
	cv::idft(m, m, cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
}

void Lens(cv::Mat m, float f, float l, float px, float py) {
	int N = m.rows;
	int M = m.cols;
	m.forEach<Pix>([&](auto& holo_pix, const int* pos) -> void {
		int n = pos[0] - N / 2, m = pos[1] - M / 2;
		holo_pix *= std::exp(Pix(0, -CV_PI / (l * f) * (n*n*px*px + m * m*py*py)));
	});
}

void NormAmp(cv::Mat m) {
	m.forEach<Pix>([&](auto& holo_pix, const int* pos) -> void {
		float re = std::real(holo_pix);
		float im = std::imag(holo_pix);
		float fi = std::atan2f(im, re);
		holo_pix = Pix(std::cosf(fi), std::sinf(fi));
	});
}



void ASDX(cv::Mat holo, float d, float px, float py, float l = 632.8e-9) {
	FFTShiftX<Pix>(holo);
	FFTX(holo);
	FFTShiftX<Pix>(holo);

	int N = holo.rows;
	int M = holo.cols;

	holo.forEach<Pix>([&](auto& holo_pix, const int* pos) -> void {
		int n = pos[0] - N / 2 - 1, m = pos[1] - M / 2 - 1;
		float f = std::pow(m / (M * py), 2.f);
		holo_pix *= std::exp(Pix(0, -CV_PI * l*d*f));
	});

	FFTShiftX<Pix>(holo);
	IFFTX(holo);
	FFTShiftX<Pix>(holo);
}

void ASDY(cv::Mat holo, float d, float px, float py, float l = 632.8e-9) {

	FFTShiftY<Pix>(holo);
	cv::rotate(holo, holo, cv::ROTATE_90_COUNTERCLOCKWISE);
	FFTX(holo);
	cv::rotate(holo, holo, cv::ROTATE_90_CLOCKWISE);
	FFTShiftY<Pix>(holo);

	int N = holo.rows;
	int M = holo.cols;

	holo.forEach<Pix>([&](auto& holo_pix, const int* pos) -> void {
		int n = pos[0] - N / 2 - 1, m = pos[1] - M / 2 -1 ;
		float f = std::pow(n / (N * px), 2.f);
		holo_pix *= std::exp(Pix(0, -CV_PI * l*d*f));
	});

	FFTShiftY<Pix>(holo);
	cv::rotate(holo, holo, cv::ROTATE_90_COUNTERCLOCKWISE);
	IFFTX(holo);
	cv::rotate(holo, holo, cv::ROTATE_90_CLOCKWISE);
	FFTShiftY<Pix>(holo);
}

void ASD(cv::Mat holo, float d, float px, float py, float l = 632.8e-9) {

	FFTShift<Pix>(holo);
	FFT(holo);
	FFTShift<Pix>(holo);

	int N = holo.rows;
	int M = holo.cols;

	holo.forEach<Pix>([&](auto& holo_pix, const int* pos) -> void {
		int n = pos[0] - N / 2 - 1, m = pos[1] - M / 2 - 1;
		float f = std::pow(n / (N * px), 2.f) + std::pow(m / (M * py), 2.f);
		holo_pix *= std::exp(Pix(0, -CV_PI * l*d*f));
	});

	FFTShift<Pix>(holo);
	IFFT(holo);
	FFTShift<Pix>(holo);
}


cv::Mat_<Pix> AddAmplitudes(cv::Mat m1, cv::Mat m2) {
	cv::Mat_<Pix> holo = m1.clone();
	holo.forEach([&](auto& holo_pix, const int* pos)->void {		
		auto a1 = std::abs(holo_pix);
		auto a2 = std::abs(m2.at<Pix>(pos));
		holo_pix = std::polar<float>(a1 + a2, 0.f);
	});
	return holo;
}

void LoopOfDeath(cv::Mat holoXSRC, cv::Mat holoYSRC,cv::Mat SRC2d, float minpx, float minpy, float l, float mind, float maxP, float maxD, float iP, float iD, cv::Mat ref, bool FileWrite)
{
	cv::Rect roi = cv::Rect(holoXSRC.cols / 4, holoXSRC.rows / 4, holoXSRC.cols / 2, holoXSRC.rows / 2);
	for (float d = mind; d < maxD; d = d + iD)
	{


		for (float px = minpx; px < maxP; px = px + iP)
		{
			float py = px;
			cv::Mat holoX = holoXSRC.clone();
			cv::Mat holoY = holoYSRC.clone();

			// X
			ASDX(holoX, d, px, py, l);
			NormAmp(holoX);
			// Y
			ASDY(holoY, d, px, py, l);
			NormAmp(holoY);
			//// X + Y
			cv::Mat holo3;
			cv::Mat test3;
			ASDX(holoX, -d, px, py, l);
			ASDY(holoY, -d, px, py, l);

			auto xprop = ShowInt(holoX);

			auto yprop = ShowInt(holoY);

			float xnonz = (float)cv::countNonZero(xprop) / xprop.total();
			float ynonz = (float)cv::countNonZero(yprop) / yprop.total();

			system("cls");
			//std::cout << "xnonz = " << xnonz << "\nynonz= " << ynonz << std::endl;

			cv::addWeighted(holoX, xnonz, holoY, ynonz, 1, holo3);
			/*
			bitwise operations dont seem to work
				cv::bitwise_or(holoX, holoY, test3);
				imwrite("xor.bmp", ShowInt(test3));
	*/

			std::stringstream ss;
			ss << "export/374um/Dist" << d * 1e+3 << "mm.bmp";
			std::string filename = ss.str();

			cv::Mat inte = ShowInt(holo3);
			cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
			std::cout << filename;
			if (FileWrite) {
				cv::imwrite(filename, inte);
			}


			ss.str("");
			ss.clear();

			inte.convertTo(inte, CV_64F);
			double SSIM = qm::ssim(ref, inte, 4, 1);
			//			double PSNR = qm::psnr(ref, inte, 4);
			std::ofstream ostrm("1D.txt", std::ofstream::app);
			ostrm << d * 1e+3 << "," << px * 1e+6 << "," << SSIM << '\n';
			ostrm.close();
		//	std::cout << "Propagation Distance\nCurrent: " << d << "m, Max:" << maxD << "m \nSampling\nCurrent: " << px * 1e+6 << "um, Max:" << maxP * 1e+6 << "um \nIteration: " << progress << " out of " << nostep << "\nProgress: " << progress / nostep * 100 << "\n" << "dif = " << sum << "\n" << std::endl;
			//progress++;
			std::cout << "SSIM = " << SSIM << "\n" << std::endl;

			ss << "export/374um/CropDist" << d * 1e+3 << "mm.bmp";
			filename = ss.str();
			inte = ShowInt(holo3(roi));
			cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
			if (FileWrite) {
				cv::imwrite(filename, inte);
			}
			//2Dprop
			cv::Mat holo2d = SRC2d.clone();

			ASD(holo2d, d, px, py, l);
			NormAmp(holo2d);
			ASD(holo2d, -d, px, py, l);
			cv::Mat h2d = ShowInt(holo2d);
			h2d.convertTo(h2d, CV_64F);

			SSIM = qm::ssim(ref, h2d, 4, 1);
			cv::imwrite("2prop.bmp", h2d);
			cv::imwrite("ref.bmp", ref);

			std::ofstream ostrm2("2D.txt", std::ofstream::app);
			ostrm2 << d * 1e+3 << "," << px * 1e+6 << "," << SSIM << '\n';
			ostrm2.close();


			ss.str("");
			ss.clear();
		}
	}

}