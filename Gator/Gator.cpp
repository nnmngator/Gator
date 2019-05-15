#include "Gator.h"


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
		//holo_pix = Pix(A*std::cosf(phi), A*std::sinf(phi));
		holo_pix = Pix(A, 0);
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


void ConvertToEdge(cv::Mat src, cv::Mat_<Pix> &holoX, cv::Mat_<Pix> &holoY, cv::Mat &cmn) {
	//calculate both derivatives (frame is lost)
	cv::Mat edgesX = EdgeX(src);
	cv::Mat edgesY = EdGey(src);
	//Calculate common part of both images 
	cmn = edgesX.mul(edgesY, 1.0 / 255.0);
	//Subtract common part from Y image (avoid recalculating common parts in 2nd propagation)
	cv::subtract(edgesY, cmn, edgesY);

	//Reintroduce frame
	AddFrame(edgesX);
	AddFrame(edgesY);
	//Create RE IM array
	//image imported as RE part, IM left empty(0)

	//X
	holoX = AddZeroPhase(edgesX);
	//Y
	holoY = AddZeroPhase(edgesY);
}


template<class T>
void FFTShift(cv::Mat m) {
	m.forEach<T>([&](T& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		if (row >= m.rows / 2) return;
		int next_row = row + m.rows / 2;
		int next_col = (col < (m.cols / 2)) ? (col + (m.cols / 2)) : (col - (m.cols / 2));
		T& next_pix = m.at<T>(next_row, next_col);
		std::swap(next_pix, pix);
	});
}

template<class T>
void FFTShiftX(cv::Mat m) {
	m.forEach<T>([&](T& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		if (col >= m.cols / 2) return;
		int next_row = row;
		int next_col = col + m.cols / 2;
		T& next_pix = m.at<T>(next_row, next_col);
		std::swap(next_pix, pix);
	});
}

template<class T>
void FFTShiftY(cv::Mat m) {
	m.forEach<T>([&](T& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		if (row >= m.rows / 2) return;
		int next_row = row + m.rows / 2;
		int next_col = col;
		T& next_pix = m.at<T>(next_row, next_col);
		std::swap(next_pix, pix);
	});
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
		int n = pos[0] - N / 2, m = pos[1] - M / 2;
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
		int n = pos[0] - N / 2, m = pos[1] - M / 2;
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
		int n = pos[0] - N / 2, m = pos[1] - M / 2;
		float f = std::pow(n / (N * px), 2.f) + std::pow(m / (M * py), 2.f);
		holo_pix *= std::exp(Pix(0, -CV_PI * l*d*f));
	});

	FFTShift<Pix>(holo);
	IFFT(holo);
	FFTShift<Pix>(holo);
}