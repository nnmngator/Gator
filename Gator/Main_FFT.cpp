#include <opencv2\opencv.hpp>

typedef std::complex<float> Pix;

cv::Mat ShowLogInt(cv::Mat m) {
	cv::namedWindow("log int", cv::WINDOW_NORMAL);
	cv::resizeWindow("log int", 512, 512);
	cv::Mat res = cv::Mat(m.rows, m.cols, CV_32FC1, cv::Scalar(0));
	m.forEach<Pix>([&](auto& m_pix, const int * pos) -> void {
		res.at<float>(pos[0], pos[1]) = std::logf(std::powf(std::abs(m_pix), 2.f));
	});
	cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
	res.convertTo(res, CV_8UC1);
	cv::imshow("log int", res);
	cv::waitKey(0);
	return res;
}

cv::Mat ShowPhase(cv::Mat m) {
	cv::namedWindow("phase", cv::WINDOW_NORMAL);
	cv::resizeWindow("phase", 512, 512);
	cv::Mat res = cv::Mat(m.rows, m.cols, CV_32FC1, cv::Scalar(0));
	m.forEach<Pix>([&](auto& m_pix, const int * pos) -> void {
		res.at<float>(pos[0], pos[1]) = std::arg(m_pix);
	});
	cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
	res.convertTo(res, CV_8UC1);
	cv::imshow("phase", res);
	cv::waitKey(0);
	return res;
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

int main3 (int argc, char* argv[]) {

	cv::Mat lena = cv::imread("C:/LS/lena512.bmp", cv::IMREAD_GRAYSCALE);

	cv::Mat_<Pix> holo = cv::Mat_<Pix>(lena.rows, lena.cols, 0.f);
	holo.forEach([&](auto& holo_pix, const int* pos) -> void {
		holo_pix = Pix(lena.at<uchar>(pos[0], pos[1]), 0);
	});
	cv::copyMakeBorder(holo, holo, 256, 256, 256, 256, cv::BORDER_CONSTANT, cv::Scalar(255,255));

	ShowPhase(holo);
	FFTShift<Pix>(holo);
	cv::dft(holo, holo, cv::DFT_COMPLEX_OUTPUT);
	FFTShift<Pix>(holo);
	cv::Mat cv_dft = ShowPhase(holo);
	cv::Mat ls_dft = cv::imread("C:/LS/leny/fft_center.bmp", cv::IMREAD_GRAYSCALE);

	cv::imwrite("C:/LS/leny/cv_phase_fft_ifft.bmp", cv_dft);

	cv::Mat diff = cv_dft - ls_dft;
	double minVal, maxVal;
	cv::minMaxLoc(diff, &minVal, &maxVal);
	std::cout << minVal << " " << maxVal << "\n";

	cv::imshow("diff", diff);
	cv::waitKey(0);

	diff = ls_dft - cv_dft;

	cv::imshow("diff", diff);
	cv::waitKey(0);

	return 0;
}
