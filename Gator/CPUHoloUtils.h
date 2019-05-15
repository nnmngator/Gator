#pragma once

#include <opencv2\opencv.hpp>


#define SHOW_RESULTS

typedef std::complex<float> Pix;

cv::Mat ShowLogInt(cv::Mat m) {
	cv::namedWindow("log int", cv::WINDOW_NORMAL);
	cv::resizeWindow("log int", 512, 512);
	cv::Mat res = cv::Mat(m.rows, m.cols, CV_32FC1, cv::Scalar(0));
	m.forEach<Pix>([&](auto& m_pix, const int * pos) -> void {
		res.at<float>(pos[0], pos[1]) = std::logf(1.0f + std::powf(std::abs(m_pix), 2.f));
	});
	cv::normalize(res, res, 0, 1, cv::NORM_MINMAX);
	res.convertTo(res, CV_32FC1);

#ifdef SHOW_RESULTS
	cv::imshow("log int", res);
	cv::waitKey(0);
#endif

	return res;
}

cv::Mat ShowInt(cv::Mat m) {
#ifdef SHOW_RESULTS
	cv::namedWindow("int", cv::WINDOW_NORMAL);
	cv::resizeWindow("int", 1024, 1024);
#endif

	cv::Mat res = cv::Mat(m.rows, m.cols, CV_32FC1, cv::Scalar(0));
	m.forEach<Pix>([&](auto& m_pix, const int * pos) -> void {
		res.at<float>(pos[0], pos[1]) = std::powf(std::abs(m_pix), 2.f);
	});
	cv::normalize(res, res, 0, 1, cv::NORM_MINMAX);
	res.convertTo(res, CV_32FC1);

#ifdef SHOW_RESULTS
	cv::imshow("int", res);
	cv::waitKey(0);
#endif
	return res;
}

cv::Mat ShowPhase(cv::Mat m) {
	cv::namedWindow("phase", cv::WINDOW_NORMAL);
	cv::resizeWindow("phase", 1024, 1024);
	cv::Mat res = cv::Mat(m.rows, m.cols, CV_32FC1, cv::Scalar(0));
	m.forEach<Pix>([&](auto& m_pix, const int * pos) -> void {
		res.at<float>(pos[0], pos[1]) = std::arg(m_pix);
	});
	cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
	res.convertTo(res, CV_8UC1);

#ifdef SHOW_RESULTS
	cv::imshow("phase", res);
	cv::waitKey(0);
#endif
	return res;
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
