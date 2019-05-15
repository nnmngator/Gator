#include <opencv2/opencv.hpp>
#include <random>
#include <complex>

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

cv::Mat ShowInt(cv::Mat m) {
	cv::namedWindow("int", cv::WINDOW_NORMAL);
	cv::resizeWindow("int", 1024, 1024);
	cv::Mat res = cv::Mat(m.rows, m.cols, CV_32FC1, cv::Scalar(0));
	m.forEach<Pix>([&](auto& m_pix, const int * pos) -> void {
		res.at<float>(pos[0], pos[1]) = std::powf(std::abs(m_pix), 2.f);
	});
	cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
	res.convertTo(res, CV_8UC1);
	cv::imshow("int", res);
	cv::waitKey(0);
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
	cv::imshow("phase", res);
	cv::waitKey(0);
	return res;
}

cv::Mat EdgeX(cv::Mat m) {
	cv::Mat edges = cv::Mat::zeros(m.rows, m.cols,CV_8UC1);
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
		cv::Mat roi = m(cv::Rect(col - 1, row , 3, 1));
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
		cv::Mat roi = m(cv::Rect(col , row-1, 1, 3));
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

void Lens(cv::Mat m, float f, float l, float px, float py) {
	int N = m.rows;
	int M = m.cols;
	m.forEach<Pix>([&](auto& holo_pix, const int* pos) -> void {
		int n = pos[0] - N / 2, m = pos[1] - M / 2;
		holo_pix *= std::exp(Pix(0, -CV_PI / (l * f) * (n*n*px*px + m*m*py*py)));
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
		float f =  std::pow(m / (M * py), 2.f);
		holo_pix *= std::exp(Pix(0, -CV_PI*l*d*f));
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
		holo_pix *= std::exp(Pix(0, -CV_PI*l*d*f));
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
		holo_pix *= std::exp(Pix(0, -CV_PI*l*d*f));
	});

	FFTShift<Pix>(holo);
	IFFT(holo);	
	FFTShift<Pix>(holo);	
}



int xmain1() 
{
	cv::Mat lena = cv::imread("C:\\CGH\\Projects\\Propagator3000\\Propagator3000\\input_wframe.bmp", cv::IMREAD_GRAYSCALE);

	cv::Mat_<Pix> holo = cv::Mat_<Pix>(lena.rows, lena.cols, 0.f);

	holo.forEach([&](auto& holo_pix, const int* pos) -> void {
		//float phi = random(-CV_PI, CV_PI);
		float A = lena.at<uchar>(pos[0], pos[1]);
		//holo_pix = Pix(A*std::cosf(phi), A*std::sinf(phi));
		holo_pix = Pix(A, 0);
	});

	//cv::copyMakeBorder(holo, holo, 256, 256, 256, 256, cv::BORDER_CONSTANT, cv::Scalar(0));

	float px, py, l, d;
	px = py = 10e-6f;
	l = 632.8e-9f;
	d = 0.1f;
	
	ShowInt(holo);
	Lens(holo, -2 * d, l, px, py);
	ASD(holo, d, px, py, l);
	ShowInt(holo);
	ShowPhase(holo);
	

	Lens(holo, 2 * d, l, px, py);
	NormAmp(holo);
	ShowPhase(holo);

	ASD(holo, -d, px, py, l);
	ShowInt(holo);
	ShowPhase(holo);

	return 0;
}

int main2()
{
	cv::Mat lena = cv::imread("C:\\input_wframe.bmp", cv::IMREAD_GRAYSCALE);
	//lena = cv::Mat::ones(512, 512, CV_8UC1);
	cv::Mat edges = lena.clone();
	/*lena.forEach<uchar>([&](uchar& pix, const int* pos) {
		int row = pos[0];
		int col = pos[1];

		if (row == 0 || col == 0 || row == (lena.rows - 1) || col == (lena.cols - 1))
		{
			return;
		}
		if (pix == 0)
		{
			return;
		}
		cv::Mat roi = lena(cv::Rect(col - 1, row - 1, 3, 3));
		if (cv::countNonZero(roi) == 9)
		{
			edges.at<uchar>(row, col) = 0;
		}
		else {
			edges.at<uchar>(row, col) = 255;
		}
		
	});*/

	auto edgesX = EdgeX(edges);
	auto edgeY = EdGey(edges);


	auto common =edgesX.mul(edgeY,1.0/255.0);
	cv::subtract(edgeY, common, edgeY);
	
	AddFrame(edgesX);
	AddFrame(edgeY);

	cv::imwrite("shrekY.bmp", edgesX);
	cv::imwrite("shrekX.bmp", edgeY);

	cv::Mat_<Pix> holoX = cv::Mat_<Pix>(edgesX.rows, edgesX.cols, 0.f);
	holoX.forEach([&](auto& holo_pix, const int* pos) -> void {
		//float phi = random(-CV_PI, CV_PI);
		float A = edgesX.at<uchar>(pos[0], pos[1]);
		//holo_pix = Pix(A*std::cosf(phi), A*std::sinf(phi));
		holo_pix = Pix(A, 0);
	});
	//cv::copyMakeBorder(holo, holo, 256, 256, 256, 256, cv::BORDER_CONSTANT, cv::Scalar(0, 0));

	cv::Mat_<Pix> holoY = cv::Mat_<Pix>(edgeY.rows, edgeY.cols, 0.f);
	holoY.forEach([&](auto& holo_pix, const int* pos) -> void {
		//float phi = random(-CV_PI, CV_PI);
		float A = edgeY.at<uchar>(pos[0], pos[1]);
		//holo_pix = Pix(A*std::cosf(phi), A*std::sinf(phi));
		holo_pix = Pix(A, 0);
	});

	float px, py, l, d;
	px = py = 10e-6f;
	l = 632.8e-9f;
	d = 0.1f;
	cv::Rect roi = cv::Rect(holoX.cols / 4, holoX.rows / 4, holoX.cols / 2, holoX.rows / 2);
	cv::imshow("duba", edgeY(roi));
	cv::waitKey(0);
	//ShowInt(holoX(roi));
	//(holoY(roi));
	
	

	// X
	std::cout << "X - pre\n";

	//ShowInt(holoX(roi));
	//ShowPhase(holoX(roi));

	std::cout << "X - post\n";
	ASDX(holoX, d, px, py, l);
	//ShowInt(holoX(roi));
	NormAmp(holoX);
	//ShowInt(holoX(roi));
	//ShowPhase(holoX(roi));

	cv::Mat phase = ShowPhase(holoX(roi));
	//cv::normalize(phase, phase, 0, 255, cv::NORM_MINMAX);
	//phase.convertTo(phase, CV_8UC1);
	//cv::imwrite("1propXpha.bmp", phase);
	cv::Mat inte = ShowInt(holoX(roi));
	//cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
	//phase.convertTo(phase, CV_8UC1);
	//cv::imwrite("1propXint.bmp", inte);
	
	// Y
	std::cout << "Y - pre\n";
	//ShowInt(holoY(roi));
	//ShowPhase(holoY(roi));
	//std::cout << "Y - post\n";
	ASDY(holoY, d, px, py, l);
	//ShowInt(holoY(roi));
	NormAmp(holoY);
	//ShowInt(holoY(roi));
	//ShowPhase(holoY(roi));

	//phase = ShowPhase(holoY(roi));
	//cv::normalize(phase, phase, 0, 255, cv::NORM_MINMAX);
	//phase.convertTo(phase, CV_8UC1);
	//cv::imwrite("1propYpha.bmp", phase);
	//inte = ShowInt(holoY(roi));
	//cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
	//phase.convertTo(phase, CV_8UC1);
	//cv::imwrite("1propYint.bmp", inte);

	//// X + Y
	cv::Mat holo3;
	//holo3= holoX + holoY;
	////cv::multiply(holo, holo2, holo3);
	//std::cout << "X+Y\n";
	//ShowInt(holo3(roi));
	//NormAmp(holo3);
	//ShowInt(holo3(roi));
	//ShowPhase(holo3(roi));

	// Reconstruction
	/*std::cout << "REC\n";

	ASD(holo3, -d, px, py, l);
	ShowInt(holo3(roi));
	ShowPhase(holo3(roi));
	phase = ShowPhase(holo3(roi));
	cv::normalize(phase, phase, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite("2DreconstrPha.bmp", phase);
	inte = ShowInt(holo3(roi));
	cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite("2DreconstrInt.bmp", inte);*/


	ASDX(holoX, -d, px, py, l);
	phase = ShowPhase(holoX(roi));
	cv::normalize(phase, phase, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite("2propXpha.bmp", phase);
	inte = ShowInt(holoX(roi));
	cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite("2propXint.bmp", inte);

	ASDY(holoY, -d, px, py, l);
	phase = ShowPhase(holoY(roi));
	cv::normalize(phase, phase, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite("2propYpha.bmp", phase);
	inte = ShowInt(holoY(roi));
	cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite("2propYint.bmp", inte);
	
	auto xprop = ShowInt(holoX);
	ShowPhase(holoX(roi));

	auto yprop=ShowInt(holoY);
	ShowPhase(holoY(roi));

	float xnonz = (float)cv::countNonZero(xprop)/4194304.0f;
	float ynonz = (float)cv::countNonZero(yprop)/4194304.0f;
	

	std::cout << "xnonz = " << xnonz << "\nynonz= " << ynonz << std::endl;
	
	cv::addWeighted(holoX, xnonz, holoY, ynonz, 1, holo3);
	
	ShowInt(holo3(roi));

	ShowPhase(holo3(roi));

	phase = ShowPhase(holo3(roi));
	cv::normalize(phase, phase, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite("1Dx+1dYPha.bmp", phase);
	inte = ShowInt(holo3);
	cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite("1Dx+1dYInt.bmp", inte);
	inte = ShowInt(holo3(roi));
	cv::normalize(inte, inte, 0, 255, cv::NORM_MINMAX);
	phase.convertTo(phase, CV_8UC1);
	cv::imwrite("1Dx+1dYIntCrop.bmp", inte);
	cv::destroyAllWindows();
	system("pause");
	return 0;
}

