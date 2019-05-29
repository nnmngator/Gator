#include "CGator.h"

std::mt19937 CGator::gen = std::mt19937(std::random_device{}());

CGator::CGator() :
	Lambda(632.8e-9f),
	PitchX(10e-6f), PitchY(10e-6f),
	m_currField(Field::Amplitude), m_prevField(Field::Amplitude),
	m_data(cv::Mat())
{}

CGator CGator::RandomPhase(int rows, int cols)
{
	CGator result(rows, cols);

	result.m_data.forEach([&](Pix& pix, const int*) -> void {
		float phi = random<float>(-CV_PI, CV_PI);
		float re = cosf(phi);
		float im = sinf(phi);
		pix = Pix(re, im);
	});
	return result;
}

CGator::CGator(const cv::String & filename, Field field) : CGator()
{
	cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	if (!img.data)
	{
		throw std::invalid_argument("Invalid file");
	}

	m_currField = field;

	m_data = cv::Mat_<Pix>(img.size());
	m_data.forEach([&](Pix& pix, const int * pos) -> void {
		const int row = pos[0], col = pos[1];

		float re = 0.f, im = 0.f, A = 0.f, phi = 0.f;
		float value = img.at<uchar>(row, col) / 255.f;

		switch (m_currField) {
		case Field::Amplitude:
			A = value;
			phi = 1.f;
			break;
		case Field::Intensity:
			A = sqrtf(value);
			phi = 1.f;
			break;
		case Field::Phase:
			A = 1.f;
			phi = value * 2.0f * CV_PI - CV_PI; // [-PI, PI]
			break;
		}

		re = A * cosf(phi);
		im = A * sinf(phi);

		pix = Pix(re, im);
	});
}

CGator::CGator(const cv::Mat & img, Field field) : CGator()
{
	if (!img.data)
	{
		throw std::invalid_argument("Invalid image");
	}

	if (img.channels() == 2 && field != Field::Complex)
	{
		throw std::invalid_argument("Invalid field");
	}

	m_currField = field;

	if (img.type() == CV_32FC2)
	{
		m_data = img.clone();
	}
	else if (img.type() == CV_64FC2)
	{
		img.convertTo(m_data, CV_32FC2);
	}
	// 2 channel but not floating -> scale and then assign
	else if (img.channels() == 2)
	{
		img.convertTo(m_data, CV_32FC2);
		std::vector<cv::Mat> imgs(2);
		cv::split(m_data, imgs);
		cv::normalize(imgs[0], imgs[0], 0., 1., cv::NORM_MINMAX);
		cv::normalize(imgs[1], imgs[1], 0., 1., cv::NORM_MINMAX);
		cv::merge(imgs, m_data);
	}

	// Data was assigned so we can return here already
	if (m_data.data) return;

	cv::Mat tmpImg;
	img.convertTo(tmpImg, CV_32FC1);
	cv::normalize(tmpImg, tmpImg, 0, 1, cv::NORM_MINMAX);

	m_data = cv::Mat_<Pix>(img.size());
	m_data.forEach([&](Pix& pix, const int * pos) -> void {
		const int row = pos[0], col = pos[1];

		float re = 0.f, im = 0.f, A = 0.f, phi = 0.f;
		float value = img.at<float>(row, col);

		switch (m_currField) {
		case Field::Amplitude:
			A = value;
			phi = 0.f;
			break;
		case Field::Intensity:
			A = sqrtf(value);
			phi = 0.f;
			break;
		case Field::Phase:
			A = 1.f;
			phi = value * 2.0f * CV_PI - CV_PI; // [-PI, PI]
			break;
		}

		re = A * cosf(phi);
		im = A * sinf(phi);

		pix = Pix(re, im);
	});
}

CGator::CGator(int rows, int cols) : CGator()
{
	m_data = cv::Mat_<Pix>(rows, cols, Pix(1, 1));
}

CGator::CGator(const CGator & other) :
	Lambda(other.Lambda),
	PitchX(other.PitchX), PitchY(other.PitchY),
	m_currField(other.m_currField), m_prevField(other.m_prevField),
	m_data(other.m_data.clone())
{}

CGator::~CGator()
{
	m_data.release();
}

CGator & CGator::Amplitude()
{
	return SetCurrField(Field::Amplitude);
}

CGator & CGator::Intensity()
{
	return SetCurrField(Field::Intensity);
}

CGator & CGator::Phase()
{
	return SetCurrField(Field::Phase);
}

CGator & CGator::Complex()
{
	return SetCurrField(Field::Complex);
}


CGator & CGator::FFTShift()
{
	m_data.forEach([&](Pix& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];

		if (row >= m_data.rows / 2)
		{
			return;
		}

		int next_row = row + m_data.rows / 2;
		int next_col =
			(col < (m_data.cols / 2)) ?
			(col + (m_data.cols / 2)) :
			(col - (m_data.cols / 2));;

		Pix& next_pix = m_data.at<Pix>(next_row, next_col);
		std::swap(next_pix, pix);
	});
	return *this;
}

CGator & CGator::FFT()
{
	cv::dft(m_data, m_data, cv::DFT_COMPLEX_OUTPUT);
	return *this;
}

CGator & CGator::IFFT()
{
	cv::idft(m_data, m_data, cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
	return *this;
}


CGator & CGator::ASD(float distance)
{
	FFTShift().FFT().FFTShift();

	int N = m_data.rows;
	int M = m_data.cols;

	m_data.forEach([&](Pix& pix, const int* pos) -> void {
		int n = pos[0] - N / 2,
			m = pos[1] - M / 2;

		float f =
			std::pow(n / (N * PitchX), 2.f) +
			std::pow(m / (M * PitchY), 2.f);

		pix *= std::exp(Pix(0, -CV_PI * Lambda*distance*f));
	});

	FFTShift().IFFT().FFTShift();

	return *this;
}


cv::Mat1f CGator::GetAmplitude(bool normalize) const
{
	cv::Mat1f result(m_data.size());
	m_data.forEach([&](Pix& pix, const int * pos) -> void {
		const int row = pos[0], col = pos[1];
		result.at<float>(row, col) = std::abs(pix);
	});

	if (normalize)
	{
		cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	}
	return result;
}

cv::Mat1f CGator::GetIntensity(bool normalize) const
{
	cv::Mat1f result(m_data.size());
	m_data.forEach([&](Pix& pix, const int * pos) -> void {
		const int row = pos[0], col = pos[1];
		result.at<float>(row, col) = std::powf(std::abs(pix), 2.f);
	});

	if (normalize)
	{
		cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	}
	return result;
}

cv::Mat1f CGator::GetIntensityLog(bool normalize) const
{
	cv::Mat1f result(m_data.size());
	m_data.forEach([&](Pix& pix, const int * pos) -> void {
		const int row = pos[0], col = pos[1];
		result.at<float>(row, col) = std::logf(1.f + std::powf(std::abs(pix), 2.f));
	});

	if (normalize)
	{
		cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	}
	return result;
}

cv::Mat1f CGator::GetPhase(bool normalize) const
{
	cv::Mat1f result(m_data.size());
	m_data.forEach([&](Pix& pix, const int * pos) -> void {
		const int row = pos[0], col = pos[1];
		result.at<float>(row, col) = std::arg(pix);
	});

	if (normalize)
	{
		cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	}
	return result;
}


CGator& CGator::ShowAmplitude(bool destroyWindow) const
{
	return Show(GetAmplitude(), "amplitude", destroyWindow);
}
CGator& CGator::ShowIntensity(bool destroyWindow) const
{
	return Show(GetIntensity(), "intensity", destroyWindow);
}
CGator& CGator::ShowIntensityLog(bool destroyWindow) const
{
	return Show(GetIntensityLog(), "intensity log", destroyWindow);
}
CGator& CGator::ShowPhase(bool destroyWindow) const
{
	return Show(GetPhase(), "phase", destroyWindow);
}


CGator & CGator::Mul(const CGator & other)
{
	if (m_data.size() != other.m_data.size())
	{
		throw std::invalid_argument("Different sizes");
	}

	m_data.forEach([&](Pix& pix, const int * pos) -> void {
		const int row = pos[0], col = pos[1];

		// In form of [A, phi]
		Pix p1 = GetPixel(row, col);
		Pix p2 = other.GetPixel(row, col);

		pix = p1 * p2;
	});

	return *this;
}

CGator CGator::operator*(const CGator & other)
{
	return CGator(*this).Mul(other);
}

CGator & CGator::operator=(const CGator & other)
{
	CGator::~CGator();
	Lambda = other.Lambda;
	PitchX = other.PitchX;
	PitchY = other.PitchY;
	m_currField = other.m_currField;
	m_prevField = other.m_prevField;
	m_data = other.m_data.clone();
	return *this;
}

CGator & CGator::Show(const cv::Mat1f & img, const cv::String & windowName, bool destroyWindow) const
{
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::resizeWindow(windowName, 512, 512);
	cv::imshow(windowName, img);
	cv::waitKey();

	if (destroyWindow)
	{
		cv::destroyWindow(windowName);
	}
	// what the fuck, why
	return const_cast<CGator&>(*this);
}

CGator & CGator::SetCurrField(Field field)
{
	m_prevField = m_currField;
	m_currField = field;
	return *this;
}

CGator::Pix CGator::GetPixel(int row, int col) const
{
	Pix pix = m_data.at<Pix>(row, col);

	float A = std::abs(pix);
	float phi = std::arg(pix);

	switch (m_currField) {
	case Field::Complex:
		return std::polar(A, phi);
	case Field::Amplitude:
		return std::polar(A, 0.f);
	case Field::Intensity:
		return std::polar(A*A, 0.f);
	case Field::Phase:
		return Pix(1.f, phi);
	}
}
