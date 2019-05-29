#pragma once
#include <opencv2\opencv.hpp>
#include <random>

enum Field
{
	Amplitude,
	Intensity,
	Phase,
	Complex
};

class CGator
{
	static std::mt19937 gen;
	template<typename T>
	static T random(T min, T max) {
		using dist = std::conditional_t<
			std::is_integral<T>::value,
			std::uniform_int_distribution<T>,
			std::uniform_real_distribution<T>
		>;
		return dist{ min, max }(gen);
	}

	// !!! [RE, IM] !!!
	using Pix = std::complex<float>;

	Field m_currField, m_prevField;
	cv::Mat_<Pix> m_data;

	CGator();

public:
	float Lambda;
	float PitchX, PitchY;

	static CGator RandomPhase(int rows, int cols);

public:
	CGator(const cv::String& filename, Field field = Field::Amplitude);
	CGator(const cv::Mat& img, Field field = Field::Amplitude);
	CGator(int rows, int cols);
	CGator(const CGator& other);
	~CGator();

	inline int Rows() const { return m_data.rows; }
	inline int Cols() const { return m_data.cols; }
	inline cv::Size Size() const { return m_data.size(); }

	CGator& Amplitude();
	CGator& Intensity();
	CGator& Phase();
	CGator& Complex();

	CGator& FFTShift();
	CGator& FFT();
	CGator& IFFT();

	CGator& ASD(float distance);

	cv::Mat1f GetAmplitude(bool normalize = true) const;
	cv::Mat1f GetIntensity(bool normalize = true) const;
	cv::Mat1f GetIntensityLog(bool normalize = true) const;
	cv::Mat1f GetPhase(bool normalize = true) const;
	
	CGator& ShowAmplitude(bool destroyWindow = true) const;
	CGator& ShowIntensity(bool destroyWindow = true) const;
	CGator& ShowIntensityLog(bool destroyWindow = true) const;
	CGator& ShowPhase(bool destroyWindow = true) const;

	CGator& Mul(const CGator& other);
	CGator operator*(const CGator& other);

	CGator& operator=(const CGator& other);

private:
	CGator& Show(const cv::Mat1f& img, const cv::String& windowName, bool destroyWindow) const;
	CGator& SetCurrField(Field field);
	Pix GetPixel(int row, int col) const;
};

