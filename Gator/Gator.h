#pragma once
#include <opencv2/opencv.hpp>



typedef std::complex<float> Pix;


cv::Mat ShowLogInt(cv::Mat m);

cv::Mat ShowInt(cv::Mat m);

cv::Mat ShowPhase(cv::Mat m);

cv::Mat_<Pix> AddZeroPhase(cv::Mat intensity);

cv::Mat_<Pix> ImportAsPhase(cv::Mat phase);

void SaveResults(cv::Mat_<Pix> holo, std::string filename, bool crop=0);

cv::Mat Edge(cv::Mat m);

cv::Mat EdgeX(cv::Mat m);

cv::Mat EdGey(cv::Mat m);

void AddFrame(cv::Mat m);

void ConvertToEdge(cv::Mat src, cv::Mat_<Pix>& holoX, cv::Mat_<Pix>& holoY, cv::Mat & cmn);

void FFTX(cv::Mat m);

void IFFTX(cv::Mat m);

void FFT(cv::Mat m);

void IFFT(cv::Mat m);

void Lens(cv::Mat m, float f, float l, float px, float py);

void NormAmp(cv::Mat m);

void ASDX(cv::Mat holo, float d, float px, float py, float l);

void ASDY(cv::Mat holo, float d, float px, float py, float l);

void ASD(cv::Mat holo, float d, float px, float py, float l);

cv::Mat_<Pix> AddAmplitudes(cv::Mat m1, cv::Mat m2);

void LoopOfDeath(cv::Mat holoXSRC, cv::Mat holoYSRC, cv::Mat SRC2d, float minpx, float minpy, float l, float mind, float maxP, float maxD, float iP, float iD, cv::Mat ref, bool FileWrite=0);