#include <opencv2/opencv.hpp>

int main34() {

	cv::Mat m = cv::Mat::ones(512, 512, CV_8UC1);
	m.forEach<uchar>([&](uchar& px, const int* pos) -> void {
		px = (pos[0] + pos[1]) % 256;
	});
	cv::imshow("m", m);
	cv::waitKey(0);

	system("pause");
	return 0;
}