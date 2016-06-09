#ifndef EYE_CENTER_H
#define EYE_CENTER_H

#include "opencv2/imgproc/imgproc.hpp"

struct EyeInfo {
	cv::Point coordinates;
	bool openedEye;
};

EyeInfo findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow, cv::CascadeClassifier &eye_cascade);

#endif