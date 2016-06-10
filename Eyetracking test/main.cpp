#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <fstream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <array>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

#include <Windows.h>



/** Constants **/


/** Function Headers */
std::array<EstimatedEyeInfo, 2> detectAndDisplay(cv::Mat frame);
void initKalmanFilter(cv::KalmanFilter &KF);

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "D:/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
cv::String eye_cascade_name = "D:/opencv/build/etc/haarcascades/haarcascade_eye.xml";
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eye_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

std::ofstream outFile;
cv::KalmanFilter KF_left(4, 2, 0);
cv::KalmanFilter KF_right(4, 2, 0);
cv::Mat_<float> measurement(2, 1);
cv::Mat prediction;
cv::Mat estimated;
FILE * pFile;
int counter;


/**
 * @function main
 */
int main(int argc, const char** argv) {

	cv::Mat frame;
	std::array<EstimatedEyeInfo, 2> eyes;
	EstimatedEyeInfo estLeftEyeInfo, estRightEyeInfo;

	initKalmanFilter(KF_left);
	initKalmanFilter(KF_right);

	//outFile.open("output.txt", std::ios::trunc);
	pFile = fopen("myfile.txt", "w");


	// Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };
	if (!eye_cascade.load(eye_cascade_name)) { printf("--(!)Error loading eye cascade, please change face_cascade_name in source code.\n"); return -1; };

	cv::namedWindow(main_window_name, CV_WINDOW_NORMAL);
	cv::moveWindow(main_window_name, 400, 100);
	cv::namedWindow(face_window_name, CV_WINDOW_NORMAL);
	cv::moveWindow(face_window_name, 10, 100);
	cv::namedWindow("Right Eye", CV_WINDOW_NORMAL);
	cv::moveWindow("Right Eye", 10, 600);
	cv::namedWindow("Left Eye", CV_WINDOW_NORMAL);
	cv::moveWindow("Left Eye", 10, 800);
	cv::namedWindow("aa", CV_WINDOW_NORMAL);
	cv::moveWindow("aa", 10, 800);
	cv::namedWindow("aaa", CV_WINDOW_NORMAL);
	cv::moveWindow("aaa", 10, 800);

	createCornerKernels();
	ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
		43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

	// I make an attempt at supporting both 2.x and 3.x OpenCV

	cv::VideoCapture capture(0);
	if (capture.isOpened()) {
		while (true) {
			capture.read(frame);
			// mirror it
			cv::flip(frame, frame, 1);
			frame.copyTo(debugImage);

			// Apply the classifier to the frame
			if (!frame.empty()) {
				eyes = detectAndDisplay(frame);
				estLeftEyeInfo = eyes[0];
				estRightEyeInfo = eyes[1];
				if (estRightEyeInfo.closedEye)
					counter++;
				else
					counter = 0;

				if (counter == 10) {
					float x = estLeftEyeInfo.coordinates.x;
					float y = estLeftEyeInfo.coordinates.y;
					if (x > 35)
						if (y < 30)
							std::cout << "Prawy górny" << std::endl;
						else
							std::cout << "Prawy dolny" << std::endl;
					else
						if (y < 30)
							std::cout << "Lewy górny" << std::endl;
						else
							std::cout << "Lewy dolny" << std::endl;
				}

			}
			else {
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			imshow(main_window_name, debugImage);

			int c = cv::waitKey(10);
			if ((char)c == 'c') { 
				//outFile.close();
				fclose(pFile);
				break; 
			}
			if ((char)c == 'f') {
				imwrite("frame.png", frame);
			}

		}
	}

	releaseCornerKernels();

	return 0;
}

std::array<EstimatedEyeInfo, 2> findEyes(cv::Mat frame_gray, cv::Rect face) {
	cv::Mat faceROI = frame_gray(face);
	cv::Mat debugFace = faceROI;
	std::array<EstimatedEyeInfo, 2> eyes;

	if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * face.width;
		GaussianBlur(faceROI, faceROI, cv::Size(0, 0), sigma);
	}


		//std::cout << eyes[i] << std::endl;
	//std::cout << std::endl;*/


	//-- Find eye regions and draw them
	int eye_region_width = face.width * (kEyePercentWidth / 100.0);
	int eye_region_height = face.width * (kEyePercentHeight / 100.0);
	int eye_region_top = face.height * (kEyePercentTop / 100.0);
	int eye_region_left = face.width*(kEyePercentSide / 100.0);
	cv::Rect leftEyeRegion(eye_region_left, eye_region_top, eye_region_width, eye_region_height);
	cv::Rect rightEyeRegion(face.width - eye_region_width - eye_region_left,
		eye_region_top, eye_region_width, eye_region_height);

	//-- Find Eye Centers
	EyeInfo leftPupilInfo = findEyeCenter(faceROI, leftEyeRegion, "Left Eye", eye_cascade);
	EyeInfo rightPupilInfo = findEyeCenter(faceROI, rightEyeRegion, "Right Eye", eye_cascade);

	cv::Point leftPupil = leftPupilInfo.coordinates;
	cv::Point rightPupil = rightPupilInfo.coordinates;

	bool rightEyeOpened = rightPupilInfo.closedEye;

	prediction = KF_left.predict();
	measurement(0) = (float)leftPupil.x;
	measurement(1) = (float)leftPupil.y;
	estimated = KF_left.correct(measurement);

	cv::Point_<float> estLeftEye(estimated.at<float>(0), estimated.at<float>(1));

	fprintf(pFile, "%f\t%f\t%f\t%f\t%f\t%f\t", prediction.at<float>(0), prediction.at<float>(1),
		measurement(0), measurement(1), estimated.at<float>(0), estimated.at<float>(1));

	//printf("%f\t", estimated.at<float>(1));

	prediction = KF_right.predict();
	measurement(0) = (float)rightPupil.x;
	measurement(1) = (float)rightPupil.y;
	estimated = KF_right.correct(measurement);

	cv::Point_<float> estRightEye(estimated.at<float>(0), estimated.at<float>(1));
	eyes[0] = EstimatedEyeInfo() = { estLeftEye, leftPupilInfo.closedEye };
	eyes[1] = EstimatedEyeInfo() = { estRightEye, rightPupilInfo.closedEye };

	fprintf(pFile, "%f\t%f\t%f\t%f\t%f\t%f\n", prediction.at<float>(0), prediction.at<float>(1),
		measurement(0), measurement(1), estimated.at<float>(0), estimated.at<float>(1));

	

	

	//outFile << leftPupil << std::endl;
	//printf("%f\n", estimated.at<float>(1));

	//printf("%f\t%f\n", estimated.at<float>(0), estimated.at<float>(1));

	// get corner regions
	cv::Rect leftRightCornerRegion(leftEyeRegion);
	leftRightCornerRegion.width -= leftPupil.x;
	leftRightCornerRegion.x += leftPupil.x;
	leftRightCornerRegion.height /= 2;
	leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
	cv::Rect leftLeftCornerRegion(leftEyeRegion);
	leftLeftCornerRegion.width = leftPupil.x;
	leftLeftCornerRegion.height /= 2;
	leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
	cv::Rect rightLeftCornerRegion(rightEyeRegion);
	rightLeftCornerRegion.width = rightPupil.x;
	rightLeftCornerRegion.height /= 2;
	rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
	cv::Rect rightRightCornerRegion(rightEyeRegion);
	rightRightCornerRegion.width -= rightPupil.x;
	rightRightCornerRegion.x += rightPupil.x;
	rightRightCornerRegion.height /= 2;
	rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
	rectangle(debugFace, leftRightCornerRegion, 200);
	rectangle(debugFace, leftLeftCornerRegion, 200);
	rectangle(debugFace, rightLeftCornerRegion, 200);
	rectangle(debugFace, rightRightCornerRegion, 200);
	// change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;
	// draw eye centers
	circle(debugFace, rightPupil, 3, 1234);
	circle(debugFace, leftPupil, 3, 1234);

	//-- Find Eye Corners
	if (kEnableEyeCorner) {
		cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
		leftRightCorner.x += leftRightCornerRegion.x;
		leftRightCorner.y += leftRightCornerRegion.y;
		cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
		leftLeftCorner.x += leftLeftCornerRegion.x;
		leftLeftCorner.y += leftLeftCornerRegion.y;
		cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
		rightLeftCorner.x += rightLeftCornerRegion.x;
		rightLeftCorner.y += rightLeftCornerRegion.y;
		cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
		rightRightCorner.x += rightRightCornerRegion.x;
		rightRightCorner.y += rightRightCornerRegion.y;
		circle(faceROI, leftRightCorner, 3, 200);
		circle(faceROI, leftLeftCorner, 3, 200);
		circle(faceROI, rightLeftCorner, 3, 200);
		circle(faceROI, rightRightCorner, 3, 200);
	}

	//imshow(face_window_name, faceROI);
	//  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
	//  cv::Mat destinationROI = debugImage( roi );
	//  faceROI.copyTo( destinationROI );
	return eyes;
}


cv::Mat findSkin(cv::Mat &frame) {
	cv::Mat input;
	cv::Mat output = cv::Mat(frame.rows, frame.cols, CV_8U);

	cvtColor(frame, input, CV_BGR2YCrCb);

	for (int y = 0; y < input.rows; ++y) {
		const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
		//    uchar *Or = output.ptr<uchar>(y);
		cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
		for (int x = 0; x < input.cols; ++x) {
			cv::Vec3b ycrcb = Mr[x];
			//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
			if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
				Or[x] = cv::Vec3b(0, 0, 0);
			}
		}
	}
	return output;
}

/**
 * @function detectAndDisplay
 */
std::array<EstimatedEyeInfo, 2> detectAndDisplay(cv::Mat frame) {
	std::vector<cv::Rect> faces;
	//cv::Mat frame_gray;

	std::vector<cv::Mat> rgbChannels(3);
	cv::split(frame, rgbChannels);
	cv::Mat frame_gray = rgbChannels[2];

	//cvtColor( frame, frame_gray, CV_BGR2GRAY );
	//equalizeHist( frame_gray, frame_gray );
	//cv::pow(frame_gray, CV_64F, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150));
	//  findSkin(debugImage);

	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(debugImage, faces[i], 1234);
	}

	//-- Show what you got
	if (faces.size() > 0) {
		return findEyes(frame_gray, faces[0]);
	}
	return std::array<EstimatedEyeInfo, 2>();
}

void initKalmanFilter(cv::KalmanFilter &KF) {
	KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
	measurement.setTo(cv::Scalar(0));
	KF.statePre.at<float>(0) = 38.0f;
	KF.statePre.at<float>(1) = 32.0f;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, cv::Scalar::all(3));
	setIdentity(KF.errorCovPost, cv::Scalar::all(9));
}

void callibrate() {

}

