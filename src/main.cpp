#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>

#include <math.h>
#include <cstdlib>

using namespace cv;
using namespace std;

typedef struct {
	float x, y, z;
} Vector3f;

//Vector3f constructVector3f(const Vector3f &V, float _x, float _y, float _z)
//{
//    V.x = _x;
//    V.y = _y;
//    V.z = _z;
//    return V;
//}

Vector3f Normalize(const Vector3f &V) {
	float Len = sqrt(V.x * V.x + V.y * V.y + V.z * V.z);
	if (Len == 0.0f) {
		return V;
	} else {
		float Factor = 1.0f / Len;
		Vector3f result;
		result.x = V.x * Factor;
		result.y = V.y * Factor;
		result.z = V.z * Factor;
		//return Vector3f(V.x * Factor, V.y * Factor, V.z * Factor);
		return result;
	}
}

float Dot(const Vector3f &Left, const Vector3f &Right) {
	return (Left.x * Right.x + Left.y * Right.y + Left.z * Right.z);
}

Vector3f Cross(const Vector3f &Left, const Vector3f &Right) {
	Vector3f Result;
	Result.x = Left.y * Right.z - Left.z * Right.y;
	Result.y = Left.z * Right.x - Left.x * Right.z;
	Result.z = Left.x * Right.y - Left.y * Right.x;
	return Result;
}

Vector3f operator *(const Vector3f &Left, float Right) {
	Vector3f result;
	result.x = Left.x * Right;
	result.y = Left.y * Right;
	result.z = Left.z * Right;
	return result;
}

Vector3f operator *(float Left, const Vector3f &Right) {

	Vector3f result;
	result.x = Right.x * Left;
	result.y = Right.y * Left;
	result.z = Right.z * Left;
	return result;

}

Vector3f operator /(const Vector3f &Left, float Right) {

	Vector3f result;
	result.x = Left.x / Right;
	result.y = Left.y / Right;
	result.z = Left.z / Right;
	return result;

}

Vector3f operator +(const Vector3f &Left, const Vector3f &Right) {

	Vector3f result;
	result.x = Left.x + Right.x;
	result.y = Left.y + Right.y;
	result.z = Left.z + Right.z;
	return result;
}

Vector3f operator -(const Vector3f &Left, const Vector3f &Right) {
	Vector3f result;
	result.x = Left.x - Right.x;
	result.y = Left.y - Right.y;
	result.z = Left.z - Right.z;
	return result;
}

Vector3f operator -(const Vector3f &V) {
	Vector3f result;
	result.x = -V.x;
	result.y = -V.y;
	result.z = -V.z;
	return result;
}

///////////////////////////////////////////////////////////////////////////////////

typedef struct {
	float a, b, c, d;
	Vector3f normal;
} Plane;

Plane ConstructFromPointNormal(const Vector3f &Pt, const Vector3f &Normal) {
	Plane Result;
	Vector3f NormalizedNormal = Normalize(Normal);
	Result.a = NormalizedNormal.x;
	Result.b = NormalizedNormal.y;
	Result.c = NormalizedNormal.z;
	Result.d = -Dot(Pt, NormalizedNormal);
	Result.normal = Normal;
	//Result.normal = NormalizedNormal;
	return Result;
}

Vector3f get3PlaneIntersection(const Plane& plane1, const Plane& plane2,
		const Plane& plane3) {
	Mat C =
			(Mat_<float>(3, 3) << plane1.normal.x, plane1.normal.y, plane1.normal.z, plane2.normal.x, plane2.normal.y, plane2.normal.z, plane3.normal.x, plane3.normal.y, plane3.normal.z);
	float det = determinant(C);
	Vector3f zero;
	zero.x = 0;
	zero.y = 0;
	zero.z = 0;
	if (det == 0) {
		return zero;
	}

	return (Cross(plane2.normal, plane3.normal) * -plane1.d
			+ Cross(plane3.normal, plane1.normal) * -plane2.d
			+ Cross(plane1.normal, plane2.normal) * -plane3.d) / det;

}

int N;
int total_number;
vector<cv::Mat> M; //params
vector<cv::Mat> silhouettes;
//Vec3f ve;

vector<Mat> K;
vector<Mat> Rt;
vector<Mat> R;
vector<Mat> Rvec;
vector<Mat> t;
vector<Mat> cameraPos;

vector<Vector3f> cameraOrigins;
vector<Vector3f> planeNormals;

int main() {

	cv::String path("dinoSR/*.png");
	vector<cv::String> fn;
	vector<cv::Mat> imageData;

	vector<Vec4d> voxels;
	cv::glob(path, fn, true); // recurse

	for (size_t k = 0; k < fn.size(); ++k) {
		cv::Mat im = cv::imread(fn[k]);
		//imshow( "k", im );
		if (im.empty())
			continue; //only proceed if sucsessful

		imageData.push_back(im);

		Vec3b bgcolor = im.at<Vec3b>(Point(1, 1));

		// Change the background from white to black, since that will help later to extract
		// better results during the use of Distance Transform
		for (int x = 0; x < im.rows; x++) {
			for (int y = 0; y < im.cols; y++) {
				if (im.at<Vec3b>(x, y) == bgcolor) {
					im.at<Vec3b>(x, y)[0] = 0;
					im.at<Vec3b>(x, y)[1] = 0;
					im.at<Vec3b>(x, y)[2] = 0;
					//cout<<bgcolor<<endl;
				}
			}
		}

		// without watershed
		//Grayscale matrix
		cv::Mat grayscaleMat(im.size(), CV_8U);

		//Convert BGR to Gray
		cv::cvtColor(im, grayscaleMat, CV_BGR2GRAY);

		//Binary image
		cv::Mat binaryMat(grayscaleMat.size(), grayscaleMat.type());

		//Apply thresholding
		cv::threshold(grayscaleMat, binaryMat, 20, 255, cv::THRESH_BINARY);

		//Show the results
		//cv::namedWindow("silhouettes", cv::WINDOW_AUTOSIZE);
		//cv::imshow("sil", binaryMat);
		//waitKey(0);

		//______________________-------------------->>>

		Mat threshold_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		/// Detect edges using Threshold
		threshold(grayscaleMat, threshold_output, 20, 255, THRESH_BINARY);
		//threshold(binaryMat, threshold_output, 20, 255, THRESH_BINARY);
		/// Find contours
		findContours(threshold_output, contours, hierarchy, CV_RETR_TREE,
				CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		//findContours(binaryMat, contours, hierarchy, CV_RETR_TREE,
		//CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		/// Approximate contours to polygons + get bounding rects and circles
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<Point2f> center(contours.size());
		vector<float> radius(contours.size());
		float maxArea = 0;
		int BBindex;

		for (uint i = 0; i < contours.size(); i++) {
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle((Mat) contours_poly[i], center[i], radius[i]);

			double a = contourArea(contours[i], false);
			if (a > maxArea) {
				maxArea = a;
				BBindex = i;  //Store the index of largest contour
				//bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
			}
		}

		/// Draw polygonal contour + bonding rects + circles

		Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

		//		RNG rng(12345);
		//		for (int i = 0; i < contours.size(); i++) {
		//			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
		//					rng.uniform(0, 255));
		//			drawContours(drawing, contours_poly, i, color, 1, 8,
		//					vector<Vec4i>(), 0, Point());
		//			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2,
		//					8, 0);
		//			circle(drawing, center[i], (int) radius[i], color, 2, 8, 0);
		//		}

		drawContours(drawing, contours, BBindex, Scalar(255), CV_FILLED, 8,
				hierarchy);
		rectangle(drawing, boundRect[BBindex].tl(), boundRect[BBindex].br(),
				Scalar(255, 255, 255), 2, 8, 0);
		Rect test = boundRect[BBindex];
		int x = test.x;
		int y = test.y;
		int width = test.width;
		int height = test.height;
		// Now with those parameters you can calculate the 4 points
		Point top_left(x, y);
		Point top_right(x + width, y);
		Point bottom_left(x, y + height);
		Point bottom_right(x + width, y + height);
		Point mid(x + width / 2, y + height / 2);

		cout << top_left << ", " << top_right << ", " << ", " << bottom_left
				<< ", " << bottom_right << endl;

		cout << mid << endl;

		/// Show in a window
//		namedWindow("Contours", CV_WINDOW_AUTOSIZE);
//		imshow("Contours", drawing);
//		waitKey(0);

		silhouettes.push_back(binaryMat);

	}

	//Read Camera Params from text file ***************************************************

	//vector<cv::Mat> K;
	//vector<cv::Mat> Rt;

	vector<string> fid;

	std::ifstream txtfile("dinoSR/dinoSR_par.txt");
	cout << "Reading text file" << endl;
	//std::ifstream txtfile("templeSR/templeSR_par.txt");
	std::string line;
	vector<string> linedata;
	std::getline(txtfile, line);
	std::stringstream linestream(line);
	int value;
	int i = 0;

	while (linestream >> value) {
		N = value;
	}

	while (std::getline(txtfile, line)) {
		std::stringstream linestream(line);
		string val;
		while (linestream >> val) {
			linedata.push_back(val);
		}
	}
	while (i < linedata.size()) {
		fid.push_back(linedata[i]);
		i++;
		//Put data into K
		Mat kk(3, 3, cv::DataType<float>::type, Scalar(1));
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				float temp = strtof((linedata[i]).c_str(), 0);

				kk.at<float>(j, k) = temp;
				i++;
			}
		}
		K.push_back(kk);

		Mat rot(3, 3, cv::DataType<float>::type, Scalar(1));
		Mat Rttemp(3, 4, cv::DataType<float>::type, Scalar(1));
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				float temp = strtof((linedata[i]).c_str(), 0);

				Rttemp.at<float>(j, k) = temp;
				rot.at<float>(j, k) = temp;
				i++;
			}
		}

		R.push_back(rot);

		int k = 3;
		Mat ttemp(3, 1, cv::DataType<float>::type, Scalar(1));
		for (int j = 0; j < 3; j++) {
			float temp = strtof((linedata[i]).c_str(), 0);
			Rttemp.at<float>(j, k) = temp;

			ttemp.at<float>(j, 0) = temp;
			i++;
		}
		Rt.push_back(Rttemp);
		t.push_back(ttemp);

	}

	// Compute M's
	for (int i = 0; i < N; i++) {

		Mat Mtemp = K[i] * Rt[i];
		M.push_back(Mtemp);
		Mat cameraPosition = -R[i].t() * t[i];
		Mat Rtrans = R[i].t();
		Vector3f cameraOrigin;
		cameraOrigin.x = cameraPosition.at<float>(0, 0);
		cameraOrigin.y = cameraPosition.at<float>(0, 1);
		cameraOrigin.z = cameraPosition.at<float>(0, 2);
		Vector3f planeNormal;
		planeNormal.x = Rtrans.at<float>(0, 2);
		planeNormal.y = Rtrans.at<float>(1, 2);
		planeNormal.z = Rtrans.at<float>(2, 2);

		cameraOrigins.push_back(cameraOrigin);
		planeNormals.push_back(planeNormal);

//		Rodrigues(R[i].t(), RvecTemp);
//		Rvec.push_back(RvecTemp);

		//cameraPos.push_back(cameraPosition);

//		cout << "camera position in world: " << cameraPosition << endl;
//		cout << "camera plane normal in world: " << RvecTemp << endl;

		cout << "camera position in world: " << cameraOrigin.x<<", "<<cameraOrigin.y<<", "<<cameraOrigin.z << endl;
		cout << "camera plane normal in world: " << planeNormal.x<<", "<<planeNormal.y<<", "<<planeNormal.z << endl;

	}

	return 0;
}
