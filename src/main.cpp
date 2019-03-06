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
	//Result.d = -Dot(Pt, NormalizedNormal);
	Result.d = Dot(Pt, NormalizedNormal);
	//Result.normal = Normal;
	Result.normal = NormalizedNormal;
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
		cout << "determinant zero, same or parallel planes provided!" << endl;
		return zero;
	}

	//cout<<"plane 1 d is : "<<plane1.d<<endl;
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
vector<Plane> cameraPlanes;
vector<Point> midpoints;

////for 2d to 3d conversion
typedef struct {
	vector<Point2f> pnts2d;
} campnts;

vector<campnts> pnts;
vector<Mat> points3D;

int main() {

	cv::String path("dinoSR/*.png");
	//cv::String path("birdR2/*.pgm");
	vector<cv::String> fn;
	vector<cv::Mat> imageData;

	vector<Vec4d> voxels;
	cv::glob(path, fn, true); // recurse

	for (size_t k = 0; k < fn.size(); ++k) {
		cv::Mat im = cv::imread(fn[k]);
		//imshow( "k", im );
		if (im.empty())
			continue; //only proceed if sucsessful

		//im = ~im; //for bird data

		imageData.push_back(im);
//		cv::namedWindow("images", cv::WINDOW_AUTOSIZE);
//		cv::imshow("image", im);
//		waitKey(0);

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
//		cv::namedWindow("silhouettes", cv::WINDOW_AUTOSIZE);
//		cv::imshow("sil", binaryMat);
//		waitKey(0);

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
		Point2f top_left(x, y);
		Point2f top_right(x + width, y);
		Point2f bottom_left(x, y + height);
		Point2f bottom_right(x + width, y + height);
		Point2f mid(x + width / 2, y + height / 2);
		midpoints.push_back(mid);

		campnts camerapnts;
		camerapnts.pnts2d.push_back(top_left);
		camerapnts.pnts2d.push_back(top_right);
		camerapnts.pnts2d.push_back(bottom_left);
		camerapnts.pnts2d.push_back(bottom_right);
		//camerapnts.pnts2d.push_back(mid);
		pnts.push_back(camerapnts);

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
	//std::ifstream txtfile("birdR2/birdR_par.txt");
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

		//bird Data M's (projection Matrix)
//		Mat P(3, 4, cv::DataType<float>::type, Scalar(1));
//		for (int j = 0; j < 3; j++) {
//			for (int k = 0; k < 4; k++) {
//				float temp = strtof((linedata[i]).c_str(), 0);
//
//				P.at<float>(j, k) = temp;
//				i++;
//			}
//		}
//		M.push_back(P);

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
		//planeNormal.y = 0.1;
		planeNormal.z = Rtrans.at<float>(2, 2);

		Plane cameraPlane = ConstructFromPointNormal(cameraOrigin, planeNormal);

		cameraOrigins.push_back(cameraOrigin);
		planeNormals.push_back(planeNormal);
		cameraPlanes.push_back(cameraPlane);
		float test = Dot(cameraOrigins[i], planeNormals[i]);
		//cout<<"test d is : "<<test<<endl;
//		Rodrigues(R[i].t(), RvecTemp);
//		Rvec.push_back(RvecTemp);

		//cameraPos.push_back(cameraPosition);

//		cout << "camera position in world: " << cameraPosition << endl;
//		cout << "camera plane normal in world: " << RvecTemp << endl;

		cout << "camera position in world: " << cameraOrigin.x << ", "
				<< cameraOrigin.y << ", " << cameraOrigin.z << endl;
		cout << "camera plane normal in world: " << planeNormal.x << ", "
				<< planeNormal.y << ", " << planeNormal.z << endl;

	}

	//for bird

	FILE *fptrx;
	FILE *fptry;
	FILE *fptrz;
	FILE *fptru;
	FILE *fptrv;
	FILE *fptrw;

	if ((fptrx = fopen("x.txt", "w")) == NULL) {
		fprintf(stderr, "Failed to open .txt file!\n");
		exit(-1);
	}

	if ((fptry = fopen("y.txt", "w")) == NULL) {
		fprintf(stderr, "Failed to open .txt file!\n");
		exit(-1);
	}

	if ((fptrz = fopen("z.txt", "w")) == NULL) {
		fprintf(stderr, "Failed to open .txt file!\n");
		exit(-1);
	}

	if ((fptru = fopen("u.txt", "w")) == NULL) {
		fprintf(stderr, "Failed to open .txt file!\n");
		exit(-1);
	}

	if ((fptrv = fopen("v.txt", "w")) == NULL) {
		fprintf(stderr, "Failed to open .txt file!\n");
		exit(-1);
	}

	if ((fptrw = fopen("w.txt", "w")) == NULL) {
		fprintf(stderr, "Failed to open .txt file!\n");
		exit(-1);
	}

	for (int i = 0; i < N; i++) {
		Mat rotm, tvec, kk;
		decomposeProjectionMatrix(M[i], kk, rotm, tvec);
		K.push_back(kk);
//		cout << kk << endl << endl;
		R.push_back(rotm);
		Mat ttemp(3, 1, cv::DataType<float>::type, Scalar(1));
		float temp4 = tvec.at<float>(3, 0);
		float temp1 = ttemp.at<float>(0, 0) = tvec.at<float>(0, 0) / temp4;
		float temp2 = ttemp.at<float>(1, 0) = tvec.at<float>(1, 0) / temp4;
		float temp3 = ttemp.at<float>(2, 0) = tvec.at<float>(2, 0) / temp4;

		t.push_back(ttemp);

		//Mat cameraPosition = -R[i].t() * t[i];
		Mat Rtrans = R[i].t();
		Mat cameraPosition = ttemp;
		//Mat Rtrans = rotm;
		Vector3f cameraOrigin;
		cameraOrigin.x = cameraPosition.at<float>(0, 0);
		if ((fptrx = fopen("x.txt", "a")) == NULL) {
			fprintf(stderr, "Failed to open .txt file!\n");
			exit(-1);
		}
		fprintf(fptrx, "%f ", cameraOrigin.x);

		cameraOrigin.y = cameraPosition.at<float>(0, 1);
		if ((fptry = fopen("y.txt", "a")) == NULL) {
			fprintf(stderr, "Failed to open .txt file!\n");
			exit(-1);
		}
		fprintf(fptry, "%f ", cameraOrigin.y);

		cameraOrigin.z = cameraPosition.at<float>(0, 2);
		if ((fptrz = fopen("z.txt", "a")) == NULL) {
			fprintf(stderr, "Failed to open .txt file!\n");
			exit(-1);
		}
		fprintf(fptrz, "%f ", cameraOrigin.z);

		Vector3f planeNormal;
		planeNormal.x = Rtrans.at<float>(0, 2);
		if ((fptru = fopen("u.txt", "a")) == NULL) {
			fprintf(stderr, "Failed to open .txt file!\n");
			exit(-1);
		}
		fprintf(fptru, "%f ", planeNormal.x);

		planeNormal.y = Rtrans.at<float>(1, 2);
		if ((fptrv = fopen("v.txt", "a")) == NULL) {
			fprintf(stderr, "Failed to open .txt file!\n");
			exit(-1);
		}
		fprintf(fptrv, "%f ", planeNormal.y);

		planeNormal.z = Rtrans.at<float>(2, 2);
		if ((fptrw = fopen("w.txt", "a")) == NULL) {
			fprintf(stderr, "Failed to open .txt file!\n");
			exit(-1);
		}
		fprintf(fptrw, "%f ", planeNormal.z);

		Plane cameraPlane = ConstructFromPointNormal(cameraOrigin, planeNormal);

		cameraOrigins.push_back(cameraOrigin);
		planeNormals.push_back(planeNormal);
		cameraPlanes.push_back(cameraPlane);

//		cout << "camera position in world: " << cameraOrigin.x << ", "
//				<< cameraOrigin.y << ", " << cameraOrigin.z << endl;
//		cout << "camera plane normal in world: " << planeNormal.x << ", "
//				<< planeNormal.y << ", " << planeNormal.z << endl;

	}
	fclose(fptrx);
	fclose(fptry);
	fclose(fptrz);
	fclose(fptru);
	fclose(fptrv);
	fclose(fptrw);

//	Vector3f intersection012 = get3PlaneIntersection(cameraPlanes[0],
//			cameraPlanes[1], cameraPlanes[2]);
//	cout << intersection012.x << ", " << intersection012.y << ", "
//			<< intersection012.z << endl;

	float xmin = 100;
	float xmax = -100;
	float ymin = 100;
	float ymax = -100;
	float zmin = 100;
	float zmax = -100;

//	for (int i = 0; i < 1; i++) {
//		for (int j = 1; j < 7; j++) {
//			for (int k = 7; k < N; k++) {
//				Vector3f intersectiontemp = get3PlaneIntersection(
//						cameraPlanes[i], cameraPlanes[j], cameraPlanes[k]);
//				if (intersectiontemp.x < xmin)
//					xmin = intersectiontemp.x;
//				if (intersectiontemp.x > xmax)
//					xmax = intersectiontemp.x;
//				if (intersectiontemp.y < ymin)
//					ymin = intersectiontemp.y;
//				if (intersectiontemp.y > ymax)
//					ymax = intersectiontemp.y;
//				if (intersectiontemp.z < zmin)
//					zmin = intersectiontemp.z;
//				if (intersectiontemp.z > zmax)
//					zmax = intersectiontemp.z;
////				cout << "for planes " << i << ", " << j << ", " << k
////						<< " intersection is: [ " << intersectiontemp.x << ", "
////						<< intersectiontemp.y << ", " << intersectiontemp.z
////						<< " ]" << endl;
//				j++;
//
//			}
//		}
//	}

//	cout << "min is: [ " << xmin << ", " << ymin << ", " << zmin << " ]"
//			<< endl;
//	cout << "max is: [ " << xmax << ", " << ymax << ", " << zmax << " ]"
//			<< endl;

//	std::vector<cv::Point2d> cam0pnts;
//
//	std::vector<cv::Point2d> cam1pnts;
//
//	cam0pnts.push_back(midpoints[1]);
//	cam1pnts.push_back(midpoints[2]);
//
//	cv::Mat pnts3D(4, cam0pnts.size(), CV_32F);
//
//	triangulatePoints(M[1], M[2], cam0pnts, cam1pnts, pnts3D);
//
//	pnts3D.at<double>(0, 0) = pnts3D.at<double>(0, 0) / pnts3D.at<double>(3, 0);
//	pnts3D.at<double>(1, 0) = pnts3D.at<double>(1, 0) / pnts3D.at<double>(3, 0);
//	pnts3D.at<double>(2, 0) = pnts3D.at<double>(2, 0) / pnts3D.at<double>(3, 0);
//	pnts3D.at<double>(3, 0) = pnts3D.at<double>(3, 0) / pnts3D.at<double>(3, 0);

	for (int a = 0; a < 3; a++) {
		Mat temp(4, pnts[0].pnts2d.size(), CV_32F);
		triangulatePoints(M[0], M[a+1], pnts[0].pnts2d, pnts[a+1].pnts2d, temp);
		//triangulatePoints(M[a], M[a+1], pnts[a].pnts2d, pnts[a+1].pnts2d, temp);
		//temp = temp.t();
		for (int k = 0; k < temp.cols; k++) {
			for (int j = 0; j < 4; j++) {
				temp.at<float>(j, k) = temp.at<float>(j, k)
						/ temp.at<float>(3, k);
				if (j == 0) {
					if (temp.at<float>(j, k) < xmin)
						xmin = temp.at<float>(j, k);
					if (temp.at<float>(j, k) > xmax)
						xmax = temp.at<float>(j, k);
				} else if (j == 1) {
					if (temp.at<float>(j, k) < ymin)
						ymin = temp.at<float>(j, k);
					if (temp.at<float>(j, k) > ymax)
						ymax = temp.at<float>(j, k);
				} else if (j == 2) {
					if (temp.at<float>(j, k) < zmin)
						zmin = temp.at<float>(j, k);
					if (temp.at<float>(j, k) > zmax)
						zmax = temp.at<float>(j, k);
				}
			}
		}
		points3D.push_back(temp);
		cout << temp << endl;
	}

	cout << "min is: [ " << xmin << ", " << ymin << ", " << zmin << " ]"
			<< endl;
	cout << "max is: [ " << xmax << ", " << ymax << ", " << zmax << " ]"
			<< endl;
	cout<<points3D[0].col(0)<<endl;

//	cout << pnts3D << endl;
//	cout << pnts3D.size() << endl;
//	cout << pnts3D.at<double>(1, 0) << endl;
	//cout<< pnts[1].pnts2d<<endl;

	return 0;
}
