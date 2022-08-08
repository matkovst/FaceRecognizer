#pragma once

#include <vector>
#include <opencv2/highgui.hpp>
#include "face.h"

void renderBoundingBoxes(cv::Mat& out, std::vector<cv::Rect> boundingBoxes);

void renderLandmarks(cv::Mat& out, std::vector<std::vector<int>> landmarks);

void renderFaces(cv::Mat& out, std::vector<Face> faces);

cv::Mat renderKfFrame(cv::Size size, cv::Rect real, cv::Rect pred);