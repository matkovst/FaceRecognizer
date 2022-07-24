#pragma once

#include <vector>
#include <utility>
#include <opencv2/core.hpp>

using Matr = std::vector<std::vector<float>>;

Matr matMult(const Matr& a, const Matr& b);

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

std::pair<int, float> searchMostSimilarEmbedding(
    const Matr& embeddings, const std::vector<float>& newComerEmbedding);

std::vector<float> avgEmbedding(const Matr& embeddings);

double getAngleBetweenEyes(const std::vector<int>& landmarks);

/** 
 * @brief Align face using eye points. Rotate, scale and translate face so that the eyes lie on a horizontal line. 
 * Inspired by https://github.com/MasteringOpenCV/code/blob/master/Chapter8_FaceRecognition/preprocessFace.cpp.

    @param cropSize Desired size of the output aligned face
    @param refPoints2 Controls how much of the face is visible after preprocessing
 */
cv::Mat alignFace2(
    const cv::Mat& image, cv::Rect faceBoundingBox, const std::vector<int>& landmarks, cv::Size cropSize, 
    const cv::Point2f refPoints2[2]);

/** 
 * @brief Align face using eye and nose points.

    @param cropSize Desired size of the output aligned face
    @param refPoints3 Reference points for calculating affine Transformation
 */
cv::Mat alignFace3(
    const cv::Mat& image, cv::Rect faceBoundingBox, const std::vector<int>& landmarks, cv::Size cropSize, 
    const cv::Point2f refPoints3[3]);