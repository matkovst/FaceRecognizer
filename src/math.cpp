#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "math.h"

Matr matMult(const Matr& a, const Matr& b)
{
    if (a.empty() || b.empty())
        throw std::runtime_error("matMult: Empty vector");

    const int aRows = a.size();
    const int aCols = a[0].size();
    const int bRows = b.size();
    const int bCols = b[0].size();
    if (aCols != bRows)
        throw std::runtime_error("matMult: vector 1 column must be equal to vector 2 row");

    Matr result(aRows, std::vector<float>(bCols));
    for(int i = 0; i < aRows; ++i)
        for(int j = 0; j < bCols; ++j)
            for(int k = 0; k < aCols; ++k)
                result[i][j] += a[i][k] * b[k][j];
    return result;
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("cosineSimilarity: vector dimentions must be equal");

    double dot = 0.0f;
    double denomA = 0.0f;
    double denomB = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        dot += a[i] * b[i];
        denomA += a[i] * a[i];
        denomB += b[i] * b[i];
    }
    return static_cast<float>(dot / (std::sqrt(denomA) * std::sqrt(denomB) + 1e-6));
}

std::pair<int, float> searchMostSimilarEmbedding(
    const Matr& embeddings, const std::vector<float>& newComerEmbedding)
{
    if (embeddings.empty() || newComerEmbedding.empty())
        throw std::runtime_error("searchMostSimilarEmbedding: Empty vector");

    int bestId = 0;
    float bestSim = -1.0f;
    for (std::size_t i = 0; i < embeddings.size(); ++i)
    {
        const auto& embedding = embeddings.at(i);
        const auto cosim = cosineSimilarity(embedding, newComerEmbedding);
        if (cosim > bestSim)
        {
            bestSim = cosim;
            bestId = i;
        }
    }
    return {bestId, bestSim};
}

std::vector<float> avgEmbedding(const Matr& embeddings)
{
    const int nEmbeddings = embeddings.size();
    if (0 == nEmbeddings)
        throw std::runtime_error("avgEmbedding: Empty vector");

    if (1 == nEmbeddings)
        return embeddings[0];

    const int embeddingDim = embeddings[0].size();

    std::vector<float> result(embeddingDim, 0.0f);
    for (const auto& embedding : embeddings)
        for (std::size_t i = 0; i < embeddingDim; ++i)
            result[i] += embedding[i];
    for (std::size_t i = 0; i < embeddingDim; ++i)
        result[i] /= nEmbeddings;
    return result;
}

double getAngleBetweenEyes(const std::vector<int>& landmarks)
{
    const cv::Point leftEye(landmarks[0], landmarks[1]);
    const cv::Point rightEye(landmarks[2], landmarks[3]);
    const cv::Point2f eyesCenter = cv::Point2f(
        (leftEye.x + rightEye.x) * 0.5f, 
        (leftEye.y + rightEye.y) * 0.5f);

    // Get the angle between the eyes
    const double dy = rightEye.y - leftEye.y;
    const double dx = rightEye.x - leftEye.x;
    const double len = std::sqrt(dx*dx + dy*dy);
    return std::atan2(dy, dx) * 180.0/M_PI; // Convert from radians to degrees
}

cv::Mat alignFace(
    const cv::Mat& image, cv::Rect faceBoundingBox, const std::vector<int>& landmarks, cv::Size cropSize, 
    cv::Point2f desiredLeftEye)
{
    if (image.empty())
        throw std::runtime_error("alignFace: Empty image");
    if (faceBoundingBox.empty())
        throw std::runtime_error("alignFace: Empty faceBoundingBox");
    if (landmarks.size() < 2)
        throw std::runtime_error("alignFace: Missing landmark coordinates");

    const cv::Point leftEye(landmarks[0], landmarks[1]);
    const cv::Point rightEye(landmarks[2], landmarks[3]);
    const cv::Point2f eyesCenter = cv::Point2f(
        (leftEye.x + rightEye.x) * 0.5f, 
        (leftEye.y + rightEye.y) * 0.5f);

    // Get the angle between the eyes
    const double dy = rightEye.y - leftEye.y;
    const double dx = rightEye.x - leftEye.x;
    const double len = std::sqrt(dx*dx + dy*dy);
    const double angle = std::atan2(dy, dx) * 180.0/M_PI; // Convert from radians to degrees

    // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image
    const double desiredRightEyeX = (1.0f - desiredLeftEye.x);
    // Get the amount we need to scale the image to be the desired fixed size we want
    const double desiredLen = (desiredRightEyeX - desiredLeftEye.x) * cropSize.width;
    const double scale = desiredLen / len;
    // Get the transformation matrix for rotating and scaling the face to the desired angle & size
    cv::Mat R = cv::getRotationMatrix2D(eyesCenter, angle, scale);
    // Shift the center of the eyes to be the desired center between the eyes
    R.at<double>(0, 2) += cropSize.width * 0.5f - eyesCenter.x;
    R.at<double>(1, 2) += cropSize.height * desiredLeftEye.y - eyesCenter.y;

    // Rotate and scale and translate the image to the desired angle & size & position!
    // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
    cv::Mat warped = cv::Mat(cropSize, CV_8UC3);
    cv::warpAffine(image, warped, R, warped.size(), cv::INTER_CUBIC);

    return warped;
}