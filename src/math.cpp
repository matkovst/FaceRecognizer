#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "math.h"

template<typename T>
cv::Mat vec2mat(const std::vector<std::vector<T>>& vec)
{
    const auto rows = vec.size();
    if (0 == rows)
        throw std::runtime_error("vec2mat: 0 == rows");
    const auto cols = vec.at(0).size();
    if (0 == cols)
        throw std::runtime_error("vec2mat: 0 == cols");

    int cvType = -1;
    if (std::is_same<T, std::uint8_t>::value)
        cvType = CV_8U;
    else if (std::is_same<T, std::float_t>::value)
        cvType = CV_32F;
    else
        throw std::runtime_error("vec2mat: unsupported T value");

    cv::Mat result(rows, cols, cvType);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.at<T>(i, j) = vec[i][j];
    return result;
}
template cv::Mat vec2mat<std::uint8_t>(const std::vector<std::vector<std::uint8_t>>& vec);
template cv::Mat vec2mat<std::float_t>(const std::vector<std::vector<std::float_t>>& vec);

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

cv::Mat alignFace2(
    const cv::Mat& image, cv::Rect faceBoundingBox, const std::vector<int>& landmarks, cv::Size cropSize, 
    const cv::Point2f refPoints2[2])
{
    if (image.empty())
        throw std::runtime_error("alignFace2: Empty image");
    if (faceBoundingBox.empty())
        throw std::runtime_error("alignFace2: Empty faceBoundingBox");
    if (landmarks.size() < 2)
        throw std::runtime_error("alignFace2: Missing landmark coordinates");

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

    // Get the amount we need to scale the image to be the desired fixed size we want
    const double desiredLen = (refPoints2[1].x - refPoints2[0].x) * cropSize.width;
    const double scale = desiredLen / len;
    // Get the transformation matrix for rotating and scaling the face to the desired angle & size
    cv::Mat R = cv::getRotationMatrix2D(eyesCenter, angle, scale);
    // Shift the center of the eyes to be the desired center between the eyes
    R.at<double>(0, 2) += cropSize.width * 0.5f - eyesCenter.x;
    R.at<double>(1, 2) += cropSize.height * refPoints2[0].y - eyesCenter.y;

    // Rotate and scale and translate the image to the desired angle & size & position!
    // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
    cv::Mat warped = cv::Mat(cropSize, CV_8UC3);
    cv::warpAffine(image, warped, R, warped.size(), cv::INTER_CUBIC);

    return warped;
}

cv::Mat alignFace3(
    const cv::Mat& image, cv::Rect faceBoundingBox, const std::vector<int>& landmarks, cv::Size cropSize, 
    const cv::Point2f refPoints3[3])
{
    if (image.empty())
        throw std::runtime_error("alignFace3: Empty image");
    if (faceBoundingBox.empty())
        throw std::runtime_error("alignFace3: Empty faceBoundingBox");

    cv::Point2f srcPoints3[3];
    srcPoints3[0] = cv::Point2f(landmarks[0], landmarks[1]); // left eye
    srcPoints3[1] = cv::Point2f(landmarks[2], landmarks[3]); // right eye
    srcPoints3[2] = cv::Point2f(landmarks[4], landmarks[5]); // nose

    cv::Point2f scaledRefPoints3[3];
    scaledRefPoints3[0] = cv::Point2f(refPoints3[0].x * faceBoundingBox.width, refPoints3[0].y * faceBoundingBox.height);
    scaledRefPoints3[1] = cv::Point2f(refPoints3[1].x * faceBoundingBox.width, refPoints3[1].y * faceBoundingBox.height);
    scaledRefPoints3[2] = cv::Point2f(refPoints3[2].x * faceBoundingBox.width, refPoints3[2].y * faceBoundingBox.height);

    const cv::Mat T = cv::getAffineTransform(srcPoints3, scaledRefPoints3);
    cv::Mat warped = cv::Mat(cropSize, CV_8UC3);
    cv::warpAffine(image, warped, T, warped.size(), cv::INTER_CUBIC);

    return warped;
}


PeriodicTrigger::PeriodicTrigger(std::int64_t frequency)
    : m_frequency(frequency)
{}
PeriodicTrigger::~PeriodicTrigger() = default;

bool PeriodicTrigger::rocknroll(std::int64_t now)
{
    if (0 > m_frequency)
        return false;
    if (0 == m_frequency)
        return true;

    if (-1 == m_lastTriggered) // первый заход
    {
        m_lastTriggered = now;
        return true;
    }

    const auto elapsed = now - m_lastTriggered;
    if (elapsed < m_frequency)
        return false;

    m_lastTriggered = now - (now % m_frequency);
    return true;
}