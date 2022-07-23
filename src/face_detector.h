#pragma once

#include <string>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

class FaceDetector final
{
public:

    using Landmarks = std::vector<int>;
    struct DetectionResult
    {
        cv::Rect boundingBox;
        Landmarks landmarks;
        float confidence { 0.0f };

        DetectionResult();
        DetectionResult(cv::Rect boundingBox, Landmarks landmarks, float confidence);
        ~DetectionResult();
    };

    FaceDetector(const fs::path& modelpath);
    ~FaceDetector();

    std::vector<DetectionResult> detect(const cv::Mat& image, float minConfidence = 0.45f);

private:
    cv::dnn::Net m_model;
    cv::Mat m_blob;
    std::vector<std::string> m_unconnectedLayersNames;
};