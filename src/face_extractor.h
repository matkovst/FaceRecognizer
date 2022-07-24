#pragma once

#include <string>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/core.hpp>

#include <torch/script.h>
#include <torch/cuda.h>

class FaceExtractor final
{
public:

    static inline const cv::Size InputSize { 112, 112 };
    static inline const cv::Point2f ReferencePoints2[2] = {
        { 0.3155687451f, 0.46157411169f },          // left eye
        { 1.0f - 0.3155687451f, 0.46157411169f }    // right eye
    };
    static inline const cv::Point2f ReferencePoints3[3] = {
        { 0.3155687451f, 0.46157411169f },          // left eye
        { 1.0f - 0.3155687451f, 0.46157411169f },   // right eye
        { 0.50026249885f, 0.64050538196f }          // nose
    };

    using Embedding = std::vector<float>;

    FaceExtractor(const fs::path& modelpath, bool enableGpu = false);
    ~FaceExtractor();

    Embedding extract(const cv::Mat& faceImage);
    std::vector<Embedding> extract(const std::vector<cv::Mat>& faceImages);

private:
    torch::DeviceType m_device;
    torch::jit::script::Module m_model;
    torch::Tensor m_inputTensor;
};