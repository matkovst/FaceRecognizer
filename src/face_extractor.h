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

    static inline const cv::Point2f DesiredLeftEye { 0.3155687451f, 0.46157411169f };
    static inline const cv::Size InputSize { 112, 112 };

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