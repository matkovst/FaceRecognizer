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

    using Embedding = std::vector<float>;

    FaceExtractor(const fs::path& modelpath);
    ~FaceExtractor();

    Embedding extract(const cv::Mat& faceImage);
    std::vector<Embedding> extract(const std::vector<cv::Mat>& faceImages);

private:
    torch::DeviceType m_device;
    torch::jit::script::Module m_model;
    torch::Tensor m_inputTensor;
};