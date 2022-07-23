#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "face_extractor.h"

namespace
{

constexpr double InputScale { 1.0 / 255.0 };
const cv::Size InputSize { 112, 112 };
constexpr double ScaleAlpha { 1.0 / 127.5 };
constexpr double ScaleBeta { -0.5 / 0.5 };

/* Convert cv::Mat to torch::Tensor */
void matToTensor(const cv::Mat& in, torch::Tensor& out)
{
    const bool isChar = (in.type() & 0xF) < 2;
    const std::vector<int64_t> dims = {in.rows, in.cols, in.channels()};
    out = torch::from_blob(in.data, dims, isChar ? torch::kByte : torch::kFloat);
}

}

FaceExtractor::FaceExtractor(const fs::path& modelpath)
    : m_device(torch::DeviceType::CPU)
{
    try
    {
        // De-serialize ScriptModule from file
        m_model = torch::jit::load(modelpath.string(), m_device);
    }
    catch(const torch::Error& e)
    {
        std::cerr << "FaceExtractor: Could not read model:\n" << e.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "FaceExtractor: Could not read model: unknown error." << std::endl;
    }
}
FaceExtractor::~FaceExtractor() = default;

FaceExtractor::Embedding FaceExtractor::extract(const cv::Mat& faceImage)
{
    if (faceImage.empty())
        throw std::runtime_error("extract: Given empty image");

    /* Pre-process */

    // 1. Resize
    cv::Mat resizedImage;
    cv::resize(faceImage, resizedImage, InputSize, 0.0, 0.0, cv::INTER_CUBIC);

    // 2. Normalize
    cv::Mat normalizedImage; // float32
    resizedImage.convertTo(normalizedImage, CV_32FC3, ScaleAlpha, ScaleBeta);

    // 3. Convert to torch::Tensor
    matToTensor(normalizedImage, m_inputTensor);

    // 4. Make blob
    m_inputTensor = m_inputTensor.permute({2,0,1}); // HWC -> CHW
    m_inputTensor = m_inputTensor.toType(torch::kFloat);
    m_inputTensor.unsqueeze_(0); // CHW -> NCHW
    std::vector<torch::jit::IValue> blob;
    blob.emplace_back(m_inputTensor.to(m_device));

    /* Infer */
    const auto y = m_model.forward(blob);
    torch::Tensor embeddingTensor;
    if (y.isTuple())
        embeddingTensor = y.toTuple()->elements()[0].toTensor();
    else if (y.isTensor())
        embeddingTensor = y.toTensor().detach().clone();

    /* Post-process result */
    Embedding result(
        embeddingTensor.data_ptr<float>(), embeddingTensor.data_ptr<float>() + embeddingTensor.numel());
    return result;
}

std::vector<FaceExtractor::Embedding> FaceExtractor::extract(const std::vector<cv::Mat>& faceImages)
{
    std::vector<Embedding> result;
    result.reserve(faceImages.size());
    for (const auto& faceImage : faceImages)
        result.emplace_back(std::move(extract(faceImage)));
    return result;
}