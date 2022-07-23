#include <cmath>
#include <stdexcept>
#include <iostream>

#include "face_detector.h"

namespace
{

constexpr double InputScale { 1.0 / 255.0 };
constexpr float NmsThreshold { 0.25f };
constexpr int cellDimention { 16 }; // xmin, ymin, xamx, ymax, box_score, x1, y1, ... ,x5, y5, face_score
const cv::Size InputSize { 640, 640 };

}


FaceDetector::DetectionResult::DetectionResult() = default;
FaceDetector::DetectionResult::DetectionResult(
    cv::Rect boundingBox, FaceDetector::Landmarks landmarks, float confidence)
        : boundingBox(boundingBox)
        , landmarks(std::move(landmarks))
        , confidence(confidence)
{}
FaceDetector::DetectionResult::~DetectionResult() = default;


FaceDetector::FaceDetector(const fs::path& modelpath, bool enableGpu)
{
    try
    {
        m_model = cv::dnn::readNet(modelpath.string());
        if (enableGpu) // dummy-style without checking GPU availability.
        {
            m_model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        m_unconnectedLayersNames = m_model.getUnconnectedOutLayersNames();
    }
    catch(const cv::Exception& e)
    {
        std::cerr << "FaceDetector: Could not read model:\n" << e.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "FaceDetector: Could not read model: unknown error." << std::endl;
    }
}
FaceDetector::~FaceDetector() = default;

std::vector<FaceDetector::DetectionResult>
FaceDetector::detect(const cv::Mat& image, float minConfidence)
{
    if (image.empty())
        throw std::runtime_error("detect: Given empty image");
    if (m_model.empty())
        throw std::runtime_error("detect: Model is not initialized");

    /* Pre-process image */
    cv::dnn::blobFromImage(
        image, m_blob, InputScale, InputSize, cv::Scalar(0, 0, 0), true, false);
    m_model.setInput(m_blob);
    std::vector<cv::Mat> outs;

    /* Infer */
    m_model.forward(outs, m_unconnectedLayersNames);

    /* Post-process result */

    const float scalex = static_cast<float>(image.cols) / InputSize.width;
	const float scaley = static_cast<float>(image.rows) / InputSize.height;
    const float* data = reinterpret_cast<float*>(outs[0].data);
    const auto nCells = outs[0].size().width;

    std::vector<cv::Rect> resultBoxes;
	std::vector<float> resultConfidences;
	std::vector<Landmarks> resultLandmarks;
    for (int cell = 0; cell < nCells; ++cell)
    {
        const auto objConfidence = data[cellDimention * cell + 4];
        if (objConfidence < minConfidence)
            continue;
        const auto classConfidence = data[cellDimention * cell + 15];
        const auto totalConfidence = objConfidence * classConfidence;
        if (totalConfidence < minConfidence)
            continue;

        const auto w = static_cast<int>(data[cellDimention * cell + 2] * scalex);
        const auto h = static_cast<int>(data[cellDimention * cell + 3] * scaley);
        const auto x = static_cast<int>(data[cellDimention * cell + 0] * scalex - 0.5 * w);
        const auto y = static_cast<int>(data[cellDimention * cell + 1] * scaley - 0.5 * h);
        const cv::Rect faceBoundingBox = cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);

		Landmarks cellLandmarks;
        cellLandmarks.reserve(10);
		for (int k = 5; k < 15; k+=2)
		{
			cellLandmarks.push_back(static_cast<int>(data[cellDimention * cell + k] * scalex));
			cellLandmarks.push_back(static_cast<int>(data[cellDimention * cell + k + 1] * scaley));
		}

        resultConfidences.push_back(totalConfidence);
        resultBoxes.emplace_back(faceBoundingBox);
		resultLandmarks.emplace_back(std::move(cellLandmarks));
    }

	std::vector<int> indices;
	cv::dnn::NMSBoxes(resultBoxes, resultConfidences, minConfidence, NmsThreshold, indices);

    std::vector<DetectionResult> result;
    result.reserve(indices.size());
    for (auto index : indices)
        result.emplace_back(
            resultBoxes[index], resultLandmarks[index], resultConfidences[index]);

    return result;
}