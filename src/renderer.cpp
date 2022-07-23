#include "renderer.h"

#include <opencv2/imgproc.hpp>

namespace
{

constexpr float borderSize { 0.2f };
const cv::Scalar FaceColor { 50, 255, 0 };

void renderBorderedBoundingBox(cv::Mat& out, cv::Rect boundingBox)
{
    const int borderWidth = borderSize * boundingBox.width;
    const int borderHeight = borderSize * boundingBox.height;

    cv::Point vertex = boundingBox.tl();
    cv::line(out, vertex, cv::Point(vertex.x + borderWidth, vertex.y), FaceColor, 2);
    cv::line(out, vertex, cv::Point(vertex.x, vertex.y + borderHeight), FaceColor, 2);
    vertex = boundingBox.tl() + cv::Point(boundingBox.width, 0);
    cv::line(out, vertex, cv::Point(vertex.x - borderWidth, vertex.y), FaceColor, 2);
    cv::line(out, vertex, cv::Point(vertex.x, vertex.y + borderHeight), FaceColor, 2);
    vertex = boundingBox.br();
    cv::line(out, vertex, cv::Point(vertex.x - borderWidth, vertex.y), FaceColor, 2);
    cv::line(out, vertex, cv::Point(vertex.x, vertex.y - borderHeight), FaceColor, 2);
    vertex = boundingBox.br() - cv::Point(boundingBox.width, 0);
    cv::line(out, vertex, cv::Point(vertex.x + borderWidth, vertex.y), FaceColor, 2);
    cv::line(out, vertex, cv::Point(vertex.x, vertex.y - borderHeight), FaceColor, 2);
}

}

void renderBoundingBoxes(cv::Mat& out, std::vector<cv::Rect> boundingBoxes)
{
    for (const auto boundingBox : boundingBoxes)
        renderBorderedBoundingBox(out, boundingBox);
}

void renderLandmarks(cv::Mat& out, std::vector<std::vector<int>> landmarks)
{
    for (const auto& landmark : landmarks)
        for (int i = 0; i < 5; ++i)
            cv::circle(out, cv::Point(landmark[2 * i], landmark[2 * i + 1]), 1, FaceColor, -1);
}

void renderFaces(cv::Mat& out, std::vector<Face> faces)
{
    for (const auto face : faces)
    {
        renderBorderedBoundingBox(out, face.boundingBox);

        for (int i = 0; i < 5; ++i)
            cv::circle(out, cv::Point(face.landmarks[2 * i], face.landmarks[2 * i + 1]), 1, FaceColor, -1);

        cv::Scalar nameColor = ("unknown" != face.name) ? FaceColor : cv::Scalar(0, 0, 255); 
        cv::putText(
            out, face.name, face.boundingBox.tl() - cv::Point(0, 15), 
            cv::FONT_HERSHEY_PLAIN, 1.2, nameColor, 2);

        cv::putText(
            out, cv::format("pid: %d", face.nameId), 
            cv::Point(face.boundingBox.x + face.boundingBox.width + 10, face.boundingBox.y + 15), 
            cv::FONT_HERSHEY_PLAIN, 1.1, nameColor, 1);
        cv::putText(
            out, cv::format("conf: %.2f", face.confidence), 
            cv::Point(face.boundingBox.x + face.boundingBox.width + 10, face.boundingBox.y + 35), 
            cv::FONT_HERSHEY_PLAIN, 1.1, FaceColor, 1);
        cv::putText(
            out, cv::format("cosine: %.2f", face.similarity), 
            cv::Point(face.boundingBox.x + face.boundingBox.width + 10, face.boundingBox.y + 55), 
            cv::FONT_HERSHEY_PLAIN, 1.1, nameColor, 1);
    }
}