#include "renderer.h"

#include <opencv2/imgproc.hpp>

namespace
{

constexpr float borderSize { 0.2f };
constexpr int Thk { 2 };
const cv::Scalar FaceColor { 50, 255, 0 };

void renderBorderedBoundingBox(cv::Mat& out, cv::Rect boundingBox)
{
    const int borderWidth = borderSize * boundingBox.width;
    const int borderHeight = borderSize * boundingBox.height;

    cv::Point vertex = boundingBox.tl();
    cv::line(out, vertex, cv::Point(vertex.x + borderWidth, vertex.y), FaceColor, Thk);
    cv::line(out, vertex, cv::Point(vertex.x, vertex.y + borderHeight), FaceColor, Thk);
    vertex = boundingBox.tl() + cv::Point(boundingBox.width, 0);
    cv::line(out, vertex, cv::Point(vertex.x - borderWidth, vertex.y), FaceColor, Thk);
    cv::line(out, vertex, cv::Point(vertex.x, vertex.y + borderHeight), FaceColor, Thk);
    vertex = boundingBox.br();
    cv::line(out, vertex, cv::Point(vertex.x - borderWidth, vertex.y), FaceColor, Thk);
    cv::line(out, vertex, cv::Point(vertex.x, vertex.y - borderHeight), FaceColor, Thk);
    vertex = boundingBox.br() - cv::Point(boundingBox.width, 0);
    cv::line(out, vertex, cv::Point(vertex.x + borderWidth, vertex.y), FaceColor, Thk);
    cv::line(out, vertex, cv::Point(vertex.x, vertex.y - borderHeight), FaceColor, Thk);
}

void renderTransparentRect(cv::Mat& out, cv::Rect rect, cv::Scalar color, double opacity)
{
    rect &= cv::Rect(0, 0, out.cols, out.rows);

    cv::Mat roi = out(rect);
    cv::Mat coloredRoi(roi.size(), roi.type(), color);
    cv::addWeighted(coloredRoi, opacity, roi, 1.0 - opacity , 0.0, roi);
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
        // Bounding box
        renderBorderedBoundingBox(out, face.boundingBox);

        // Landmarks
        for (int i = 0; i < 5; ++i)
            cv::circle(out, cv::Point(face.landmarks[2 * i], face.landmarks[2 * i + 1]), 1, FaceColor, -1);

        // Person name
        cv::Scalar nameColor = ("unknown" != face.name) ? FaceColor : cv::Scalar(0, 0, 255); 
        cv::putText(
            out, face.name, face.boundingBox.tl() - cv::Point(0, 15), 
            cv::FONT_HERSHEY_PLAIN, 1.25, nameColor, Thk);

        // Indications
        renderTransparentRect(
            out, cv::Rect(
                face.boundingBox.x + face.boundingBox.width + 1, 
                face.boundingBox.y, 
                140, 
                (face.boundingBox.height > 80) ? face.boundingBox.height : 80), 
            cv::Scalar::all(0), 
            0.4);
        cv::Point origin(face.boundingBox.x + face.boundingBox.width + 10, face.boundingBox.y + 5);
        const cv::Point offset(0, 20);
        cv::putText(
            out, cv::format("pid: %d", face.nameId), 
            origin += offset, cv::FONT_HERSHEY_PLAIN, 1.2, nameColor, 1);
        cv::putText(
            out, cv::format("conf: %.2f", face.confidence), 
            origin += offset, cv::FONT_HERSHEY_PLAIN, 1.2, FaceColor, 1);
        cv::putText(
            out, cv::format("cosine: %.2f", face.similarity), 
            origin += offset, cv::FONT_HERSHEY_PLAIN, 1.2, nameColor, 1);
    }
}