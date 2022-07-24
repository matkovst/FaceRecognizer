#pragma once

#include <string>
#include <vector>
#include <opencv2/core/types.hpp>

struct Face final
{
    cv::Rect boundingBox;
    std::vector<int> landmarks;
    float confidence;
    int nameId;
    std::string name;
    float similarity;
    cv::Mat crop;

    Face() = default;
    Face(
        cv::Rect boundingBox, 
        std::vector<int> landmarks, 
        float confidence, 
        int nameId, 
        std::string name, 
        cv::Mat crop, 
        float similarity)
            : boundingBox(boundingBox)
            , landmarks(std::move(landmarks))
            , confidence(confidence)
            , nameId(nameId)
            , name(std::move(name))
            , crop(std::move(crop))
            , similarity(similarity)
    {}
    
    ~Face() = default;
};