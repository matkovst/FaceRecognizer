#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "math.h"

class BoxTracker final
{
public:

    /**
     * @brief Takes a cv::Rect and returns [cx,cy,s,r] where cx,cy is the centre of the box 
     * and s is the scale/area and r is the aspect ratio
     */
    static cv::Mat to_xysr(cv::Rect bbox);

    /**
     * @brief Takes [cx,cy,s,r] and returns it in the cv::Rect form
     */
    static cv::Rect to_xywh(const cv::Mat& state, cv::Rect sceneRect);

    explicit BoxTracker(cv::Size sceneSize, float measurementNoise = 0.1f);
    ~BoxTracker();

    void init(cv::Rect bbox);

    cv::Rect update(cv::Rect bbox = cv::Rect());

    bool initialized() const noexcept;

private:
    const std::vector<std::vector<float>> F = {
        {1,0,0,0,1,0,0}, 
        {0,1,0,0,0,1,0}, 
        {0,0,1,0,0,0,1}, 
        {0,0,0,1,0,0,0}, 
        {0,0,0,0,1,0,0}, 
        {0,0,0,0,0,1,0}, 
        {0,0,0,0,0,0,1}
    };
    const std::vector<std::vector<float>> H = {
        {1,0,0,0,0,0,0},
        {0,1,0,0,0,0,0},
        {0,0,1,0,0,0,0},
        {0,0,0,1,0,0,0},
    };
    const int StateDim { 7 };
    const int MeasDim { 4 };
    
    cv::Rect m_sceneRect;
    cv::KalmanFilter m_kf;
    bool m_initialized;
};