#include <algorithm>
#include "box_tracker.h"

cv::Mat BoxTracker::to_xysr(cv::Rect bbox)
{
    const float cx = bbox.x + static_cast<float>(bbox.width / 2);
    const float cy = bbox.y + static_cast<float>(bbox.height / 2);
    const float s = bbox.area();
    const float r = bbox.width / static_cast<float>(bbox.height);
    return cv::Mat_<float>(4, 1) << cx, cy, s, r;
}

cv::Rect BoxTracker::to_xywh(const cv::Mat& state, cv::Rect sceneRect)
{
    const auto cx = state.at<float>(0);
    const auto cy = state.at<float>(1);
    const auto s = state.at<float>(2);
    const auto r = state.at<float>(3);
    const auto w = std::sqrt(s * r);
    const auto h = s / w;
    return cv::Rect(cx - w/2, cy - h/2, w, h) & sceneRect;
}

BoxTracker::BoxTracker(cv::Size sceneSize, float measurementNoise)
    : m_sceneRect(0, 0, sceneSize.width, sceneSize.height)
    , m_kf(StateDim, MeasDim)
    , m_initialized(false)
{
    measurementNoise = std::clamp(measurementNoise, 0.0f, 1.0f);

    /* Set up matrices */

    m_kf.transitionMatrix = vec2mat(F).clone();
    m_kf.measurementMatrix = vec2mat(H).clone();
    cv::setIdentity(m_kf.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    cv::setIdentity(m_kf.errorCovPost, cv::Scalar::all(10.0f));
    m_kf.errorCovPost.at<float>(4, 4) = 1000.0f;
    cv::setIdentity(m_kf.processNoiseCov, cv::Scalar::all(1.0f));
    m_kf.processNoiseCov.at<float>(4, 4) = 0.2f;
    m_kf.processNoiseCov.at<float>(5, 5) = 0.2f;
    m_kf.processNoiseCov.at<float>(6, 6) = 0.2f * 0.2f;
}

BoxTracker::~BoxTracker() = default;

void BoxTracker::init(cv::Rect bbox)
{
    if (bbox.empty())
        throw std::runtime_error("BoxTracker::init: Empty bbox");

    const auto xysr = to_xysr(bbox);
    for (int i = 0; i < 4; ++i)
        m_kf.statePost.at<float>(i) = xysr.at<float>(i);
    m_initialized = true;
}

cv::Rect BoxTracker::update(cv::Rect bbox)
{
    if (!m_initialized)
    {
        if (bbox.empty())
            return cv::Rect();
        init(bbox);
        return to_xywh(m_kf.statePost, m_sceneRect);
    }

    const auto predState = m_kf.predict();
    if (bbox.empty())
    {
        return to_xywh(predState, m_sceneRect);
    }
    else
    {
        const auto corrState = m_kf.correct(to_xysr(bbox));
        return to_xywh(corrState, m_sceneRect);
    }
}

bool BoxTracker::initialized() const noexcept
{
    return m_initialized;
}