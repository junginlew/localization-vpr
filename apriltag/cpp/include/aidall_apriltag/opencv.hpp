// Optional OpenCV convenience helpers for aidall_apriltag.
//
// This header is header-only and pulls in OpenCV. Include it ONLY in
// translation units that already depend on OpenCV — the core library does not.
//
//   #include "aidall_apriltag/aidall_apriltag.hpp"
//   #include "aidall_apriltag/opencv.hpp"
//
//   cv::Mat frame = ...;                 // BGR or grayscale
//   auto dets = aidall::apriltag::detect(detector, frame);
#pragma once

#include <vector>

#include "aidall_apriltag/aidall_apriltag.hpp"

#include <opencv2/imgproc.hpp>

namespace aidall::apriltag {

// Detect tags in a cv::Mat. Color images are converted to grayscale; an
// already-grayscale (CV_8UC1) Mat is used directly without a copy.
inline std::vector<Detection> detect(Detector& detector, const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::invalid_argument("detect(): unsupported channel count");
    }
    if (gray.type() != CV_8UC1) {
        gray.convertTo(gray, CV_8UC1);
    }
    return detector.detect(gray.data, gray.cols, gray.rows, static_cast<int>(gray.step));
}

// Draw detected tag outlines + ids onto a BGR image, in place.
inline void draw(cv::Mat& bgr, const std::vector<Detection>& detections) {
    for (const auto& d : detections) {
        for (int i = 0; i < 4; ++i) {
            const auto& a = d.corners[i];
            const auto& b = d.corners[(i + 1) % 4];
            cv::line(bgr, cv::Point2d(a[0], a[1]), cv::Point2d(b[0], b[1]), cv::Scalar(0, 255, 0), 2);
        }
        cv::putText(bgr, std::to_string(d.id), cv::Point2d(d.center[0], d.center[1]), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2);
    }
}

}  // namespace aidall::apriltag
