// Minimal C++ example: detect AprilTags in an image file and optionally
// estimate pose.
//
//   detect_image <image> [tag_size_m fx fy cx cy]
//
// Requires OpenCV (only used here for image loading and the optional preview).
#include <cstdlib>
#include <iostream>

#include "aidall_apriltag/aidall_apriltag.hpp"
#include "aidall_apriltag/opencv.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image> [tag_size_m fx fy cx cy]\n";
        return 1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "failed to read image: " << argv[1] << "\n";
        return 1;
    }

    aidall::apriltag::DetectorOptions opts;
    opts.family = aidall::apriltag::Family::Tag36h11;
    if (argc >= 7) {
        opts.tag_size = std::atof(argv[2]);
        opts.camera = {std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]), std::atof(argv[6])};
    }

    aidall::apriltag::Detector detector(opts);
    auto                       detections = aidall::apriltag::detect(detector, image);

    std::cout << "found " << detections.size() << " tag(s)\n";
    for (const auto& d : detections) {
        std::cout << "  id=" << d.id << " hamming=" << d.hamming << " margin=" << d.decision_margin << " center=("
                  << d.center[0] << ", " << d.center[1] << ")";
        if (d.pose) {
            std::cout << " t=(" << d.pose->t[0] << ", " << d.pose->t[1] << ", " << d.pose->t[2] << ") m";
        }
        std::cout << "\n";
    }

    aidall::apriltag::draw(image, detections);
    cv::imwrite("detections.png", image);
    std::cout << "wrote detections.png\n";
    return 0;
}
