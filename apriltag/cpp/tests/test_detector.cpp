#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "aidall_apriltag/aidall_apriltag.hpp"

namespace at = aidall::apriltag;

int main() {
    // Family name round-trips.
    auto fam = at::family_from_string("tag36h11");
    assert(fam.has_value());
    assert(at::to_string(*fam) == "tag36h11");
    assert(!at::family_from_string("nope").has_value());

    // A blank image yields no detections, and the detector cleans up after
    // itself.
    const int                 w = 64, h = 48;
    std::vector<std::uint8_t> blank(w * h, 0);
    at::Detector              detector;
    auto                      dets = detector.detect(blank.data(), w, h);
    assert(dets.empty());

    // Invalid input throws.
    bool threw = false;
    try {
        detector.detect(nullptr, w, h);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "all C++ smoke tests passed\n";
    return 0;
}
