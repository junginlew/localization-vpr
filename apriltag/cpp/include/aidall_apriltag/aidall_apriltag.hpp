// Official library: https://github.com/AprilRobotics/apriltag (BSD-2-Clause).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace aidall::apriltag {

// The tag families shipped with the official library. tag36h11 is the
// recommended default: it has the lowest false-positive rate of the classic
// families and is what most robotics setups use.
enum class Family {
    Tag36h11,
    Tag36h10,
    Tag25h9,
    Tag16h5,
    TagStandard41h12,
    TagStandard52h13,
    TagCircle21h7,
    TagCircle49h12,
    TagCustom48h12,
};

// Parse a family from its canonical string name (e.g. "tag36h11").
// Returns std::nullopt for unknown names.
std::optional<Family> family_from_string(const std::string& name);
std::string           to_string(Family family);

// A rendered tag image at native resolution: one pixel per tag cell, including
// the tag's own border but NOT any surrounding quiet zone. 0 = black,
// 255 = white, row-major (data.size() == width * height).
struct TagImage {
    int                       width = 0;
    int                       height = 0;
    std::vector<std::uint8_t> data;
};

// Render tag `id` of `family` to its native cell grid (see TagImage). Reuses
// the official apriltag_to_image(), so a rendered tag is guaranteed decodable
// by Detector. Throws std::out_of_range if `id` is not in the family.
TagImage render(Family family, int id);

// Number of distinct IDs (codes) available in a family.
int family_size(Family family);

// Native grid geometry, in cells. total_width is the full rendered grid edge
// (what render() produces); border_width is the black-border edge — the length
// that corresponds to the pose `tag_size`. Their ratio lets you size a print so
// the black border matches a desired physical tag_size.
int family_total_width(Family family);
int family_border_width(Family family);

// Pinhole camera intrinsics, in pixels. Required for 6-DOF pose estimation.
struct CameraParams {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    bool valid() const { return fx > 0.0 && fy > 0.0; }
};

// The 6-DOF pose of a tag in the camera optical frame, as recovered by the
// official estimate_tag_pose(). R is row-major 3x3, t is in the same units as
// the configured tag size (meters by convention).
struct Pose {
    std::array<std::array<double, 3>, 3> R{};                 // rotation, camera <- tag
    std::array<double, 3>                t{};                 // translation (tag origin in camera frame)
    double                               object_error = 0.0;  // reprojection error returned by the solver
};

// A single detected tag.
struct Detection {
    int                                  id = -1;
    int                                  hamming = 0;          // corrected error bits (lower is more confident)
    float                                decision_margin = 0;  // decode quality; higher is better
    std::array<double, 2>                center{};
    // Corners in image pixel coordinates, wrapping counter-clockwise.
    std::array<std::array<double, 2>, 4> corners{};
    // Populated only when pose estimation is enabled (tag size + camera set).
    std::optional<Pose>                  pose;
};

// Knobs forwarded to the underlying apriltag_detector_t, plus the bits this
// wrapper needs for pose estimation.
struct DetectorOptions {
    Family family = Family::Tag36h11;

    // Lower-resolution detection of quads, then refine. 2.0 is a good speed
    // default; set to 1.0 for maximum accuracy on small/distant tags.
    float quad_decimate = 2.0f;
    // Gaussian blur applied before segmentation. >0 helps very noisy images.
    float quad_sigma = 0.0f;
    // Spatially refine quad edges. Improves pose accuracy at a small cost.
    bool  refine_edges = true;
    // Detector worker threads.
    int   nthreads = 1;

    // Physical tag size (black border included), in meters. When > 0 AND camera
    // intrinsics are provided, detect() fills Detection::pose.
    double       tag_size = 0.0;
    CameraParams camera{};
};

// RAII wrapper around apriltag_detector_t and its family. Non-copyable,
// movable. Not thread-safe: use one Detector per thread.
class Detector {
  public:
    explicit Detector(DetectorOptions options = {});
    ~Detector();

    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    // Detect tags in an 8-bit grayscale image. `stride` is the number of bytes
    // per row (defaults to `width` for tightly packed buffers). The buffer is
    // not retained after the call returns.
    std::vector<Detection> detect(const std::uint8_t* gray, int width, int height, int stride);
    std::vector<Detection> detect(const std::uint8_t* gray, int width, int height) {
        return detect(gray, width, height, width);
    }

    // Update camera intrinsics / tag size after construction.
    void set_camera(const CameraParams& camera);
    void set_tag_size(double meters);

    const DetectorOptions& options() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aidall::apriltag
