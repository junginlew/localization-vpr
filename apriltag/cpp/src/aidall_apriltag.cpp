#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#include "aidall_apriltag/aidall_apriltag.hpp"

extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "common/image_u8.h"
#include "common/matd.h"
#include "tag16h5.h"
#include "tag25h9.h"
#include "tag36h10.h"
#include "tag36h11.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
}

namespace aidall::apriltag {

namespace {

struct FamilyVtable {
    const char* name;
    apriltag_family_t* (*create)();
    void (*destroy)(apriltag_family_t*);
};

const FamilyVtable& vtable_for(Family family) {
    static const std::unordered_map<Family, FamilyVtable> kTable = {
        {Family::Tag36h11, {"tag36h11", tag36h11_create, tag36h11_destroy}},
        {Family::Tag36h10, {"tag36h10", tag36h10_create, tag36h10_destroy}},
        {Family::Tag25h9, {"tag25h9", tag25h9_create, tag25h9_destroy}},
        {Family::Tag16h5, {"tag16h5", tag16h5_create, tag16h5_destroy}},
        {Family::TagStandard41h12, {"tagStandard41h12", tagStandard41h12_create, tagStandard41h12_destroy}},
        {Family::TagStandard52h13, {"tagStandard52h13", tagStandard52h13_create, tagStandard52h13_destroy}},
        {Family::TagCircle21h7, {"tagCircle21h7", tagCircle21h7_create, tagCircle21h7_destroy}},
        {Family::TagCircle49h12, {"tagCircle49h12", tagCircle49h12_create, tagCircle49h12_destroy}},
        {Family::TagCustom48h12, {"tagCustom48h12", tagCustom48h12_create, tagCustom48h12_destroy}},
    };
    return kTable.at(family);
}

}  // namespace

std::optional<Family> family_from_string(const std::string& name) {
    static const std::unordered_map<std::string, Family> kByName = {
        {"tag36h11", Family::Tag36h11},
        {"tag36h10", Family::Tag36h10},
        {"tag25h9", Family::Tag25h9},
        {"tag16h5", Family::Tag16h5},
        {"tagStandard41h12", Family::TagStandard41h12},
        {"tagStandard52h13", Family::TagStandard52h13},
        {"tagCircle21h7", Family::TagCircle21h7},
        {"tagCircle49h12", Family::TagCircle49h12},
        {"tagCustom48h12", Family::TagCustom48h12},
    };
    auto it = kByName.find(name);
    if (it == kByName.end())
        return std::nullopt;
    return it->second;
}

std::string to_string(Family family) {
    return vtable_for(family).name;
}

// Owns a family for the duration of a render/size call.
namespace {
struct FamilyHandle {
    apriltag_family_t* tf = nullptr;
    void (*destroy)(apriltag_family_t*) = nullptr;
    explicit FamilyHandle(Family family) {
        const FamilyVtable& vt = vtable_for(family);
        tf = vt.create();
        destroy = vt.destroy;
    }
    ~FamilyHandle() {
        if (tf && destroy)
            destroy(tf);
    }
    FamilyHandle(const FamilyHandle&) = delete;
    FamilyHandle& operator=(const FamilyHandle&) = delete;
};
}  // namespace

int family_size(Family family) {
    FamilyHandle h(family);
    return static_cast<int>(h.tf->ncodes);
}

int family_total_width(Family family) {
    FamilyHandle h(family);
    return h.tf->total_width;
}

int family_border_width(Family family) {
    FamilyHandle h(family);
    return h.tf->width_at_border;
}

TagImage render(Family family, int id) {
    FamilyHandle h(family);
    if (id < 0 || static_cast<std::uint32_t>(id) >= h.tf->ncodes) {
        throw std::out_of_range("render(): id out of range for family " + to_string(family));
    }

    image_u8_t* im = apriltag_to_image(h.tf, static_cast<std::uint32_t>(id));
    TagImage    out;
    out.width = im->width;
    out.height = im->height;
    out.data.resize(static_cast<std::size_t>(im->width) * im->height);
    // Copy row by row to drop any stride padding.
    for (int y = 0; y < im->height; ++y) {
        const std::uint8_t* row = im->buf + static_cast<std::size_t>(y) * im->stride;
        std::copy(row, row + im->width, out.data.begin() + static_cast<std::size_t>(y) * im->width);
    }
    image_u8_destroy(im);
    return out;
}

struct Detector::Impl {
    DetectorOptions      options;
    apriltag_detector_t* td = nullptr;
    apriltag_family_t*   tf = nullptr;
    void (*tf_destroy)(apriltag_family_t*) = nullptr;

    explicit Impl(DetectorOptions opts) : options(std::move(opts)) {
        const FamilyVtable& vt = vtable_for(options.family);
        tf = vt.create();
        tf_destroy = vt.destroy;
        td = apriltag_detector_create();
        apriltag_detector_add_family(td, tf);
        apply_options();
    }

    ~Impl() {
        if (td)
            apriltag_detector_destroy(td);
        if (tf && tf_destroy)
            tf_destroy(tf);
    }

    void apply_options() {
        td->quad_decimate = options.quad_decimate;
        td->quad_sigma = options.quad_sigma;
        td->refine_edges = options.refine_edges ? 1 : 0;
        td->nthreads = options.nthreads;
    }

    bool pose_enabled() const { return options.tag_size > 0.0 && options.camera.valid(); }
};

Detector::Detector(DetectorOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

void Detector::set_camera(const CameraParams& camera) {
    impl_->options.camera = camera;
}
void Detector::set_tag_size(double meters) {
    impl_->options.tag_size = meters;
}
const DetectorOptions& Detector::options() const {
    return impl_->options;
}

std::vector<Detection> Detector::detect(const std::uint8_t* gray, int width, int height, int stride) {
    if (gray == nullptr || width <= 0 || height <= 0 || stride < width) {
        throw std::invalid_argument("detect(): invalid image buffer or dimensions");
    }

    // image_u8_t fields are const, so wrap the caller's buffer without copying.
    image_u8_t im{width, height, stride, const_cast<std::uint8_t*>(gray)};

    zarray_t*              raw = apriltag_detector_detect(impl_->td, &im);
    std::vector<Detection> out;
    const int              n = zarray_size(raw);
    out.reserve(n);

    for (int i = 0; i < n; ++i) {
        apriltag_detection_t* det = nullptr;
        zarray_get(raw, i, &det);

        Detection d;
        d.id = det->id;
        d.hamming = det->hamming;
        d.decision_margin = det->decision_margin;
        d.center = {det->c[0], det->c[1]};
        for (int c = 0; c < 4; ++c) {
            d.corners[c] = {det->p[c][0], det->p[c][1]};
        }

        if (impl_->pose_enabled()) {
            apriltag_detection_info_t info;
            info.det = det;
            info.tagsize = impl_->options.tag_size;
            info.fx = impl_->options.camera.fx;
            info.fy = impl_->options.camera.fy;
            info.cx = impl_->options.camera.cx;
            info.cy = impl_->options.camera.cy;

            apriltag_pose_t pose;
            const double    err = estimate_tag_pose(&info, &pose);

            Pose p;
            p.object_error = err;
            for (int r = 0; r < 3; ++r) {
                for (int cc = 0; cc < 3; ++cc) {
                    p.R[r][cc] = MATD_EL(pose.R, r, cc);
                }
                p.t[r] = MATD_EL(pose.t, r, 0);
            }
            matd_destroy(pose.R);
            matd_destroy(pose.t);
            d.pose = p;
        }

        out.push_back(std::move(d));
    }

    apriltag_detections_destroy(raw);
    return out;
}

}  // namespace aidall::apriltag
