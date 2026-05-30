// pybind11 bindings exposing the aidall::apriltag C++ core to Python.
//
// The detect() binding takes a 2-D, row-major, uint8 NumPy array (a grayscale
// image). Color-image and convenience handling lives in the Python package.
#include <cstring>

#include "aidall_apriltag/aidall_apriltag.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace at = aidall::apriltag;

namespace {

std::vector<at::Detection> detect_array(at::Detector&                                                        detector,
                                        py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> image) {
    const auto buf = image.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("detect() expects a 2-D grayscale uint8 array (H, W)");
    }
    const int height = static_cast<int>(buf.shape[0]);
    const int width = static_cast<int>(buf.shape[1]);
    const int stride = static_cast<int>(buf.strides[0]);  // bytes per row

    // Release the GIL while the native detector runs.
    std::vector<at::Detection> out;
    {
        py::gil_scoped_release release;
        out = detector.detect(static_cast<const std::uint8_t*>(buf.ptr), width, height, stride);
    }
    return out;
}

// Render a tag straight into a 2-D (H, W) uint8 NumPy array.
py::array_t<std::uint8_t> render_array(at::Family family, int id) {
    at::TagImage              img = at::render(family, id);
    py::array_t<std::uint8_t> arr({img.height, img.width});
    std::memcpy(arr.mutable_data(), img.data.data(), img.data.size());
    return arr;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "aidall_apriltag native core (wraps AprilRobotics/apriltag)";

    py::enum_<at::Family>(m, "Family")
        .value("Tag36h11", at::Family::Tag36h11)
        .value("Tag36h10", at::Family::Tag36h10)
        .value("Tag25h9", at::Family::Tag25h9)
        .value("Tag16h5", at::Family::Tag16h5)
        .value("TagStandard41h12", at::Family::TagStandard41h12)
        .value("TagStandard52h13", at::Family::TagStandard52h13)
        .value("TagCircle21h7", at::Family::TagCircle21h7)
        .value("TagCircle49h12", at::Family::TagCircle49h12)
        .value("TagCustom48h12", at::Family::TagCustom48h12);

    m.def("family_from_string", &at::family_from_string);
    m.def("family_to_string", &at::to_string);
    m.def("family_size", &at::family_size);
    m.def("family_total_width", &at::family_total_width);
    m.def("family_border_width", &at::family_border_width);
    m.def("render", &render_array, py::arg("family"), py::arg("id"));

    py::class_<at::CameraParams>(m, "CameraParams")
        .def(py::init<>())
        .def(py::init([](double fx, double fy, double cx, double cy) { return at::CameraParams{fx, fy, cx, cy}; }),
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"))
        .def_readwrite("fx", &at::CameraParams::fx)
        .def_readwrite("fy", &at::CameraParams::fy)
        .def_readwrite("cx", &at::CameraParams::cx)
        .def_readwrite("cy", &at::CameraParams::cy)
        .def("valid", &at::CameraParams::valid);

    py::class_<at::Pose>(m, "Pose")
        .def_readonly("R", &at::Pose::R)
        .def_readonly("t", &at::Pose::t)
        .def_readonly("object_error", &at::Pose::object_error);

    py::class_<at::Detection>(m, "Detection")
        .def_readonly("id", &at::Detection::id)
        .def_readonly("hamming", &at::Detection::hamming)
        .def_readonly("decision_margin", &at::Detection::decision_margin)
        .def_readonly("center", &at::Detection::center)
        .def_readonly("corners", &at::Detection::corners)
        .def_readonly("pose", &at::Detection::pose);

    py::class_<at::DetectorOptions>(m, "DetectorOptions")
        .def(py::init<>())
        .def_readwrite("family", &at::DetectorOptions::family)
        .def_readwrite("quad_decimate", &at::DetectorOptions::quad_decimate)
        .def_readwrite("quad_sigma", &at::DetectorOptions::quad_sigma)
        .def_readwrite("refine_edges", &at::DetectorOptions::refine_edges)
        .def_readwrite("nthreads", &at::DetectorOptions::nthreads)
        .def_readwrite("tag_size", &at::DetectorOptions::tag_size)
        .def_readwrite("camera", &at::DetectorOptions::camera);

    py::class_<at::Detector>(m, "Detector")
        .def(py::init<at::DetectorOptions>(), py::arg("options") = at::DetectorOptions{})
        .def("detect", &detect_array, py::arg("image"))
        .def("set_camera", &at::Detector::set_camera)
        .def("set_tag_size", &at::Detector::set_tag_size);
}
