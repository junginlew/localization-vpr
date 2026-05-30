cmake_minimum_required(VERSION 3.18)
project(aidall_apriltag VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- Official AprilTag library (git submodule) -----------------------------
# Build it statically and skip the bits we don't need so it bakes cleanly into
# our targets with nothing extra to ship.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_WRAPPER OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/apriltag EXCLUDE_FROM_ALL)

# Compile AprilTag with hidden visibility so, once baked into our shared library,
# its symbols are not exported (no clash with a consumer's own copy). This works
# on every platform; internal references from our wrapper still resolve within
# the same image. (Linux gets belt-and-suspenders via --exclude-libs below.)
set_target_properties(apriltag PROPERTIES
  C_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

# C++ wrapper core
add_library(aidall_apriltag_core STATIC cpp/src/aidall_apriltag.cpp)
target_include_directories(aidall_apriltag_core PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>")
target_link_libraries(aidall_apriltag_core PUBLIC apriltag)
target_compile_features(aidall_apriltag_core PUBLIC cxx_std_17)

# Installable SHARED library
option(AIDALL_APRILTAG_BUILD_SHARED "Build/install the shared C++ library" ON)
if(AIDALL_APRILTAG_BUILD_SHARED AND NOT SKBUILD)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  add_library(aidall_apriltag SHARED cpp/src/aidall_apriltag.cpp)
  add_library(aidall::apriltag ALIAS aidall_apriltag)
  target_compile_features(aidall_apriltag PUBLIC cxx_std_17)
  target_include_directories(aidall_apriltag PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_link_libraries(aidall_apriltag PRIVATE apriltag)  # baked in, not re-exported
  set_target_properties(aidall_apriltag PROPERTIES
    EXPORT_NAME apriltag
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    WINDOWS_EXPORT_ALL_SYMBOLS ON)  # auto-export our API on Windows
  # Hide the baked-in apriltag symbols so they can't clash with a consumer's own
  # copy (GNU/LLVM linkers; our own object symbols stay visible).
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT APPLE)
    target_link_options(aidall_apriltag PRIVATE "LINKER:--exclude-libs,ALL")
  endif()

  install(TARGETS aidall_apriltag EXPORT aidall_apriltagTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY cpp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT aidall_apriltagTargets
    NAMESPACE aidall::
    FILE aidall_apriltagTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aidall_apriltag)

  configure_package_config_file(
    cmake/aidall_apriltagConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/aidall_apriltagConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aidall_apriltag)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/aidall_apriltagConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/aidall_apriltagConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/aidall_apriltagConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aidall_apriltag)
endif()

# --- Python bindings (built only when invoked through scikit-build-core) ----
option(AIDALL_APRILTAG_BUILD_PYTHON "Build the pybind11 module" OFF)
if(AIDALL_APRILTAG_BUILD_PYTHON OR DEFINED SKBUILD)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core python/bindings/bindings.cpp)
  target_link_libraries(_core PRIVATE aidall_apriltag_core)
  # Only this component is packed into the wheel (see pyproject.toml).
  install(TARGETS _core LIBRARY DESTINATION aidall_apriltag COMPONENT python)
endif()

# --- C++ examples (off by default; need OpenCV) ----------------------------
option(AIDALL_APRILTAG_BUILD_EXAMPLES "Build C++ examples (requires OpenCV)" OFF)
if(AIDALL_APRILTAG_BUILD_EXAMPLES)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
  add_executable(detect_image cpp/examples/detect_image.cpp)
  target_link_libraries(detect_image PRIVATE aidall_apriltag_core ${OpenCV_LIBS})
  target_include_directories(detect_image PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

# --- C++ tests -------------------------------------------------------------
option(AIDALL_APRILTAG_BUILD_TESTS "Build C++ tests" OFF)
if(AIDALL_APRILTAG_BUILD_TESTS)
  enable_testing()
  add_executable(test_detector cpp/tests/test_detector.cpp)
  target_link_libraries(test_detector PRIVATE aidall_apriltag_core)
  add_test(NAME test_detector COMMAND test_detector)
endif()
