# C++ 사용법

## 설치

C++ 라이브러리를 시스템(또는 사용자 prefix)에 설치한다.

```bash
sudo make install                  # 기본 prefix /usr/local
make install PREFIX=$HOME/.local   # 사용자 prefix
```

## 빌드 / 링크

```cmake
find_package(aidall_apriltag REQUIRED)
target_link_libraries(my_app PRIVATE aidall::apriltag)
```

비표준 prefix면 소비자 구성 시 경로를 알려준다:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/.local
```

## 검출

```cpp
#include "aidall_apriltag/aidall_apriltag.hpp"

aidall::apriltag::Detector detector;        // 기본값 tag36h11

// gray: 빈틈 없이 채워진 8비트 그레이스케일 버퍼, width*height 바이트.
auto detections = detector.detect(gray, width, height);

for (const auto& d : detections) {
  printf("id=%d center=(%.1f, %.1f) margin=%.1f\n",
         d.id, d.center[0], d.center[1], d.decision_margin);
}
```

이미지 버퍼에 패딩이 있으면 명시적 stride(행당 바이트 수)를 지정해야 한다.

```cpp
detector.detect(gray, width, height, row_stride);
```

## 검출기 설정

| 옵션 | 효과 | 언제 바꾸나 |
|--------|--------|----------------|
| `quad_decimate` | 사각형 검출 전 이미지 다운샘플링. `2.0` 면 약 4배 빠름. | 작거나 먼 태그면 `1.0`으로 낮추고, 큰 태그 속도엔 높임. |
| `quad_sigma` | 세그멘테이션 전 가우시안 블러. | 노이즈/저조도엔 높임(`0.8`+), 깨끗하면 `0` 유지. |
| `refine_edges` | 서브픽셀 코너 정제. | 정확한 포즈가 필요하면 `true` 유지. |
| `nthreads` | 검출기 워커 스레드. | 멀티코어에서 고해상도/고fps면 높임. |

```cpp
aidall::apriltag::DetectorOptions opts;
opts.family        = aidall::apriltag::Family::TagStandard41h12;
opts.quad_decimate = 1.0f;     // 속도보다 정확도
opts.quad_sigma    = 0.8f;     // 노이즈 많은 프레임에 도움
opts.refine_edges  = true;
opts.nthreads      = 4;

aidall::apriltag::Detector detector(opts);
```

## 포즈 추정

태그 크기(미터)와 카메라 내부 파라미터를 주면 각 검출이 `pose`를 담는다.

```cpp
aidall::apriltag::DetectorOptions opts;
opts.tag_size = 0.162;                       // 미터, 검은 테두리 에지
opts.camera   = {fx, fy, cx, cy};            // 픽셀
aidall::apriltag::Detector detector(opts);

for (const auto& d : detector.detect(gray, w, h)) {
  if (d.pose) {
    const auto& t = d.pose->t;               // 이동, 미터
    printf("tag %d at (%.3f, %.3f, %.3f)\n", d.id, t[0], t[1], t[2]);
    // d.pose->R 은 3x3 회전, 행 우선(row-major).
  }
}
```

카메라 파라미터 및 태그 사이즈는 Detector 생성 이후에도 설정 가능하다.

```cpp
detector.set_camera({fx, fy, cx, cy});
detector.set_tag_size(0.162);
```

## OpenCV

```cpp
#include "aidall_apriltag/aidall_apriltag.hpp"
#include "aidall_apriltag/opencv.hpp"

cv::Mat frame = cv::imread("scene.png");     // BGR 또는 그레이스케일
auto dets = aidall::apriltag::detect(detector, frame);   // 그레이로 변환
aidall::apriltag::draw(frame, dets);         // 윤곽 + id를 그 자리에 그림
cv::imwrite("out.png", frame);
```

전체 예제는 [`cpp/examples/detect_image.cpp`](../cpp/examples/detect_image.cpp)에
있으며 다음으로 빌드한다.

```bash
cmake -S apriltag -B build -DAIDALL_APRILTAG_BUILD_EXAMPLES=ON
cmake --build build
./build/detect_image scene.png 0.162 600 600 320 240   # img tagsize fx fy cx cy
```

## 수명 & 멀티스레딩

- `Detector`는 **이동 가능, 복사 불가**.
- 스레드 안전하지 않음.
- `detect()`에 넘긴 이미지 버퍼는 호출 후 보관되지 않음.

