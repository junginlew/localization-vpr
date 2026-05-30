# Python 사용법

## 빌드 및 설치

### uv

#### 빌드

```shell
uv build --wheel
```

`dist/` 폴더에 빌드된 `.whl` 파일이 생성된다. 빌드된 휠은 print 엑스트라를 포함한다.

#### 설치

```shell
uv add dist/aidall_apriltag-*.whl         # 프로젝트에 의존성으로 추가
uv pip install dist/aidall_apriltag-*.whl # 단순 설치
```

### pip

#### 빌드

```shell
python -m build --wheel
```

## 검출

```python
import aidall_apriltag as at

detector = at.Detector(family="tag36h11")   # 패밀리 이름 문자열 또는 at.Family.*

# image: NumPy uint8 배열, 그레이스케일 (H, W) 또는 컬러 (H, W, 3/4).
for d in detector.detect(image):
    print(d.id, d.center, d.decision_margin)
```

`Detection`이 노출하는 속성:

| 속성 | 타입 | 의미 |
|-----------|------|------|
| `id` | `int` | 디코딩된 태그 ID |
| `hamming` | `int` | 정정된 오류 비트 (0 = 깨끗) |
| `decision_margin` | `float` | 디코딩 신뢰도 (높을수록 좋음) |
| `center` | `(x, y)` | 중심(픽셀) |
| `corners` | 4×`(x, y)` | 코너, 반시계 방향 |
| `pose` | `Pose` 또는 `None` | 활성화 시 6-DOF 포즈 |

## 검출기 설정

| 옵션 | 효과 | 언제 바꾸나 |
|--------|--------|----------------|
| `quad_decimate` | 사각형 검출 전 이미지 다운샘플링. `2.0` 면 약 4배 빠름. | 작거나 먼 태그면 `1.0`으로 낮추고, 큰 태그 속도엔 높임. |
| `quad_sigma` | 세그멘테이션 전 가우시안 블러. | 노이즈/저조도엔 높임(`0.8`+), 깨끗하면 `0` 유지. |
| `refine_edges` | 서브픽셀 코너 정제. | 정확한 포즈가 필요하면 `true` 유지. |
| `nthreads` | 검출기 워커 스레드. | 멀티코어에서 고해상도/고fps면 높임. |

```python
detector = at.Detector(
    family="tagStandard41h12",
    quad_decimate=1.0,     # 속도보다 정확도
    quad_sigma=0.8,        # 저조도 프레임 노이즈 완화
    refine_edges=True,
    nthreads=4,
)
```

## 포즈 추정

```python
import aidall_apriltag as at

detector = at.Detector(
    family="tag36h11",
    tag_size=0.162,                          # 미터 (검은 테두리 에지)
    camera=at.CameraParams(fx, fy, cx, cy),  # 픽셀
)

for d in detector.detect(frame):
    if d.pose is not None:
        tx, ty, tz = d.pose.t                # 이동, 미터
        print(f"tag {d.id}: {tx:.3f} {ty:.3f} {tz:.3f}  err={d.pose.object_error:.3g}")
        R = d.pose.R                         # 3x3 중첩 리스트/배열
```

내부 파라미터는 나중에 갱신할 수 있다.

```python
detector.set_camera(at.CameraParams(fx, fy, cx, cy))
detector.set_tag_size(0.162)
```

## 코너 접근 (NumPy / OpenCV)

```python
import numpy as np, cv2

for d in detector.detect(frame):
    pts = np.array(d.corners, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(frame, str(d.id), tuple(np.int32(d.center)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
```

## 실시간 카메라 루프

```python
import cv2, aidall_apriltag as at

cap = cv2.VideoCapture(0)
detector = at.Detector(family="tag36h11")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    for d in detector.detect(frame):
        pts = np.array(d.corners, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    cv2.imshow("apriltag", frame)
    if cv2.waitKey(1) == 27:        # Esc
        break
```

파일 기반 예제는 [`python/examples/detect.py`](../python/examples/detect.py)를 참고.
