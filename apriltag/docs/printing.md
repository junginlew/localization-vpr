# 태그 생성과 인쇄

## 세 가지 출력 형식

| 형식 | 용도 | 특징 |
|------|------|------|
| **PNG** | 화면 표시, 검출 왕복 테스트 | 래스터. 크기는 셀당 픽셀로 결정 |
| **SVG** | 단일 태그 인쇄 | 벡터. 정확한 mm, 무한 확대 |
| **PDF 시트** | 여러 태그 일괄 인쇄 | 한 페이지에 배치, quiet zone·ID 라벨·컷 마크 |

## 주의

### Quiet zone (필수)

태그 둘레의 **흰 여백**이다. 없으면 검출이 안 된다.

모든 생성 함수가 `quiet_zone`(셀 단위)을 받으며, 인쇄용은 최소 1–2셀을 권장한다.

### 확대는 nearest-neighbor

셀을 픽셀 블록으로 확대할 때 쌍선형 등의 보간 방법을 쓰면 안된다(엣지가 흐려져 검출, 인쇄 품질 저하). 

### 물리 크기와 `tag_size`

`apriltag_to_image`가 주는 격자는 전체 너비(`total_width`) 셀이다.

포즈의 `tag_size`가 가리키는 **검은 테두리**는 그 안쪽 `border_width` 셀이다. SVG/PDF의 `tag_size_mm`는 **검은 테두리 변**을 그 크기로 맞춘다.

```python
import aidall_apriltag as at
at.border_size_to_total_mm("tag36h11", 50.0)  # 검은테두리 50mm일 때 전체 격자 인쇄 폭
```

## Python API

### 배열 / PNG

```python
import aidall_apriltag as at

# 네이티브 셀 격자 (HxW uint8, 0/255)
grid = at.generate("tag36h11", 0)

# PNG로 저장 (Pillow는 기본 의존성으로 함께 설치됨)
at.save_png("tag0.png", "tag36h11", 0, pixels_per_cell=20, quiet_zone=1)

# 직접 가공하고 싶다면
big = at.scale(at.add_quiet_zone(grid, cells=1), pixels_per_cell=20)
```

### SVG (단일 태그, 정확한 mm)

```python
at.save_svg("tag0.svg", "tag36h11", 0,
            tag_size_mm=80.0,    # 검은 테두리 변 = 80mm
            quiet_zone_cells=1,
            label=True)          # 하단에 "tag36h11 id=0" 표기
```

### PDF 시트 (일괄 인쇄)

```python
# reportlab 필요: pip install ".[print]"
at.save_pdf_sheet(
    "sheet.pdf", "tag36h11", ids=range(0, 24),
    tag_size_mm=50.0,
    quiet_zone_cells=2,
    page="A4",            # 또는 "letter"
    label=True,
    cut_marks=True,       # 모서리 컷 가이드
)
```

## CLI

설치 시 `aidall-apriltag` CLI 명령으로 태그 이미지 생성이 가능하다.

```shell
# 단일 태그 — 확장자로 형식 결정 (.png / .svg)
aidall-apriltag generate --family tag36h11 --id 0 --out tag0.png --pixels-per-cell 20
aidall-apriltag generate --family tag36h11 --id 0 --out tag0.svg --tag-size-mm 80

# 여러 태그 PDF 시트
aidall-apriltag sheet --family tag36h11 --ids 0-23 --out sheet.pdf --tag-size-mm 50
aidall-apriltag sheet --family tag36h11 --ids 0,2,5,10-12 --out sel.pdf --page letter
```

`--ids`는 `0-23`, `0,2,5`, `0-3,10,12-14` 같은 범위, 목록 혼합을 지원한다.

## 인쇄 체크리스트

- **무광 용지**에 인쇄할 것 (광택은 반사광으로 검출 방해).
- 프린터 "실제 크기 / 100%"로 출력한다. "맞춤(fit)"은 크기를 바꿔 `tag_size`가 어긋난다. 인쇄 후 자로 검은 테두리를 재서 `tag_size_mm`과 일치하는지 확인.
- 단단한 평면(폼보드 등)에 부착할 것. 휘면 포즈가 틀어진다.
- quiet zone(흰 여백)을 자르지 말 것.