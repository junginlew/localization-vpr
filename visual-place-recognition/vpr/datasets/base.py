from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Calibration:
    K: np.ndarray    # (3, 3) 핀홀 내부 파라미터 행렬
    baseline: float  # 좌우 카메라 광학 중심 사이 거리 (m)
    width: int
    height: int


@dataclass
class Frame:
    timestamp: int           # 나노초
    cam0_path: Path          # 왼쪽 이미지
    cam1_path: Path          # 오른쪽 이미지
    pose: np.ndarray | None  # (4, 4) GT 포즈 T_world_cam0, 미터 단위. 없으면 None
