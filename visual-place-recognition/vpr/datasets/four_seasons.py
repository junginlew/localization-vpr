from pathlib import Path

import numpy as np

from vpr.datasets.base import Calibration, Frame
from vpr.geometry import make_se3, quat_to_rotmat


RECORDINGS = {
    ("office_loop", "map"): "recording_2020-04-07_10-20-32",      # 봄
    ("office_loop", "query"): "recording_2021-01-07_12-04-03",    # 겨울
    ("parking_garage", "map"): "recording_2020-12-22_12-04-35",   # 겨울
    ("parking_garage", "query"): "recording_2021-05-10_19-15-19", # 봄
}


def _pose_from_values(vals, scale):
    t = np.array([float(v) for v in vals[0:3]])
    qx, qy, qz, qw = (float(v) for v in vals[3:7])
    return make_se3(quat_to_rotmat(qx, qy, qz, qw), scale * t)


def load_gnss_poses(path):
    """GNSSPoses.txt를 {timestamp_ns: (4,4) T_world_cam0 pose}로 파싱한다."""
    poses = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        vals = line.split(",")
        poses[int(vals[0])] = _pose_from_values(vals[1:8], scale=float(vals[8]))
    return poses


def load_transformations(path):
    """Transformations.txt를 {이름: (4,4) 변환}으로 파싱한다."""
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    result = {}
    for header, value in zip(lines[::2], lines[1::2]):
        name = header.lstrip("#").split(":")[0].strip()
        vals = value.split(",")
        if len(vals) == 1:
            result[name] = float(vals[0])
        else:
            result[name] = _pose_from_values(vals, scale=1.0)
    return result


def load_calibration(calib_dir):
    """undistorted 캘리브레이션 파싱. rectified라 좌우 동일- calib_0만 읽음."""
    calib_dir = Path(calib_dir)
    tokens = (calib_dir / "undistorted_calib_0.txt").read_text().split()
    fx, fy, cx, cy = (float(t) for t in tokens[1:5])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    T_stereo = np.loadtxt(calib_dir / "undistorted_calib_stereo.txt")
    return Calibration(
        K=K,
        baseline=float(abs(T_stereo[0, 3])),
        width=int(tokens[9]),
        height=int(tokens[10]),
    )


class FourSeasonsSequence:
    """4Seasons 레코딩 하나. frames()는 GT 포즈가 있는 키프레임만 반환한다."""

    def __init__(self, data_root, course, role):
        data_root = Path(data_root)
        self.recording = RECORDINGS[(course, role)]
        self.root = data_root / self.recording
        self.calib = load_calibration(data_root / "calibration")
        self.transforms = load_transformations(self.root / "Transformations.txt")
        self._poses = load_gnss_poses(self.root / "GNSSPoses.txt")

    def frames(self):
        cam0 = self.root / "undistorted_images" / "cam0"
        cam1 = self.root / "undistorted_images" / "cam1"
        out = []
        for ts in sorted(self._poses):
            p0 = cam0 / f"{ts}.png"
            if not p0.exists():  # GT는 있는데 이미지가 없는 키프레임은 제외 (손상된 데이터 방어용)
                continue
            out.append(Frame(timestamp=ts, cam0_path=p0, cam1_path=cam1 / f"{ts}.png", pose=self._poses[ts]))
        return out
