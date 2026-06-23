from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class Keyframe:
    kf_id: int
    pose: np.ndarray         # (4, 4) T_world_cam
    global_desc: np.ndarray  # (D,) L2 정규화된 전역 descriptor
    keypoints: np.ndarray    # (N, 2) 픽셀 좌표
    local_desc: np.ndarray   # (N, Dd) 로컬 descriptor
    points3d: np.ndarray     # (N, 3) 월드 좌표, 삼각측량 실패 키포인트는 NaN


def save_map(path, keyframes):
    with h5py.File(path, "w") as f:
        for kf in keyframes:
            n = len(kf.keypoints)
            if len(kf.local_desc) != n or len(kf.points3d) != n:
                raise ValueError(
                    f"keyframe {kf.kf_id}: keypoints/local_desc/points3d 개수 불일치 "
                    f"({n}, {len(kf.local_desc)}, {len(kf.points3d)})"
                )
            g = f.create_group(f"keyframes/{kf.kf_id}")
            g.create_dataset("pose", data=np.asarray(kf.pose, dtype=np.float64))
            g.create_dataset("global_desc", data=np.asarray(kf.global_desc, dtype=np.float32))
            g.create_dataset("keypoints", data=np.asarray(kf.keypoints, dtype=np.float32))
            g.create_dataset("local_desc", data=np.asarray(kf.local_desc, dtype=np.float32))
            g.create_dataset("points3d", data=np.asarray(kf.points3d, dtype=np.float64))


def load_map(path):
    keyframes = []
    with h5py.File(path, "r") as f:
        for key in f["keyframes"]:
            g = f["keyframes"][key]
            keyframes.append(Keyframe(
                kf_id=int(key),
                pose=g["pose"][()],
                global_desc=g["global_desc"][()],
                keypoints=g["keypoints"][()],
                local_desc=g["local_desc"][()],
                points3d=g["points3d"][()],
            ))
    keyframes.sort(key=lambda kf: kf.kf_id)
    return keyframes


def stack_global_descs(keyframes):
    return np.stack([kf.global_desc for kf in keyframes])
