import numpy as np


def quat_to_rotmat(qx, qy, qz, qw):
    """쿼터니언(qx, qy, qz, qw)을 3x3 회전행렬로 변환한다. 입력은 내부에서 정규화된다."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("zero quaternion")
    qx, qy, qz, qw = q / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def make_se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def invert_se3(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_se3(R.T, -R.T @ t)


def apply_se3(T, points):
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def _check_correspondences(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3 or src.shape != dst.shape:
        raise ValueError(f"expected matching (N,3) arrays, got {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError("need at least 3 correspondences")
    return src, dst


def align_se3(src, dst):
    """dst ≈ R @ src + t 를 푸는 강체 정렬(Kabsch). 스케일을 보정하지 않음. 평가용 정렬은 이것만 사용."""
    src, dst = _check_correspondences(src, dst)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)  # 센터링된 src와 dst의 방향 분포 대응을 요약
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[2, 2] = -1.0
    R = Vt.T @ S @ U.T  # src를 dst에 겹치게 만드는 3x3 회전행렬 (회전축+회전량, S는 거울상 교정)
    return make_se3(R, mu_d - R @ mu_s)  # 4x4 SE3 변환 행렬 (회전 R + 이동 t)


def align_sim3(src, dst):
    """dst ≈ s * R @ src + t 를 푸는 Umeyama 정렬, (s, T) 반환. 스케일 진단 전용. 평가 정렬에 사용 안함."""
    src, dst = _check_correspondences(src, dst)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    cov = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_src)
    return s, make_se3(R, mu_d - s * R @ mu_s) # 배율, 변환행렬


def apply_sim3(s, T, points):
    points = np.asarray(points, dtype=np.float64)
    return s * (points @ T[:3, :3].T) + T[:3, 3]


def translation_error(T_est, T_gt):
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def rotation_error_deg(T_est, T_gt):
    R_rel = T_est[:3, :3].T @ T_gt[:3, :3]
    cos = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
