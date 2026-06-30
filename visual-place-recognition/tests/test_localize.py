import numpy as np

from vpr.geometry import (apply_se3, invert_se3, make_se3, quat_to_rotmat,
                          rotation_error_deg, translation_error)
from vpr.localize import build_index, estimate_pose, search_topk


def normalized(rng, m, d):
    x = rng.normal(size=(m, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])


def project(T_world_cam, points3d):
    """월드점을 T_world_cam 카메라로 투영해 픽셀 좌표 (N, 2)를 만든다."""
    T_cam_world = invert_se3(T_world_cam)
    cam = points3d @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def test_topk_ordered_by_similarity():
    rng = np.random.default_rng(1)
    descs = normalized(rng, 50, 16)
    index = build_index(descs)
    idx, scores = search_topk(index, descs[3], k=10)
    assert idx[0] == 3                       # 자기 자신이 1등
    assert abs(scores[0] - 1.0) < 1e-5       # 정규화 벡터의 self 내적 = 코사인 1.0
    assert np.all(np.diff(scores) <= 1e-6)   # 점수가 내림차순


def test_nearby_descriptor_ranks_high():
    rng = np.random.default_rng(2)
    descs = normalized(rng, 30, 16)
    index = build_index(descs)
    # 5번 키프레임에 약간의 노이즈를 더한 쿼리 → 5번이 최상위여야함
    q = descs[5] + 0.01 * rng.normal(size=16).astype(np.float32)
    q /= np.linalg.norm(q)
    idx, _ = search_topk(index, q, k=3)
    assert idx[0] == 5


def random_pose(rng):
    q = rng.normal(size=4)
    R = quat_to_rotmat(*(q / np.linalg.norm(q)))
    return make_se3(R, rng.normal(size=3))


def points_in_front(rng, T_world_cam, n):
    """카메라 앞쪽(z>0)에 점을 만들고 월드 좌표로 변환해 반환한다."""
    cam = np.column_stack([
        rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), rng.uniform(4, 12, n),
    ])
    return apply_se3(T_world_cam, cam)


def test_pnp_recovers_known_pose():
    rng = np.random.default_rng(10)
    T_true = random_pose(rng)
    points3d = points_in_front(rng, T_true, 50)
    points2d = project(T_true, points3d)
    T_est, inliers, _ = estimate_pose(points3d, points2d, K)
    assert T_est is not None
    assert translation_error(T_est, T_true) < 1e-6
    assert rotation_error_deg(T_est, T_true) < 1e-4


def test_pnp_rejects_outliers():
    rng = np.random.default_rng(11)
    T_true = random_pose(rng)
    points3d = points_in_front(rng, T_true, 60)
    points2d = project(T_true, points3d)
    points2d[:12] += rng.uniform(-80, 80, size=(12, 2))  # 12개를 가짜 대응으로 오염
    T_est, inliers, _ = estimate_pose(points3d, points2d, K)
    assert T_est is not None
    assert translation_error(T_est, T_true) < 1e-3        # 인라이어만으로 정확 복원
    assert set(inliers).isdisjoint(range(12))             # 오염된 12개는 인라이어에서 제외
