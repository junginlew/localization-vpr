import numpy as np

from vpr.map_db import Keyframe, load_map, save_map


def make_kf(kf_id, rng, n_pts=20):
    points3d = rng.normal(size=(n_pts, 3))
    points3d[::5] = np.nan  # 삼각측량 실패 키포인트
    return Keyframe(
        kf_id=kf_id,
        pose=rng.normal(size=(4, 4)),
        global_desc=rng.normal(size=512).astype(np.float32),
        keypoints=rng.normal(size=(n_pts, 2)).astype(np.float32),
        local_desc=rng.normal(size=(n_pts, 64)).astype(np.float32),
        points3d=points3d,
    )


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    kfs = [make_kf(i, rng) for i in range(5)]
    path = tmp_path / "map.h5"
    save_map(path, kfs)
    loaded = load_map(path)
    assert len(loaded) == 5
    for a, b in zip(kfs, loaded):
        assert a.kf_id == b.kf_id
        np.testing.assert_array_equal(a.pose, b.pose)
        np.testing.assert_array_equal(a.global_desc, b.global_desc)
        np.testing.assert_array_equal(a.keypoints, b.keypoints)
        np.testing.assert_array_equal(a.local_desc, b.local_desc)
        np.testing.assert_array_equal(a.points3d, b.points3d)


def test_load_sorted_by_int_id(tmp_path):
    rng = np.random.default_rng(1)
    kfs = [make_kf(i, rng) for i in (3, 11, 2)]
    path = tmp_path / "map.h5"
    save_map(path, kfs)
    assert [kf.kf_id for kf in load_map(path)] == [2, 3, 11]