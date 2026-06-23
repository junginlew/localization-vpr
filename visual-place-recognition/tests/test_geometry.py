import numpy as np
import pytest

from vpr import geometry as geo


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return geo.quat_to_rotmat(*q)


def test_quat_identity():
    assert np.allclose(geo.quat_to_rotmat(0, 0, 0, 1), np.eye(3))


def test_quat_z_90deg():
    half = np.pi / 4
    R = geo.quat_to_rotmat(0, 0, np.sin(half), np.cos(half))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_invert_se3():
    rng = np.random.default_rng(0)
    T = geo.make_se3(random_rotation(rng), rng.normal(size=3))
    assert np.allclose(T @ geo.invert_se3(T), np.eye(4), atol=1e-12)


def test_align_se3_recovers_rigid_transform():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    T_true = geo.make_se3(random_rotation(rng), rng.normal(size=3))
    dst = geo.apply_se3(T_true, src)
    T = geo.align_se3(src, dst)
    assert np.allclose(T, T_true, atol=1e-9)
    assert np.allclose(geo.apply_se3(T, src), dst, atol=1e-9)


def test_align_se3_does_not_absorb_scale():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(50, 3))
    dst = 1.05 * src
    T = geo.align_se3(src, dst)
    residual = np.linalg.norm(geo.apply_se3(T, src) - dst, axis=1).max()
    assert residual > 1e-3


def test_align_sim3_recovers_scale():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(50, 3))
    T_true = geo.make_se3(random_rotation(rng), rng.normal(size=3))
    dst = geo.apply_sim3(1.05, T_true, src)
    s, T = geo.align_sim3(src, dst)
    assert abs(s - 1.05) < 1e-9
    assert np.allclose(geo.apply_sim3(s, T, src), dst, atol=1e-9)


def test_align_sim3_scale_is_one_for_rigid():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(50, 3))
    T_true = geo.make_se3(random_rotation(rng), rng.normal(size=3))
    s, _ = geo.align_sim3(src, geo.apply_se3(T_true, src))
    assert abs(s - 1.0) < 1e-9


def test_pose_errors():
    T_gt = np.eye(4)
    half = np.deg2rad(5) / 2
    R = geo.quat_to_rotmat(0, 0, np.sin(half), np.cos(half))
    T_est = geo.make_se3(R, [0.3, 0.4, 0.0])
    assert abs(geo.translation_error(T_est, T_gt) - 0.5) < 1e-12
    assert abs(geo.rotation_error_deg(T_est, T_gt) - 5.0) < 1e-9


def test_align_rejects_bad_input():
    with pytest.raises(ValueError):
        geo.align_se3(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        geo.align_sim3(np.zeros((4, 2)), np.zeros((4, 2)))
