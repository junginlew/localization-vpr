import numpy as np

from vpr.geometry import align_se3, align_sim3, rotation_error_deg, translation_error

# (위치 m, 회전 deg) 성공 임계: 4Seasons 공식 프로토콜의 high/medium/coarse precision
SUCCESS_THRESHOLDS = ((0.1, 1.0), (0.25, 2.0), (1.0, 5.0))


def _centers(poses):
    """포즈 리스트에서 카메라 중심 (N, 3)만 뽑는다."""
    return np.array([np.asarray(T)[:3, 3] for T in poses], dtype=np.float64)


def align_map_to_gt(poses_est, poses_gt):
    """추정·GT 카메라 중심쌍으로 SE3(Kabsch, s=1) 정렬. 반환 T_align: 지도 좌표 → GT 좌표.

    스케일을 보정하지 않으므로(s=1) 지도 스케일 오차가 측위 오차에 그대로 드러남.
    """
    return align_se3(_centers(poses_est), _centers(poses_gt))


def scale_error(poses_est, poses_gt):
    """Sim3(Umeyama) 정렬 스케일 s의 |s-1| - 메트릭 스케일 진단. 작을수록 미터 스케일이 맞음."""
    s, _ = align_sim3(_centers(poses_est), _centers(poses_gt))
    return abs(s - 1.0)


def pose_errors(poses_est, poses_gt, T_align):
    """정렬(T_align)을 적용한 추정 포즈와 GT의 쿼리별 (위치오차 m, 회전오차 deg) (N, 2)."""
    errors = []
    for T_est, T_gt in zip(poses_est, poses_gt):
        T_aligned = T_align @ np.asarray(T_est)
        errors.append((translation_error(T_aligned, T_gt), rotation_error_deg(T_aligned, T_gt)))
    return np.array(errors, dtype=np.float64).reshape(-1, 2)


def success_rates(errors, thresholds=SUCCESS_THRESHOLDS):
    """쿼리별 (위치, 회전) 오차에서 임계별 성공률을 낸다. 측위 실패는 inf 행으로 넣어 분모에 포함.

    errors: (N, 2). 반환 {(위치임계, 회전임계): 성공 비율}.
    """
    errors = np.asarray(errors, dtype=np.float64).reshape(-1, 2)
    return {
        thr: float(np.mean((errors[:, 0] <= thr[0]) & (errors[:, 1] <= thr[1])))
        for thr in thresholds
    }


def median_errors(errors):
    """쿼리별 오차의 (위치 중앙값, 회전 중앙값). 실패(inf)가 절반 넘으면 inf."""
    errors = np.asarray(errors, dtype=np.float64).reshape(-1, 2)
    return float(np.median(errors[:, 0])), float(np.median(errors[:, 1]))


def recall_at_k(hits, ks=(1, 5, 10)):
    """검색 평가. hits: 쿼리별 Top-K 후보가 정답 장소인지의 불리언 시퀀스(유사도 내림차순).

    반환 {k: 상위 k 후보 안에 정답이 하나라도 있는 쿼리 비율}.
    """
    return {k: float(np.mean([bool(np.any(np.asarray(h)[:k])) for h in hits])) for k in ks}
