"""전체 파이프라인 평가 진입점.

코스마다: 지도 생성(build_map) → 지도 좌표계를 GT(ECEF)에 SE3 정렬 → 쿼리 측위(localize_query)
→ 측위 오차/성공률/중앙값, 검색 Recall@K, 스케일 진단을 출력한다.

실행: python scripts/evaluate.py [configs/default.yaml]
"""
import os
# COLMAP(pycolmap) 매처와 torch가 한 프로세스에 공존하면 OpenMP 런타임 충돌로 매칭 중 segfault가 난다.
# 모델 추론은 GPU에서 돌아 CPU BLAS 스레드 제한의 영향이 거의 없으므로 단일 스레드로 고정한다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vpr.datasets.four_seasons import FourSeasonsSequence
from vpr.geometry import rotation_error_deg
from vpr.localize import build_index, localize_query
from vpr.map_db import stack_global_descs
from vpr.mapping import build_map
from vpr.metrics import (align_map_to_gt, median_errors, recall_at_k,
                         scale_error, success_rates)
from vpr.models import GlobalExtractor, LocalExtractor, LocalMatcher


def _center(T):
    return np.asarray(T)[:3, 3]


def evaluate_course(course, cfg, data_root, work_root, models):
    global_extractor, local_extractor, matcher = models
    m, lz, ev = cfg["mapping"], cfg["localize"], cfg["eval"]

    map_seq = FourSeasonsSequence(data_root, course, "map")
    query_seq = FourSeasonsSequence(data_root, course, "query")
    calib = map_seq.calib
    image_size = (calib.width, calib.height)

    # --- 지도 생성 (프레임 서브샘플링 후 SfM) ---
    map_frames = map_seq.frames()[:: m["sfm_frame_stride"]]
    work = work_root / course
    keyframes = build_map(
        map_frames, calib, work, work / "map.h5", global_extractor, local_extractor,
        init_min_tri_angle=m["init_min_tri_angle"],
        min_keyframe_translation=m["min_keyframe_translation"],
        min_keyframe_rotation_deg=m["min_keyframe_rotation_deg"],
        min_stereo_depth=m["min_stereo_depth"], max_stereo_depth=m["max_stereo_depth"],
    )
    index = build_index(stack_global_descs(keyframes))

    # --- 지도 좌표계 → GT(ECEF) SE3 정렬 + 스케일 진단 ---
    map_gt = map_seq.gt_poses_ecef()                 # {timestamp: T_ecef_cam0}
    est = [kf.pose for kf in keyframes]              # SfM 지도 좌표계 포즈
    gt = [map_gt[kf.kf_id] for kf in keyframes]      # 같은 키프레임의 GT (ECEF)
    T_align = align_map_to_gt(est, gt)              # SfM 지도 좌표계 → ECEF
    s_err = scale_error(est, gt)                    # |s-1| 스케일 진단

    # --- 쿼리 측위 ---
    query_gt = query_seq.gt_poses_ecef()
    kf_centers = np.array([_center(map_gt[kf.kf_id]) for kf in keyframes])  # 후보 hit 판정용

    errors, hits = [], []
    for q in query_seq.frames()[:: ev["query_stride"]]:
        res = localize_query(
            q.cam0_path, keyframes, index, calib.K, image_size,
            global_extractor, local_extractor, matcher,
            top_k=lz["top_k"], min_essential_inliers=lz["min_essential_inliers"],
            min_pnp_points=lz["min_pnp_points"],
        )
        q_center = _center(query_gt[q.timestamp])

        if res.pose is None:                         # 측위 실패 → 분모에 inf로 포함
            errors.append((np.inf, np.inf))
        else:
            T_est = T_align @ res.pose               # 추정 포즈를 ECEF로 올려 쿼리 GT와 비교
            errors.append((
                float(np.linalg.norm(_center(T_est) - q_center)),
                rotation_error_deg(T_est, query_gt[q.timestamp]),
            ))
        # Recall: 검색 후보가 정답 장소(GT 거리 이내)인지 (유사도 순)
        dists = np.linalg.norm(kf_centers[res.candidate_ids] - q_center, axis=1)
        hits.append(dists <= ev["recall_gt_dist"])

    errors = np.array(errors, dtype=np.float64)
    thresholds = [tuple(t) for t in ev["success_thresholds"]]
    return {
        "scale_err": s_err,
        "success": success_rates(errors, thresholds),
        "median": median_errors(errors),
        "recall": recall_at_k(hits, tuple(ev["recall_ks"])),
        "num_keyframes": len(keyframes),
        "num_queries": len(errors),
    }


def _print_report(course, r, scale_gate):
    print(f"\n=== {course} (keyframes={r['num_keyframes']}, queries={r['num_queries']}) ===")
    gate = "OK" if r["scale_err"] < scale_gate else "WARN(스케일 재검토 필요)"
    print(f"  scale |s-1| = {r['scale_err']:.4f}  [{gate}]")
    print(f"  median error = {r['median'][0]:.3f} m / {r['median'][1]:.3f} deg")
    for (pos, rot), rate in r["success"].items():
        print(f"  success @ ({pos} m, {rot} deg) = {rate:.3f}")
    for k, rate in r["recall"].items():
        print(f"  Recall@{k} = {rate:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["data_root"])
    work_root = Path("runs")

    models = (GlobalExtractor(), LocalExtractor(), LocalMatcher(min_conf=cfg["localize"]["match_min_conf"]))
    for course in cfg["courses"]:
        r = evaluate_course(course, cfg, data_root, work_root, models)
        _print_report(course, r, cfg["eval"]["scale_gate"])


if __name__ == "__main__":
    main()
