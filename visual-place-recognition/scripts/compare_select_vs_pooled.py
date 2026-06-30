"""옛(select_candidate) vs 새(pooled) 측위를 같은 지도 위에서 프레임별로 비교.

질문: "이전에 크게 튄(>5m) 프레임이 pooled로 고쳐졌나?"
방법: 4차에서 빌드된 runs/office_loop/map.h5를 고정 → 쿼리마다 같은 매칭을 두 방식에 먹여
      err_old(select_candidate), err_pool(pooled)을 GT 대비로 계산 → 프레임별 출력.
옛 verify_essential/select_candidate는 코드에서 삭제됐으므로 git HEAD~1 원본을 그대로 인라인.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path
import numpy as np
import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vpr.datasets.four_seasons import FourSeasonsSequence
from vpr.localize import build_index, estimate_pose, search_topk
from vpr.map_db import load_map, stack_global_descs
from vpr.metrics import align_map_to_gt, scale_error
from vpr.models import GlobalExtractor, LocalExtractor, LocalMatcher

cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
lz, ev = cfg["localize"], cfg["eval"]
TOPK = lz["top_k"]; MIN_PNP = lz["min_pnp_points"]; GT_DIST = ev["recall_gt_dist"]
MIN_ESSENTIAL_INLIERS = 30   # 옛 기본값 (HEAD~1)
course = "office_loop"


# ---- 옛 코드 그대로 (git HEAD~1:vpr/localize.py) ----
def verify_essential(query_pts, kf_pts, K, min_inliers=MIN_ESSENTIAL_INLIERS, ransac_thresh=1.0):
    query_pts = np.ascontiguousarray(query_pts, dtype=np.float64)
    kf_pts = np.ascontiguousarray(kf_pts, dtype=np.float64)
    if len(query_pts) < min_inliers:
        return None
    E, mask = cv2.findEssentialMat(
        query_pts, kf_pts, np.asarray(K, dtype=np.float64),
        method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh,
    )
    if E is None or mask is None:
        return None
    inliers = np.flatnonzero(mask.ravel())
    if len(inliers) < min_inliers:
        return None
    return inliers


def select_candidate(candidate_matches, K, min_inliers=MIN_ESSENTIAL_INLIERS):
    best_idx, best_count = None, 0
    for i, (query_pts, kf_pts) in enumerate(candidate_matches):
        inliers = verify_essential(query_pts, kf_pts, K, min_inliers)
        if inliers is not None and len(inliers) > best_count:
            best_idx, best_count = i, len(inliers)
    return best_idx
# ----------------------------------------------------

map_seq = FourSeasonsSequence(ROOT / cfg["data_root"], course, "map")
query_seq = FourSeasonsSequence(ROOT / cfg["data_root"], course, "query")
calib = map_seq.calib
image_size = (calib.width, calib.height); K = calib.K

keyframes = load_map(ROOT / "runs" / course / "map.h5")
index = build_index(stack_global_descs(keyframes))
map_gt = map_seq.gt_poses_ecef()
T_align = align_map_to_gt([kf.pose for kf in keyframes], [map_gt[kf.kf_id] for kf in keyframes])
print(f"[load] kf={len(keyframes)} scale|s-1|={scale_error([kf.pose for kf in keyframes],[map_gt[kf.kf_id] for kf in keyframes]):.4f}", flush=True)

query_gt = query_seq.gt_poses_ecef()
ge, le, mt = GlobalExtractor(), LocalExtractor(), LocalMatcher(min_conf=lz["match_min_conf"])


def center(T): return np.asarray(T)[:3, 3]
def perr(T_map, qc):
    return np.inf if T_map is None else float(np.linalg.norm(center(T_align @ T_map) - qc))


rows = []  # (qi, err_old, err_pool)
queries = query_seq.frames()[:: ev["query_stride"]]
for qi, q in enumerate(queries):
    qc = center(query_gt[q.timestamp])
    qg = ge(q.cam0_path); qkp, qdesc = le(q.cam0_path)
    cand_ids, _ = search_topk(index, qg, TOPK)
    cand_ids = cand_ids[cand_ids >= 0]

    cand_matches, matched, pool3d, pool2d = [], [], [], []
    for kf_i in cand_ids:
        kf = keyframes[kf_i]
        idx = mt(qkp, qdesc, kf.keypoints, kf.local_desc, image_size)
        cand_matches.append((qkp[idx[:, 0]], kf.keypoints[idx[:, 1]]))
        matched.append((kf_i, idx))
        p3 = kf.points3d[idx[:, 1]]; valid = ~np.isnan(p3).any(axis=1)
        pool3d.append(p3[valid]); pool2d.append(qkp[idx[:, 0]][valid])

    # --- 옛 방식: select_candidate → 그 후보만 PnP ---
    T_old = None
    if cand_matches:
        best = select_candidate(cand_matches, K, MIN_ESSENTIAL_INLIERS)
        if best is not None:
            kf_i, idx = matched[best]; kf = keyframes[kf_i]
            p3 = kf.points3d[idx[:, 1]]; p2 = qkp[idx[:, 0]]; v = ~np.isnan(p3).any(axis=1)
            T_old, _ = estimate_pose(p3[v], p2[v], K, min_points=MIN_PNP)
    err_old = perr(T_old, qc)

    # --- 새 방식: pooled ---
    T_pool = None
    if pool3d:
        P3, P2 = np.concatenate(pool3d), np.concatenate(pool2d)
        T_pool, _ = estimate_pose(P3, P2, K, min_points=MIN_PNP)
    err_pool = perr(T_pool, qc)

    rows.append((qi, err_old, err_pool))
    if qi % 25 == 0:
        print(f"  [{qi}/{len(queries)}] old={err_old:.1f} pool={err_pool:.1f}", flush=True)

rows = np.array(rows, dtype=np.float64)
qi_, old, pool = rows[:, 0], rows[:, 1], rows[:, 2]


def summ(name, c):
    return (f"{name}: median={np.median(c):.3f}m  <=1m={np.mean(c<=1).item():.3f}  "
            f"<=5m={np.mean(c<=5).item():.3f}  >5m={np.mean(c>5).item():.3f}  fail={np.mean(~np.isfinite(c)).item():.3f}  n={len(c)}")


print("\n==== office_loop 옛(select_candidate) vs 새(pooled) — 같은 4차 지도 ====", flush=True)
print(summ("old(select)", old))
print(summ("pooled     ", pool))

# 옛 방식이 크게 튄(>5m 또는 실패) 프레임에서 pooled가 어떻게 됐나
bad = (old > 5.0) | ~np.isfinite(old)
print(f"\n옛 방식 크게 튐(>5m 또는 실패): {int(bad.sum())} / {len(old)} 프레임")
if bad.sum():
    fixed = bad & (pool <= 5.0) & np.isfinite(pool)
    still = bad & ((pool > 5.0) | ~np.isfinite(pool))
    print(f"  → pooled가 ≤5m로 교정: {int(fixed.sum())}")
    print(f"  → pooled도 여전히 >5m/실패: {int(still.sum())}")
    print("  프레임별 (qi, old→pool):")
    for i in np.flatnonzero(bad):
        print(f"    q{int(qi_[i]):3d}: old={old[i]:8.2f}  pool={pool[i]:8.2f}  {'FIXED' if (pool[i]<=5 and np.isfinite(pool[i])) else 'still-bad'}")

# 반대로 pooled가 새로 망친 경우(옛≤5m인데 pooled>5m)
regress = (old <= 5.0) & np.isfinite(old) & ((pool > 5.0) | ~np.isfinite(pool))
print(f"\n옛≤5m인데 pooled가 >5m/실패로 악화시킨 프레임: {int(regress.sum())}")
for i in np.flatnonzero(regress):
    print(f"    q{int(qi_[i]):3d}: old={old[i]:8.2f}  pool={pool[i]:8.2f}")
print("[done]", flush=True)
