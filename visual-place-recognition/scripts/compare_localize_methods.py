"""4차 office 지도 고정 → 같은 매칭으로 네 가지 측위 방식을 프레임별로 비교.

목적: re-ranking(현재 localize_query 방식)의 순수 효과 측정 — pool보다 나은가, best-single(천장)에
얼마나 근접하나, q0·q1을 회수하나. SfM 재빌드 없이 측위만(~몇 분).

비교하는 네 방식 (모두 같은 Top-K 후보·같은 로컬 매칭에서 출발):
- select_candidate (옛 방식): 후보마다 essential matrix(쿼리↔키프레임 2D-2D) 인라이어를 세어,
  가장 많은 후보 1개를 고르고 그 후보만 PnP. essential은 "장소가 맞나"의 2D-2D proxy일 뿐 포즈를
  직접 풀지 않는다. (코드에선 삭제됐으므로 git HEAD~1 원본 로직 그대로 인라인)
- pooled: Top-K 전부의 (3D, 2D) 대응을 한 풀에 합쳐 단일 PnP-RANSAC 1회. 후보를 안 고르는 대신
  틀린 후보의 대응이 섞여 다수결을 오염시킬 수 있다.
- rerank (현재 구현): 후보마다 따로 PnP를 돌린 뒤 PnP 품질로 리랭킹(inlier 수 많은 순 → 재투영 오차
  작은 순 → VPR score 높은 순)해 1등 후보의 포즈를 채택. = 현재 localize_query 동작.
- best-single (oracle 천장): 후보별 독립 PnP 결과 중 GT에 가장 가까운 포즈. GT를 컨닝하므로 실전엔
  못 쓰고, "후보를 완벽히 고르면 도달 가능한 상한"을 보는 기준선.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path
import numpy as np, cv2, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vpr.datasets.four_seasons import FourSeasonsSequence
from vpr.geometry import rotation_error_deg
from vpr.localize import build_index, estimate_pose, search_topk
from vpr.map_db import load_map, stack_global_descs
from vpr.metrics import (align_map_to_gt, median_errors, recall_at_k,
                         scale_error, success_rates)
from vpr.models import GlobalExtractor, LocalExtractor, LocalMatcher

cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
lz, ev = cfg["localize"], cfg["eval"]
TOPK = lz["top_k"]; MIN_PNP = lz["min_pnp_points"]; MIN_ESS = 30
course = "office_loop"


def ess_inliers(qp, kp, K, min_inliers=MIN_ESS, thr=1.0):
    """select_candidate가 쓰던 essential 인라이어 수. 매칭<30이거나 inlier<30이면 0(폐기)."""
    qp = np.ascontiguousarray(qp, np.float64); kp = np.ascontiguousarray(kp, np.float64)
    if len(qp) < min_inliers:
        return 0
    E, mask = cv2.findEssentialMat(qp, kp, np.asarray(K, np.float64), method=cv2.RANSAC, prob=0.999, threshold=thr)
    if E is None or mask is None:
        return 0
    n = int(np.count_nonzero(mask.ravel()))
    return n if n >= min_inliers else 0


map_seq = FourSeasonsSequence(ROOT / cfg["data_root"], course, "map")
query_seq = FourSeasonsSequence(ROOT / cfg["data_root"], course, "query")
calib = map_seq.calib; image_size = (calib.width, calib.height); K = calib.K
keyframes = load_map(ROOT / "runs" / course / "map.h5")
index = build_index(stack_global_descs(keyframes))
map_gt = map_seq.gt_poses_ecef()
est = [kf.pose for kf in keyframes]; gt = [map_gt[kf.kf_id] for kf in keyframes]
T_align = align_map_to_gt(est, gt)
print(f"[load] kf={len(keyframes)} scale|s-1|={scale_error(est, gt):.4f}", flush=True)
query_gt = query_seq.gt_poses_ecef()
ge, le, mt = GlobalExtractor(), LocalExtractor(), LocalMatcher(min_conf=lz["match_min_conf"])
center = lambda T: np.asarray(T)[:3, 3]
def perr(T, qc):
    return np.inf if T is None else float(np.linalg.norm(center(T_align @ T) - qc))
def rerr(T, ts):
    return np.inf if T is None else rotation_error_deg(T_align @ T, query_gt[ts])

GT_DIST = ev["recall_gt_dist"]; KS = tuple(ev["recall_ks"])
THR = [tuple(t) for t in ev["success_thresholds"]]
kf_centers = np.array([center(map_gt[kf.kf_id]) for kf in keyframes])
METHODS = ["old(select)", "pooled", "rerank(현재)", "best-single(천장)"]
errs = {m: [] for m in METHODS}   # 방식별 (위치오차, 회전오차) — evaluate.py와 동일
hits = []                          # Recall용 (검색 결과라 방식 무관)
perframe = []                      # (qi, old, pool, rerank, best) 위치오차 — 큰 오차 프레임 추적용

queries = query_seq.frames()[:: ev["query_stride"]]
for qi, q in enumerate(queries):
    ts = q.timestamp; qc = center(query_gt[ts])
    qg = ge(q.cam0_path); qkp, qdesc = le(q.cam0_path)
    cand_ids, cand_scores = search_topk(index, qg, TOPK)
    keep = cand_ids >= 0; cand_ids, cand_scores = cand_ids[keep], cand_scores[keep]

    cands, pool3d, pool2d = [], [], []
    for kf_i, vpr in zip(cand_ids, cand_scores):
        kf = keyframes[kf_i]
        idx = mt(qkp, qdesc, kf.keypoints, kf.local_desc, image_size)
        p3 = kf.points3d[idx[:, 1]]; valid = ~np.isnan(p3).any(axis=1)
        p3v, p2v = p3[valid], qkp[idx[:, 0]][valid]
        pool3d.append(p3v); pool2d.append(p2v)
        pose, inl, reproj = estimate_pose(p3v, p2v, K, min_points=MIN_PNP)
        ess = ess_inliers(qkp[idx[:, 0]], kf.keypoints[idx[:, 1]], K)
        cands.append({"pose": pose, "n_inl": 0 if inl is None else len(inl),
                      "reproj": np.inf if reproj is None else reproj,
                      "vpr": float(vpr), "ess": ess, "err": perr(pose, qc)})

    ok = [c for c in cands if c["pose"] is not None]
    ess_ok = [c for c in cands if c["ess"] > 0]
    Tp, _, _ = estimate_pose(np.concatenate(pool3d), np.concatenate(pool2d), K, min_points=MIN_PNP)
    T = {
        "old(select)": (max(ess_ok, key=lambda c: c["ess"])["pose"] if ess_ok else None),  # essential 최다 후보의 PnP
        "pooled": Tp,                                                                       # 전부 합쳐 단일 PnP
        "rerank(현재)": (max(ok, key=lambda c: (c["n_inl"], -c["reproj"], c["vpr"]))["pose"] if ok else None),
        "best-single(천장)": (min(ok, key=lambda c: c["err"])["pose"] if ok else None),       # GT 최근접(oracle)
    }
    for m in METHODS:
        errs[m].append((perr(T[m], qc), rerr(T[m], ts)))
    hits.append(np.linalg.norm(kf_centers[cand_ids] - qc, axis=1) <= GT_DIST)
    perframe.append((qi, perr(T["old(select)"], qc), perr(T["pooled"], qc),
                     perr(T["rerank(현재)"], qc), perr(T["best-single(천장)"], qc)))
    if qi % 25 == 0:
        print(f"  [{qi}/{len(queries)}]", flush=True)

print("\n==== office_loop 4차 지도 고정 — evaluate.py 지표 (방식별) ====", flush=True)
print(f"scale |s-1| = {scale_error(est, gt):.4f}  (지도·정렬 공통)")
rec = recall_at_k(hits, KS)
print("Recall  " + "  ".join(f"@{k}={r:.3f}" for k, r in rec.items()) + "   (검색 결과라 방식 공통)")
for m in METHODS:
    e = np.array(errs[m], float); med = median_errors(e); sr = success_rates(e, THR)
    pos = e[:, 0]   # 위치 오차(m)
    print(f"\n[{m}]")
    print(f"  median error = {med[0]:.3f} m / {med[1]:.3f} deg")
    print(f"  위치오차 분포: <=1m={np.mean(pos<=1):.3f}  >5m={np.mean(pos>5):.3f}  fail(inf)={np.mean(~np.isfinite(pos)):.3f}")
    for (p, r), v in sr.items():
        print(f"  success @ ({p} m, {r} deg) = {v:.3f}")

pf = np.array(perframe, float); qi_, old, pool, rerank, best = pf.T
bad = (old > 5) | ~np.isfinite(old)
print(f"\n옛 select_candidate가 >5m/실패였던 프레임: {int(bad.sum())}")
for i in np.flatnonzero(bad):
    print(f"  q{int(qi_[i]):3d}: old={old[i]:7.2f}  pool={pool[i]:7.2f}  rerank={rerank[i]:7.2f}  best={best[i]:7.2f}")
print("[done]", flush=True)
