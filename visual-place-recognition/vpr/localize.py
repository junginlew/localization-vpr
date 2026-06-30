from dataclasses import dataclass

import cv2
import faiss
import numpy as np

from vpr.geometry import make_se3

MIN_PNP_POINTS = 15            # PnP 입력 3D-2D 대응 하한 (이 미만이면 측위 실패)


@dataclass
class LocalizationResult:
    """localize_query의 결과. 측위 실패해도 검색 후보(candidate_ids)는 채워 recall 평가를 가능케 한다."""
    pose: np.ndarray | None              # T_world_cam (4,4). 측위 실패 시 None
    candidate_ids: np.ndarray            # 검색 Top-K 후보 키프레임 행 번호(유사도 순, faiss -1 패딩 제외)
    chosen_id: int | None = None         # 리랭킹 1등으로 선택돼 포즈를 채택한 키프레임 행 번호. 실패면 None
    num_pnp_points: int = 0              # 선택 후보의 PnP 입력 3D-2D 대응 수 (매칭 ∩ 삼각측량 성공)
    num_inliers: int = 0                # 선택 후보의 PnP RANSAC 인라이어 수 (실패 시 0)
    inlier_ratio: float = 0.0           # 선택 후보의 인라이어 비율 (n_inliers / n_points)
    reproj_err: float | None = None     # 선택 후보의 인라이어 평균 재투영 오차(px). 실패면 None


def build_index(global_descs):
    """L2 정규화된 전역 descriptor (M, D)로 코사인 검색 인덱스를 만든다."""
    descs = np.ascontiguousarray(global_descs, dtype=np.float32)
    index = faiss.IndexFlatIP(descs.shape[1])  # 내적으로 검색하는 인덱스 객체 (내적 계산은 search 시점). 정규화 벡터라 내적==코사인 유사도
    index.add(descs)
    return index


def search_topk(index, query_desc, k):
    """쿼리 전역 descriptor 하나로 유사한 키프레임 상위 k개의 (행 번호, 유사도 점수)를 반환한다.

    행 번호 i는 stack_global_descs에 넣은 키프레임 순서의 i번째에 대응.
    """
    q = np.ascontiguousarray(query_desc, dtype=np.float32).reshape(1, -1)
    scores, indices = index.search(q, k)
    return indices[0], scores[0]


def estimate_pose(points3d, points2d, K, reproj_err=4.0, min_points=MIN_PNP_POINTS):
    """3D 월드점과 쿼리 2D 대응에서 쿼리 카메라 포즈 T_world_cam을 PnP로 추정한다.

    반환: (T_world_cam (4,4), 인라이어 인덱스, 인라이어 평균 재투영 오차[px]).
    입력 점 또는 RANSAC 인라이어가 min_points 미만이거나 solvePnPRansac이 실패하면 (None, None, None).
    """
    points3d = np.ascontiguousarray(points3d, dtype=np.float64)
    points2d = np.ascontiguousarray(points2d, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if len(points3d) < min_points:
        return None, None, None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d, points2d, K, None,
        reprojectionError=reproj_err, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < min_points:
        return None, None, None

    inliers = inliers.ravel()
    # 인라이어 3D점을 추정 포즈로 다시 투영해 실제 2D와의 평균 픽셀 거리(재투영 오차)를 잰다 — 후보 리랭킹용.
    proj, _ = cv2.projectPoints(points3d[inliers], rvec, tvec, K, None)
    mean_reproj = float(np.linalg.norm(proj.reshape(-1, 2) - points2d[inliers], axis=1).mean())

    # solvePnP는 월드→카메라(T_cam_world)를 주므로 T_world_cam으로 역변환.
    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T
    t_wc = (-R_wc @ tvec).ravel()
    return make_se3(R_wc, t_wc), inliers, mean_reproj


def localize_query(query_image, keyframes, index, K, image_size,
                   global_extractor, local_extractor, matcher, top_k=10,
                   min_pnp_points=MIN_PNP_POINTS):
    """쿼리 이미지 한 장에서 6-DoF 포즈를 추정해 LocalizationResult로 반환한다.

    검색(Top-K) → 후보별 독립 PnP-RANSAC → PnP 품질로 리랭킹해 최선 후보의 포즈를 채택.
    후보를 섞지 않으므로(풀링 X) 틀린 후보가 결과를 오염시키지 않는다. 리랭킹 순위는
    inlier 수 많은 순 → 재투영 오차 작은 순 → VPR score 높은 순.
    측위에 실패해도 candidate_ids는 채워 recall 평가 가능.
    """
    query_global = global_extractor(query_image)
    query_kp, query_desc = local_extractor(query_image)

    cand_ids, cand_scores = search_topk(index, query_global, top_k)
    keep = cand_ids >= 0                  # faiss는 키프레임이 k보다 적으면 -1로 채움 → 제외
    cand_ids, cand_scores = cand_ids[keep], cand_scores[keep]

    candidates = []                       # PnP 성공 후보별 포즈·품질
    for kf_i, vpr_score in zip(cand_ids, cand_scores):
        kf = keyframes[kf_i]
        idx = matcher(query_kp, query_desc, kf.keypoints, kf.local_desc, image_size)
        p3 = kf.points3d[idx[:, 1]]              # 매칭된 키프레임 키포인트의 월드 3D
        valid = ~np.isnan(p3).any(axis=1)        # 삼각측량 실패(NaN) 키포인트 제외
        p3v, p2v = p3[valid], query_kp[idx[:, 0]][valid]
        pose, inliers, reproj = estimate_pose(p3v, p2v, K, min_points=min_pnp_points)
        if pose is None:                  # 이 후보 PnP 실패 → 버림
            continue
        candidates.append({
            "id": int(kf_i), "pose": pose, "n_points": int(len(p3v)),
            "n_inliers": int(len(inliers)), "inlier_ratio": len(inliers) / len(p3v),
            "reproj_err": float(reproj), "vpr_score": float(vpr_score),
        })

    if not candidates:      # 모든 후보 PnP 실패 → 측위 실패 (후보는 보존)
        return LocalizationResult(pose=None, candidate_ids=cand_ids)

    # 리랭킹: inlier 수 많은 순 → 재투영 오차 작은 순 → VPR score 높은 순
    best = max(candidates, key=lambda c: (c["n_inliers"], -c["reproj_err"], c["vpr_score"]))
    return LocalizationResult(
        pose=best["pose"],
        candidate_ids=cand_ids,
        chosen_id=best["id"],
        num_pnp_points=best["n_points"],
        num_inliers=best["n_inliers"],
        inlier_ratio=best["inlier_ratio"],
        reproj_err=best["reproj_err"],
    )
