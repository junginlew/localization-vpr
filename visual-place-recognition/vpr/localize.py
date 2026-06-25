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
    chosen_id: int | None = None         # PnP 인라이어를 가장 많이 낸 키프레임 행 번호(진단용). 측위 실패면 None
    num_pnp_points: int = 0              # PnP 입력 3D-2D 대응 수 (매칭 ∩ 삼각측량 성공)
    num_inliers: int = 0                # PnP RANSAC 인라이어 수 (실패 시 0)


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

    입력 점 또는 RANSAC 인라이어가 min_points 미만이거나 solvePnPRansac이 실패하면 측위 실패로 봄.
    """
    points3d = np.ascontiguousarray(points3d, dtype=np.float64)
    points2d = np.ascontiguousarray(points2d, dtype=np.float64)
    if len(points3d) < min_points:
        return None, None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d, points2d, np.asarray(K, dtype=np.float64), None,
        reprojectionError=reproj_err, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < min_points:
        return None, None

    # solvePnP는 월드→카메라(T_cam_world)를 주므로 T_world_cam으로 역변환.
    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T
    t_wc = (-R_wc @ tvec).ravel()
    return make_se3(R_wc, t_wc), inliers.ravel()  # T_world_cam (4,4), 인라이어 인덱스


def localize_query(query_image, keyframes, index, K, image_size,
                   global_extractor, local_extractor, matcher, top_k=10,
                   min_pnp_points=MIN_PNP_POINTS):
    """쿼리 이미지 한 장에서 6-DoF 포즈를 추정해 LocalizationResult로 반환한다.

    검색(Top-K) → 후보별 로컬 매칭 → 모든 후보의 (월드 3D, 쿼리 2D)를 한 풀에 모아 단일 PnP-RANSAC.
    틀린 후보의 대응은 RANSAC이 자동으로 outlier로 버린다. 측위에 실패해도 candidate_ids는 채워 recall 평가 가능.
    """
    query_global = global_extractor(query_image)
    query_kp, query_desc = local_extractor(query_image)

    cand_ids, _ = search_topk(index, query_global, top_k)
    cand_ids = cand_ids[cand_ids >= 0]   # faiss는 키프레임이 k보다 적으면 -1로 채움 → 제외

    pts3d, pts2d, src = [], [], []       # 풀: 월드 3D, 쿼리 2D, 각 대응의 출처 키프레임 번호
    for kf_i in cand_ids:
        kf = keyframes[kf_i]
        idx = matcher(query_kp, query_desc, kf.keypoints, kf.local_desc, image_size)
        p3 = kf.points3d[idx[:, 1]]              # 매칭된 키프레임 키포인트의 월드 3D
        valid = ~np.isnan(p3).any(axis=1)        # 삼각측량 실패(NaN) 키포인트 제외
        pts3d.append(p3[valid])
        pts2d.append(query_kp[idx[:, 0]][valid])  # 그에 대응하는 쿼리 2D
        src.append(np.full(int(valid.sum()), kf_i))

    if not pts3d:           # 후보가 하나도 없음 → 측위 실패 (후보는 보존)
        return LocalizationResult(pose=None, candidate_ids=cand_ids)
    pts3d = np.concatenate(pts3d)
    pts2d = np.concatenate(pts2d)
    src = np.concatenate(src)

    T_world_cam, inliers = estimate_pose(pts3d, pts2d, K, min_points=min_pnp_points)
    chosen = None
    if inliers is not None and len(inliers):
        vals, counts = np.unique(src[inliers], return_counts=True)  # 인라이어의 출처별 개수
        chosen = int(vals[counts.argmax()])                        # 가장 많이 기여한 키프레임
    return LocalizationResult(
        pose=T_world_cam,
        candidate_ids=cand_ids,
        chosen_id=chosen,
        num_pnp_points=int(len(pts3d)),
        num_inliers=0 if inliers is None else len(inliers),
    )
