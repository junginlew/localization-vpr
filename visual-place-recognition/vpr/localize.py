import cv2
import faiss
import numpy as np

from vpr.geometry import make_se3

MIN_PNP_POINTS = 15            # PnP 입력 3D-2D 대응 하한 (이 미만이면 측위 실패)
MIN_ESSENTIAL_INLIERS = 30     # 에피폴라 검증 인라이어 하한 (이 미만 후보는 장소 오인식으로 폐기)


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


def verify_essential(query_pts, kf_pts, K, min_inliers=MIN_ESSENTIAL_INLIERS, ransac_thresh=1.0):
    """매칭된 쿼리·키프레임 2D 점쌍을 essential matrix RANSAC으로 검증해 에피폴라 인라이어 인덱스를 반환한다."""
    query_pts = np.ascontiguousarray(query_pts, dtype=np.float64)
    kf_pts = np.ascontiguousarray(kf_pts, dtype=np.float64)
    if len(query_pts) < min_inliers:  # 매칭 자체가 임계보다 적으면 폐기
        return None

    E, mask = cv2.findEssentialMat(
        query_pts, kf_pts, np.asarray(K, dtype=np.float64),
        method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh,
    )
    if E is None or mask is None:
        return None
    inliers = np.flatnonzero(mask.ravel())  # mask는 (N,1) 0/1 → 인라이어 행 번호 배열
    if len(inliers) < min_inliers:
        return None
    return inliers


def select_candidate(candidate_matches, K, min_inliers=MIN_ESSENTIAL_INLIERS):
    """Top-K 후보별 (쿼리 2D, 키프레임 2D) 매칭에서 에피폴라 인라이어가 가장 많은 후보를 고른다."""
    best_idx = None
    best_count = 0
    for i, (query_pts, kf_pts) in enumerate(candidate_matches):
        inliers = verify_essential(query_pts, kf_pts, K, min_inliers)
        if inliers is not None and len(inliers) > best_count:
            best_idx, best_count = i, len(inliers)
    return best_idx


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
                   global_extractor, local_extractor, matcher, top_k=10):
    """쿼리 이미지 한 장에서 6-DoF 포즈 T_world_cam 추정.

    검색(Top-K) → 후보별 로컬 매칭 → 에피폴라 검증·후보 선택 → 확정 후보의 3D-2D로 PnP.
    """
    query_global = global_extractor(query_image)
    query_kp, query_desc = local_extractor(query_image)

    cand_ids, _ = search_topk(index, query_global, top_k)

    candidate_matches = []  # (쿼리 2D, 키프레임 2D)
    matched = []            # 후보별 (키프레임 번호, 매칭 인덱스쌍)
    for kf_i in cand_ids:
        if kf_i < 0:        # faiss는 키프레임이 k보다 적으면 -1로 채움
            continue
        kf = keyframes[kf_i]
        idx = matcher(query_kp, query_desc, kf.keypoints, kf.local_desc, image_size)
        candidate_matches.append((query_kp[idx[:, 0]], kf.keypoints[idx[:, 1]]))
        matched.append((kf_i, idx))

    best = select_candidate(candidate_matches, K)
    if best is None:        # 모든 후보가 에피폴라 검증에서 폐기 → 측위 실패
        return None

    kf_i, idx = matched[best]
    kf = keyframes[kf_i]
    points3d = kf.points3d[idx[:, 1]]      # 매칭된 키프레임 키포인트의 월드 3D
    points2d = query_kp[idx[:, 0]]         # 그에 대응하는 쿼리 2D
    valid = ~np.isnan(points3d).any(axis=1)  # 삼각측량 실패(NaN) 키포인트 제외
    T_world_cam, _ = estimate_pose(points3d[valid], points2d[valid], K)
    return T_world_cam
