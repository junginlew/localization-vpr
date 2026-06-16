import cv2
import faiss
import numpy as np

from vpr.geometry import make_se3

MIN_PNP_POINTS = 15  # PnP 입력 3D-2D 대응 하한 (이 미만이면 측위 실패)


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
