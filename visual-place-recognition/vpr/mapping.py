import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # COLMAP faiss CPU 매처의 OpenBLAS 스레드 충돌(segfault) 회피

import shutil
from pathlib import Path

import cv2
import numpy as np
import pycolmap as pc

from vpr.geometry import apply_se3, rotation_error_deg, translation_error
from vpr.map_db import Keyframe, save_map

INIT_MIN_TRI_ANGLE = 2.0          # 차량 직진+짧은 baseline은 삼각측량 각도가 작아 기본 16°로는 초기화 실패
MIN_KEYFRAME_TRANSLATION = 2.0    # 키프레임 선별: 직전 키프레임 대비 이동 하한 (m)
MIN_KEYFRAME_ROTATION_DEG = 15.0  # 키프레임 선별: 직전 키프레임 대비 회전 하한 (deg)
MIN_STEREO_DEPTH = 0.5            # m, 너무 가깝거나 음수 깊이(삼각측량 실패) 제외
MAX_STEREO_DEPTH = 50.0           # m, 30cm baseline에선 원거리 깊이 신뢰도 급락


def _setup_image_dir(work, frames):
    """선택 프레임의 cam0/cam1 이미지를 work/images/{cam}/{타임스탬프}.png 심볼릭으로 모은다."""
    img_dir = work / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    for cam, attr in (("cam0", "cam0_path"), ("cam1", "cam1_path")):
        (img_dir / cam).mkdir(parents=True)
        for f in frames:
            link = img_dir / cam / f"{f.timestamp}.png"
            link.symlink_to(Path(getattr(f, attr)).resolve())
    return img_dir


def _rig_config(calib):
    """cam0=기준 센서, cam1=baseline만큼 떨어진 고정 상대 포즈 (스케일을 메트릭으로 고정)."""
    cam0 = pc.RigConfigCamera(ref_sensor=True, image_prefix="cam0/")
    cam1_from_cam0 = pc.Rigid3d(np.array([[1.0, 0, 0, -calib.baseline],
                                          [0, 1.0, 0, 0],
                                          [0, 0, 1.0, 0]]))
    cam1 = pc.RigConfigCamera(ref_sensor=False, image_prefix="cam1/", cam_from_rig=cam1_from_cam0)
    return pc.RigConfig(cameras=[cam0, cam1])


def run_sfm(frames, calib, work_dir, init_min_tri_angle=INIT_MIN_TRI_ANGLE):
    """선택 프레임의 스테레오 이미지로 rig 제약 SfM을 돌려 메트릭 reconstruction을 반환한다.

    GT 포즈는 사용안함(평가 전용 격리) - 입력 frames에서 이미지 경로만 사용.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    img_dir = _setup_image_dir(work, frames)

    db_path = work / "database.db"
    if db_path.exists():
        db_path.unlink()
    reader = pc.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    fx, fy, cx, cy = calib.K[0, 0], calib.K[1, 1], calib.K[0, 2], calib.K[1, 2]
    reader.camera_params = f"{fx},{fy},{cx},{cy}"
    pc.extract_features(db_path, img_dir, camera_mode=pc.CameraMode.PER_FOLDER, reader_options=reader)
    pc.match_exhaustive(db_path)

    db = pc.Database.open(str(db_path))           # cam0-cam1을 30cm 고정 rig으로 묶어 메트릭 스케일 확보
    pc.apply_rig_config([_rig_config(calib)], db)
    db.close()

    out = work / "sparse"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    opt = pc.IncrementalPipelineOptions()
    opt.mapper.init_min_tri_angle = init_min_tri_angle
    recs = pc.incremental_mapping(db_path, img_dir, out, options=opt)
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())  # 가장 많이 등록된 모델 채택


def _to_se3(rigid):
    """pycolmap Rigid3d → (4, 4) SE3 행렬."""
    T = np.eye(4)
    T[:3, :] = rigid.matrix()
    return T


def select_keyframes(reconstruction, min_translation=MIN_KEYFRAME_TRANSLATION,
                     min_rotation_deg=MIN_KEYFRAME_ROTATION_DEG):
    """등록된 cam0 프레임을 시간순으로 훑어, 직전 키프레임 대비 이동/회전이 기준을 넘을 때마다 채택한다.

    반환: [(timestamp, T_world_cam0), ...]
    """
    cam0 = []
    for img in reconstruction.images.values():
        cam, fname = img.name.split("/")
        if cam != "cam0" or not img.has_pose:
            continue
        ts = int(fname[:-4])  # timestamp.png
        T_world_cam0 = _to_se3(img.cam_from_world().inverse())
        cam0.append((ts, T_world_cam0))
    cam0.sort(key=lambda e: e[0]) # 타임스탬프 순

    selected = []
    last_pose = None
    for ts, pose in cam0:
        if (last_pose is None
                or translation_error(pose, last_pose) >= min_translation
                or rotation_error_deg(pose, last_pose) >= min_rotation_deg):
            selected.append((ts, pose))
            last_pose = pose
    return selected


def _mutual_nn(desc0, desc1):
    """두 디스크립터 집합의 상호 최근접(mutual NN) 매칭 인덱스쌍 (M, 2)를 반환한다."""
    d0 = desc0 / np.linalg.norm(desc0, axis=1, keepdims=True)
    d1 = desc1 / np.linalg.norm(desc1, axis=1, keepdims=True)
    sim = d0 @ d1.T                       # (N0, N1) 코사인 유사도
    nn01 = sim.argmax(1)                  # 각 desc0의 최근접 desc1
    nn10 = sim.argmax(0)                  # 각 desc1의 최근접 desc0
    idx0 = np.arange(len(desc0))
    mutual = nn10[nn01] == idx0           # 서로가 서로의 1등
    return np.column_stack([idx0[mutual], nn01[mutual]])


def triangulate_stereo(pts0, pts1, K, baseline):
    """rectified 스테레오 점쌍(cam0·cam1 픽셀)을 cam0 좌표계의 3D 점 (N, 3)으로 삼각측량한다."""
    K = np.asarray(K, dtype=np.float64)
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])                      #  cam0의 3D→픽셀 투영행렬 = [I | 0]
    P1 = K @ np.hstack([np.eye(3), np.array([[-baseline], [0.0], [0.0]])])  # cam1의 3D→픽셀 투영행렬
    pts4d = cv2.triangulatePoints(P0, P1,
                                  np.ascontiguousarray(pts0.T, dtype=np.float64),
                                  np.ascontiguousarray(pts1.T, dtype=np.float64))
    return (pts4d[:3] / pts4d[3]).T       # 동차좌표 (4,N) → (N,3)


def triangulate_keyframe(cam0_path, cam1_path, T_world_cam0, calib, extractor,
                         min_depth=MIN_STEREO_DEPTH, max_depth=MAX_STEREO_DEPTH):
    """키프레임의 cam0 키포인트마다 cam0↔cam1 스테레오 삼각측량으로 월드 3D를 만든다."""
    kp0, desc0 = extractor(cam0_path)
    kp1, desc1 = extractor(cam1_path)
    points3d = np.full((len(kp0), 3), np.nan) # cam0 키포인트마다 월드 3D 좌표 (삼각측량 실패 시 NaN)

    matches = _mutual_nn(desc0, desc1)
    if len(matches):
        cam_pts = triangulate_stereo(kp0[matches[:, 0]], kp1[matches[:, 1]], calib.K, calib.baseline)
        depth = cam_pts[:, 2]
        ok = (depth > min_depth) & (depth < max_depth)  # 음수·너무 가깝거나 먼 깊이 제외
        world_pts = apply_se3(T_world_cam0, cam_pts)                  # cam0 좌표 → 월드
        points3d[matches[ok, 0]] = world_pts[ok]
    return kp0, desc0, points3d # keypoints (N,2), local_desc (N,D), points3d (N,3)


def build_map(frames, calib, work_dir, out_path, global_extractor, local_extractor,
              init_min_tri_angle=INIT_MIN_TRI_ANGLE,
              min_keyframe_translation=MIN_KEYFRAME_TRANSLATION,
              min_keyframe_rotation_deg=MIN_KEYFRAME_ROTATION_DEG,
              min_stereo_depth=MIN_STEREO_DEPTH, max_stereo_depth=MAX_STEREO_DEPTH):
    """프레임 시퀀스로 지도를 만들어 파일로 저장한다.

    SfM으로 포즈 복원 → 키프레임 선별 → 키프레임마다 전역 desc + 로컬 desc + 3D 추출 → 저장.
    """
    reconstruction = run_sfm(frames, calib, work_dir, init_min_tri_angle)
    if reconstruction is None:
        raise RuntimeError("SfM 재구성 실패: 등록된 이미지가 없음")

    frame_by_ts = {f.timestamp: f for f in frames}  # 키프레임 timestamp로 원본 이미지 경로를 찾음
    keyframes = []
    for ts, T_world_cam0 in select_keyframes(reconstruction, min_keyframe_translation, min_keyframe_rotation_deg):
        frame = frame_by_ts[ts]
        global_desc = global_extractor(frame.cam0_path)
        keypoints, local_desc, points3d = triangulate_keyframe(
            frame.cam0_path, frame.cam1_path, T_world_cam0, calib, local_extractor,
            min_stereo_depth, max_stereo_depth)
        keyframes.append(Keyframe(
            kf_id=ts,
            pose=T_world_cam0,
            global_desc=global_desc,
            keypoints=keypoints,
            local_desc=local_desc,
            points3d=points3d,
        ))
    save_map(out_path, keyframes)
    return keyframes
