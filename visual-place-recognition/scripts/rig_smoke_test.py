"""pycolmap rig 스모크 테스트.

작은 스테레오 시퀀스를 rig off / rig on 두 번 재구성하고, 프레임별 좌우 카메라 중심
거리의 산포(변동계수 = std/mean)를 비교한다.
  - rig on  → cam0-cam1 상대 포즈가 BA에 공유 제약으로 들어가 거리 산포 ≈ 0
  - rig off → 좌우 이미지가 독립 삼각측량되어 거리 산포가 크다
산포가 충분히 줄면 rig 경로 채택, 비슷하면 중앙값 스케일 보정 폴백(plan 리스크 표).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # COLMAP faiss CPU 매처의 OpenBLAS 스레드 충돌(segfault) 회피

from pathlib import Path
import shutil

import numpy as np
import pycolmap as pc

REC = Path("data/4seasons/recording_2020-04-07_10-20-32/undistorted_images")
# undistorted_calib_0: PINHOLE fx fy cx cy / 800x400, stereo baseline x = -0.3005
CAM_PARAMS = [501.4757919305817, 501.4757919305817, 421.7953735163109, 167.65799492501083]
BASELINE_X = -0.3004961618953472  # undistorted_calib_stereo.txt T[0,3]

N_PAIRS = 40   # 스테레오 쌍 개수 (소규모 스모크)
STEP = 2       # 프레임 간격 (차량 이동 parallax 확보)
WORK = Path("/tmp/rig_smoke")


def select_timestamps():
    names = sorted(p.stem for p in (REC / "cam0").glob("*.png"))
    return names[: N_PAIRS * STEP : STEP]


def setup_images(timestamps):
    img_dir = WORK / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    for cam in ("cam0", "cam1"):
        (img_dir / cam).mkdir(parents=True)
        for ts in timestamps:
            (img_dir / cam / f"{ts}.png").symlink_to((REC / cam / f"{ts}.png").resolve())
    return img_dir


def build_database(img_dir):
    db_path = WORK / "database.db"
    if db_path.exists():
        db_path.unlink()
    reader = pc.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = ",".join(str(v) for v in CAM_PARAMS)
    pc.extract_features(db_path, img_dir, camera_mode=pc.CameraMode.PER_FOLDER,
                        reader_options=reader)
    pc.match_exhaustive(db_path)
    return db_path


def rig_config():
    """cam0=기준 센서, cam1=baseline만큼 떨어진 고정 상대 포즈."""
    cam0 = pc.RigConfigCamera(ref_sensor=True, image_prefix="cam0/")
    cam1_from_cam0 = pc.Rigid3d(np.array([[1.0, 0, 0, BASELINE_X],
                                          [0, 1.0, 0, 0],
                                          [0, 0, 1.0, 0]]))
    cam1 = pc.RigConfigCamera(ref_sensor=False, image_prefix="cam1/",
                              cam_from_rig=cam1_from_cam0)
    return pc.RigConfig(cameras=[cam0, cam1])


def reconstruct(img_dir, db_path, use_rig):
    out = WORK / ("rig_on" if use_rig else "rig_off")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if use_rig:
        db = pc.Database.open(str(db_path))
        pc.apply_rig_config([rig_config()], db)
        db.close()
    opt = pc.IncrementalPipelineOptions()
    opt.mapper.init_min_tri_angle = 2.0  # 차량 직진+30cm 스테레오는 삼각측량 각도가 작아 기본 16°를 못 넘음
    opt.min_model_size = 5
    recs = pc.incremental_mapping(db_path, img_dir, out, options=opt)
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())


def left_right_spread(recon):
    """등록된 프레임별 좌우 카메라 중심 거리의 (개수, 평균, 변동계수)."""
    by_ts = {}
    for img in recon.images.values():
        cam, fname = img.name.split("/")
        by_ts.setdefault(fname, {})[cam] = img
    dists = []
    for cams in by_ts.values():
        if "cam0" in cams and "cam1" in cams:
            c0 = np.asarray(cams["cam0"].projection_center())
            c1 = np.asarray(cams["cam1"].projection_center())
            dists.append(float(np.linalg.norm(c0 - c1)))
    dists = np.array(dists)
    if len(dists) == 0:
        return 0, 0.0, float("nan")
    cv = float(dists.std() / dists.mean()) if dists.mean() > 0 else float("nan")
    return len(dists), float(dists.mean()), cv


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    timestamps = select_timestamps()
    print(f"선택 스테레오 쌍: {len(timestamps)} (step={STEP})")
    img_dir = setup_images(timestamps)
    db_path = build_database(img_dir)

    for use_rig in (False, True):
        recon = reconstruct(img_dir, db_path, use_rig)
        label = "rig ON " if use_rig else "rig OFF"
        if recon is None:
            print(f"{label}: 재구성 실패")
            continue
        n_pairs, mean_d, cv = left_right_spread(recon)
        print(f"{label}: 등록이미지 {recon.num_reg_images()}, "
              f"좌우쌍 {n_pairs}, 평균거리 {mean_d:.4f}, 변동계수(std/mean) {cv:.4f}")


if __name__ == "__main__":
    main()
