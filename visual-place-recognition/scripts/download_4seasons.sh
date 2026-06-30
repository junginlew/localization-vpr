#!/usr/bin/env bash
set -euo pipefail

BASE=https://cvg.cit.tum.de/webshare/g/4seasons-dataset
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/4seasons"

RECORDINGS=(
    recording_2020-04-07_10-20-32   # Office Loop, 봄 (지도용)
    recording_2021-01-07_12-04-03   # Office Loop, 겨울 (쿼리용)
    recording_2020-12-22_12-04-35   # Parking Garage, 겨울 (지도용)
    recording_2021-05-10_19-15-19   # Parking Garage, 봄 (쿼리용)
)
PARTS=(reference_poses stereo_images_undistorted)

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

wget -c -nv "$BASE/calibration/calibration.zip"
unzip -n -q calibration.zip

for REC in "${RECORDINGS[@]}"; do
    for PART in "${PARTS[@]}"; do
        wget -c -nv "$BASE/dataset/$REC/${REC}_${PART}.zip"
    done
    for z in "${REC}"_*.zip; do
        unzip -n -q "$z"
    done
done

echo "완료: $DATA_DIR"
