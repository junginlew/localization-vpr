"""COLMAP SfM 워커 — torch·faiss-cpu가 없는 별도 프로세스에서 매칭을 돌려 faiss 이중 로드 충돌을 피한다.

run_sfm이 work_dir/manifest.json을 써두고 이 스크립트를 subprocess로 호출한다. 결과(가장 많이 등록된
모델)는 work_dir/model 에 저장된다. 이 프로세스는 pycolmap만 올라오므로 매칭이 안전하게 끝난다.

사용: python scripts/sfm_worker.py <work_dir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vpr.mapping import run_colmap_from_manifest

if __name__ == "__main__":
    sys.exit(run_colmap_from_manifest(sys.argv[1]))
