#!/usr/bin/env python3
"""CPU-only pretrained YOLOv3-tiny smoke test; downloads external code and weights."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolov3_helpers import download_checked, run_detection

REVISION = '59596d7880f6504768df41d6daa586f5cb2b932f'
TINY_SHA256 = 'dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c'


def smoke(workdir):
    root = Path(workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    darknet = root / 'darknet'
    if not darknet.exists():
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/AlexeyAB/darknet.git', str(darknet)], check=True)
        subprocess.run(['git', '-C', str(darknet), 'checkout', '--detach', REVISION], check=True)
    else:
        revision = subprocess.check_output(['git', '-C', str(darknet), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(darknet), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if revision != REVISION or dirty.strip():
            raise RuntimeError('Darknet checkout must be clean and at the pinned revision; use another work directory.')
    subprocess.run(['make', 'clean'], cwd=darknet, check=True)
    subprocess.run(['make', '-j2', 'GPU=0', 'CUDNN=0', 'OPENCV=0', 'LIBSO=0'], cwd=darknet, check=True)
    weights = download_checked('https://pjreddie.com/media/files/yolov3-tiny.weights', root / 'yolov3-tiny.weights', TINY_SHA256)
    log, output = run_detection(darknet, darknet / 'cfg/coco.data', darknet / 'cfg/yolov3-tiny.cfg', weights, darknet / 'data/dog.jpg')
    print(log[-5000:])
    if not re.search(r'(?m)^dog:\s*\d', log):
        raise RuntimeError('The pretrained tiny model did not detect a dog at the chosen threshold.')
    print(f'Smoke test passed: dog detected; annotated image: {output}')
    print('This checks pretrained YOLOv3-tiny inference, not custom-model accuracy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, help='Optional disposable work directory; build artifacts and a 35 MB download are stored here.')
    args = parser.parse_args()
    if args.work_dir:
        smoke(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='yolov3-smoke-') as temp:
            smoke(temp)


if __name__ == '__main__':
    main()
