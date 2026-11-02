"""Source/artifact identity checks only; no inference or deployment approval."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

BOOTSTRAP_BASE = '10ec988f3cdd5036613d4a8adaae5aa35199e3d3'
OFFICIAL_COMMIT = '48b10fa505aeb51fbd127fc60a5c660040e823b7'
OFFICIAL_MODELS = {
    'models/face_detection_yunet/face_detection_yunet_2023mar.onnx': ('8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4', 232589, '2d8804a5986e229f1fde3a1994feacc66c91b58b'),
    'models/face_recognition_sface/face_recognition_sface_2021dec.onnx': ('0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79', 38696353, '5817e559d509b2c1d5069f3c49a388bc45d4395f'),
}


def main():
    base = os.environ.get('BASE_SHA', '')
    if not re.fullmatch('[a-f0-9]{40}', base):
        raise SystemExit('Missing trusted event base SHA')
    path = 'governance/model-source-boundary.json'
    current = json.loads(Path(path).read_text())
    previous = subprocess.run(['git', 'show', f'{base}:{path}'], capture_output=True)
    if previous.returncode == 0:
        trusted = json.loads(previous.stdout)
        if current != trusted:
            raise SystemExit('Model provenance change requires independent model-specific review')
        subprocess.run(['git', 'diff', '--exit-code', base, '--', '.github', path,
                        'scripts/check_model_boundary.py', *trusted['frozen_files']], check=True)
    else:
        if base != BOOTSTRAP_BASE:
            raise SystemExit('Unreviewed model-boundary bootstrap; no permissive fallback')
        trusted = current  # Initial exact commit is independently reviewed before policy installation.
    assert trusted['upstream_repository'] == 'opencv/opencv_zoo'
    assert trusted['upstream_commit'] == OFFICIAL_COMMIT
    for flag in ('commercial_deployment_approved', 'real_inference_certified', 'hardware_certified'):
        assert trusted[flag] is False
    actual_models = {model['path']: (model['sha256'], model['size'], model['pointer_blob']) for model in trusted['models']}
    assert actual_models == OFFICIAL_MODELS
    required = {'Dockerfile', 'NOTICE.md', 'requirements.txt', 'app/main.py',
                'licenses/models/yunet-LICENSE.txt', 'licenses/models/sface-LICENSE.txt'}
    assert set(trusted['frozen_files']) == required
    for name, digest in trusted['frozen_files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    for model, (digest, size, blob) in OFFICIAL_MODELS.items():
        url = f'https://api.github.com/repos/opencv/opencv_zoo/git/blobs/{blob}'
        with urllib.request.urlopen(url, timeout=30) as response:
            envelope = json.load(response)
        raw = base64.b64decode(envelope['content'])
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob
        pointer = raw.decode()
        assert pointer.splitlines() == ['version https://git-lfs.github.com/spec/v1',
                                       f'oid sha256:{digest}', f'size {size}']
    names = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
    assert not any(name.lower().endswith(('.onnx', '.pt', '.pth', '.tflite', '.pb', '.bin')) for name in names)
    original = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BOOTSTRAP_BASE]).decode().splitlines())
    additions = {'governance/model-source-boundary.json', 'governance/interim-release-policy.json',
                 '.github/workflows/recognition-source.yml', 'scripts/check_model_boundary.py',
                 'licenses/models/yunet-LICENSE.txt', 'licenses/models/sface-LICENSE.txt'}
    for name in filter(None, names):
        assert name in original or name in additions or (name.startswith('tests/') and name.endswith('.py')), 'Unreviewed source/model addition'
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z']).decode().split('\0')
    assert all(entry.split()[0] == '100644' for entry in entries if entry), 'Executable/symlink/submodule not allowed in this model-independent lane'
    print('Official LFS identities and frozen model configuration PASS; inference/commercial approval remains OFF')


if __name__ == '__main__':
    main()
