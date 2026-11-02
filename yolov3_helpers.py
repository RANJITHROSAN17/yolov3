"""Small, dependency-free helpers embedded in the Colab notebook.

Run tests with ``python -m unittest discover -s tests -v``. If this file changes,
update the helper cell in the notebook too; a test checks they stay identical.
"""
from pathlib import Path
import hashlib
import math
import os
import re
import subprocess
import tempfile
import urllib.request


SUPPORTED_IMAGES = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_checked(url, path, expected_sha256):
    """Download to a temporary file; reject corrupt data without replacing a good file."""
    path = Path(path)
    if path.is_file() and file_sha256(path) == expected_sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.download-', dir=path.parent)
    os.close(fd)
    try:
        # Some weight hosts reject urllib's default user agent. Limit the download
        # size and verify its digest before it can replace an existing file.
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=30) as response, open(temp, 'wb') as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 200 * 1024 * 1024:
                    raise ValueError('Weight download exceeds the 200 MiB safety limit.')
                output.write(chunk)
        if file_sha256(temp) != expected_sha256:
            raise ValueError(f'Checksum mismatch for {url}; download was not used.')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def set_option(section, key, value):
    pattern = re.compile(r'^\s*' + re.escape(key) + r'\s*=.*$', re.MULTILINE)
    section, count = pattern.subn(f'{key}={value}', section)
    if count != 1:
        raise ValueError(f'Expected exactly one active {key} option, found {count}.')
    return section


def make_training_cfg(source, classes, batch=64, subdivisions=16, max_batches=None):
    """Configure the three YOLOv3 heads by section, not by fragile line numbers."""
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 1:
        raise ValueError('classes must be a positive integer')
    if batch < 1 or subdivisions < 1 or batch % subdivisions:
        raise ValueError('batch must be positive and divisible by subdivisions')
    if max_batches is None:
        max_batches = max(6000, classes * 2000)
    if not isinstance(max_batches, int) or max_batches < 2:
        raise ValueError('max_batches must be an integer of at least 2')
    sections = re.split(r'(?=^\[[^\]]+\]\s*$)', source, flags=re.MULTILINE)
    blocks = [i for i, block in enumerate(sections) if re.match(r'^\[net\]', block)]
    if len(blocks) != 1:
        raise ValueError('Expected exactly one [net] block')
    net = blocks[0]
    for key, value in [('batch', batch), ('subdivisions', subdivisions), ('max_batches', max_batches),
                       ('steps', f'{int(max_batches * 0.8)},{int(max_batches * 0.9)}')]:
        sections[net] = set_option(sections[net], key, value)
    heads = [i for i, block in enumerate(sections) if re.match(r'^\[yolo\]', block)]
    if len(heads) != 3:
        raise ValueError(f'Expected three YOLOv3 heads, found {len(heads)}')
    for index in heads:
        if index == 0 or not re.match(r'^\[convolutional\]', sections[index - 1]):
            raise ValueError('Each YOLO head must follow its prediction convolution')
        mask = re.search(r'^\s*mask\s*=\s*([0-9,\s]+)$', sections[index], re.MULTILINE)
        if not mask:
            raise ValueError('YOLO head is missing its anchor mask')
        anchors = [part.strip() for part in mask.group(1).split(',') if part.strip()]
        if len(anchors) != 3:
            raise ValueError('Expected three anchors per YOLOv3 head')
        sections[index] = set_option(sections[index], 'classes', classes)
        sections[index - 1] = set_option(sections[index - 1], 'filters', (classes + 5) * len(anchors))
    return ''.join(sections)


def checked_names(names):
    if not isinstance(names, (list, tuple)) or not names or any(not isinstance(name, str) or not name.strip() or '\n' in name or '\r' in name for name in names):
        raise ValueError('Supply one non-empty class name per line')
    result = [name.strip() for name in names]
    if len(set(result)) != len(result):
        raise ValueError('Class names must be distinct')
    return result


def darknet_label_path(image):
    """Mirror the pinned Darknet revision's image-to-label path substitutions."""
    path = str(image)
    for split in ('train2017', 'val2017', 'train2014', 'val2014'):
        path = path.replace(f'/images/{split}/', f'/labels/{split}/')
    path = path.replace('/JPEGImages/', '/labels/')
    return Path(path).with_suffix('.txt')


def validate_split(directory, class_count):
    """Require explicit labels (empty is OK for negatives) and valid normalized boxes."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    images = sorted(p.resolve() for p in root.rglob('*') if p.is_file() and p.suffix in SUPPORTED_IMAGES)
    if not images:
        raise ValueError(f'No supported images found in {root}')
    if len(set(images)) != len(images):
        raise ValueError(f'Duplicate image paths found in {root}')
    for image in images:
        if '\n' in str(image) or '\r' in str(image):
            raise ValueError(f'Image path contains a newline: {image!s}')
        label = darknet_label_path(image)
        if not label.is_file():
            raise ValueError(f'Missing label for {image}: expected {label}; use an empty file for a negative image')
        for number, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            try:
                if len(parts) != 5 or not re.fullmatch(r'\d+', parts[0]):
                    raise ValueError()
                category = int(parts[0])
                x, y, width, height = map(float, parts[1:])
                if category >= class_count or not all(math.isfinite(v) for v in (x, y, width, height)):
                    raise ValueError()
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError()
                if x - width / 2 < -1e-6 or y - height / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y + height / 2 > 1 + 1e-6:
                    raise ValueError()
            except ValueError:
                raise ValueError(f'Invalid YOLO label at {label}:{number}; expected class_id x_center y_center width height in [0, 1]') from None
    return images


def prepare_dataset(train_dir, valid_dir, names, output_dir, backup_dir):
    """Write lists and data files only after both splits pass preflight checks."""
    names = checked_names(names)
    train = validate_split(train_dir, len(names))
    valid = validate_split(valid_dir, len(names))
    if set(train) & set(valid):
        raise ValueError('Train and validation image paths overlap')
    train_hashes = {file_sha256(path) for path in train}
    if any(file_sha256(path) in train_hashes for path in valid):
        raise ValueError('Identical image bytes occur in both training and validation; use independent splits')
    output = Path(output_dir).expanduser().resolve()
    backup = Path(backup_dir).expanduser().resolve()
    if output == backup:
        raise ValueError('Output and backup directories must be different to protect existing checkpoints')
    # Darknet strips whitespace in .data values, so reject paths it would misread.
    if any(char.isspace() for char in str(output)) or any(char.isspace() for char in str(backup)):
        raise ValueError('Darknet .data paths cannot contain whitespace; choose output and backup paths without spaces')
    output.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    (output / 'obj.names').write_text('\n'.join(names) + '\n', encoding='utf-8')
    (output / 'train.txt').write_text('\n'.join(map(str, train)) + '\n', encoding='utf-8')
    (output / 'valid.txt').write_text('\n'.join(map(str, valid)) + '\n', encoding='utf-8')
    data = f'classes={len(names)}\ntrain={output / "train.txt"}\nvalid={output / "valid.txt"}\nnames={output / "obj.names"}\nbackup={backup}\n'
    (output / 'obj.data').write_text(data, encoding='utf-8')
    return train, valid


def run_detection(darknet_dir, data, cfg, weights, image, threshold=0.25):
    """Run real Darknet inference, failing on process errors or missing fresh output."""
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between 0 and 1')
    root = Path(darknet_dir).resolve()
    for name, value in [('data', data), ('cfg', cfg), ('weights', weights), ('image', image)]:
        if not Path(value).is_file():
            raise FileNotFoundError(f'{name} file not found: {value}')
    output = root / 'predictions.jpg'
    output.unlink(missing_ok=True)
    result = subprocess.run([str(root / 'darknet'), 'detector', 'test', str(Path(data).resolve()),
                             str(Path(cfg).resolve()), str(Path(weights).resolve()), str(Path(image).resolve()),
                             '-dont_show', '-thresh', str(threshold)], cwd=root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError('Darknet exited without creating predictions.jpg; check the log and runtime.')
    return result.stdout, output
