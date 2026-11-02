import importlib.util
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('yolov3_helpers', ROOT / 'yolov3_helpers.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def fixture_cfg():
    return '[net]\nbatch=1\nsubdivisions=1\nmax_batches=500200\nsteps=400000,450000\n' + (
        '[convolutional]\nfilters=255\n[yolo]\nmask = 0,1,2\nclasses=80\n' * 3
    )


class ConfigTests(unittest.TestCase):
    def test_updates_all_heads_and_the_learning_schedule(self):
        result = h.make_training_cfg(fixture_cfg(), 2)
        self.assertEqual(result.count('classes=2\n'), 3)
        self.assertEqual(result.count('filters=21\n'), 3)
        self.assertIn('max_batches=6000\n', result)
        self.assertIn('steps=4800,5400\n', result)
        self.assertIn('batch=64\n', result)
        self.assertIn('subdivisions=16\n', result)

    def test_handles_reordered_lines_and_non_default_class_count(self):
        result = h.make_training_cfg('\n# note\n' + fixture_cfg().replace('mask = 0,1,2\nclasses=80', 'classes=80\nmask = 0,1,2'), 5, max_batches=10000)
        self.assertEqual(result.count('filters=30\n'), 3)
        self.assertEqual(result.count('classes=5\n'), 3)
        self.assertIn('steps=8000,9000', result)

    def test_rejects_malformed_or_unexpected_configs(self):
        for value in (0, -1, True):
            with self.assertRaises(ValueError): h.make_training_cfg(fixture_cfg(), value)
        with self.assertRaisesRegex(ValueError, 'three YOLOv3 heads'): h.make_training_cfg(fixture_cfg().replace('[yolo]', '[other]', 1), 2)
        with self.assertRaisesRegex(ValueError, 'divisible'): h.make_training_cfg(fixture_cfg(), 2, batch=64, subdivisions=15)
        with self.assertRaisesRegex(ValueError, 'anchors'): h.make_training_cfg(fixture_cfg().replace('mask = 0,1,2', 'mask = 0,1', 1), 2)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='yolo-tests-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train, self.valid = self.root / 'train', self.root / 'valid'
        self.train.mkdir(); self.valid.mkdir()
        (self.train / 'a.jpg').write_bytes(b'train-image')
        (self.valid / 'b.png').write_bytes(b'valid-image')
        (self.train / 'a.txt').write_text('0 0.5 0.5 0.2 0.4\n')
        (self.valid / 'b.txt').write_text('')  # explicit negative image

    def test_prepares_explicit_independent_splits(self):
        train, valid = h.prepare_dataset(self.train, self.valid, ['part', 'bolt'], self.root / 'out', self.root / 'backup')
        self.assertEqual(len(train), 1); self.assertEqual(len(valid), 1)
        self.assertEqual((self.root / 'out/obj.names').read_text(), 'part\nbolt\n')
        data = (self.root / 'out/obj.data').read_text()
        self.assertIn('classes=2\n', data)
        self.assertIn(str(self.root / 'out/valid.txt'), data)
        self.assertTrue((self.root / 'backup').is_dir())

    def test_rejects_missing_or_invalid_labels(self):
        path = self.train / 'a.txt'
        for bad in ['2 0.5 0.5 0.2 0.2', '0 nan 0.5 0.2 0.2', '0 0.5 0.5 0 0.2', '0 0.1 0.5 0.4 0.2', '0 0.1']:
            path.write_text(bad)
            with self.subTest(label=bad), self.assertRaisesRegex(ValueError, 'Invalid YOLO label'):
                h.validate_split(self.train, 2)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'): h.validate_split(self.train, 2)

    def test_rejects_split_leakage_before_writing(self):
        (self.valid / 'b.png').write_bytes((self.train / 'a.jpg').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Identical image'): h.prepare_dataset(self.train, self.valid, ['part'], self.root / 'out', self.root / 'backup')
        self.assertFalse((self.root / 'out').exists())
        with self.assertRaisesRegex(ValueError, 'overlap'): h.prepare_dataset(self.train, self.train, ['part'], self.root / 'out', self.root / 'backup')

    def test_rejects_unsafe_data_paths_and_invalid_names(self):
        with self.assertRaisesRegex(ValueError, 'whitespace'): h.prepare_dataset(self.train, self.valid, ['part'], self.root / 'path with spaces', self.root / 'backup')
        with self.assertRaisesRegex(ValueError, 'must be different'): h.prepare_dataset(self.train, self.valid, ['part'], self.root / 'same', self.root / 'same')
        for names in ([], [''], ['x', 'x'], ['x\ny']):
            with self.subTest(names=names), self.assertRaises(ValueError): h.checked_names(names)

    def test_darknet_special_label_paths(self):
        self.assertEqual(str(h.darknet_label_path('/data/images/train2017/a.JPG')), '/data/labels/train2017/a.txt')
        self.assertEqual(str(h.darknet_label_path('/data/JPEGImages/a.jpg')), '/data/labels/a.txt')
        self.assertEqual(str(h.darknet_label_path('/data/train/a.png')), '/data/train/a.txt')


class DetectionTests(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'fake Darknet fixture is a POSIX shell script')
    def test_detection_propagates_failure_and_never_accepts_stale_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for file in ['obj.data', 'net.cfg', 'model.weights', 'sample.jpg']:
                (root / file).write_text('fixture')
            command = root / 'darknet'
            command.write_text('#!/bin/sh\necho "dog: 90%"\nprintf image > predictions.jpg\n')
            command.chmod(0o755)
            args = [root, root / 'obj.data', root / 'net.cfg', root / 'model.weights', root / 'sample.jpg']
            stdout, output = h.run_detection(*args)
            self.assertIn('dog: 90%', stdout); self.assertEqual(output.read_text(), 'image')
            command.write_text('#!/bin/sh\nexit 0\n')
            with self.assertRaisesRegex(RuntimeError, 'without creating'): h.run_detection(*args)
            self.assertFalse(output.exists())
            command.write_text('#!/bin/sh\nexit 1\n')
            with self.assertRaises(h.subprocess.CalledProcessError): h.run_detection(*args)
            with self.assertRaises(ValueError): h.run_detection(*args, threshold=1.0)

    def test_checked_download_rejects_bad_checksum_and_keeps_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'weights'; path.write_bytes(b'previous')
            with patch.object(h.urllib.request, 'urlopen', return_value=io.BytesIO(b'bad download')):
                with self.assertRaisesRegex(ValueError, 'Checksum mismatch'): h.download_checked('https://example.invalid/file', path, '0' * 64)
            self.assertEqual(path.read_bytes(), b'previous')
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ['weights'])

    def test_checked_download_uses_host_compatible_header_and_caches_verified_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'weights'
            digest = h.hashlib.sha256(b'verified').hexdigest()
            with patch.object(h.urllib.request, 'urlopen', return_value=io.BytesIO(b'verified')) as opener:
                self.assertEqual(h.download_checked('https://example.invalid/file', path, digest), path)
                self.assertEqual(opener.call_args.args[0].get_header('User-agent'), 'Mozilla/5.0')
            with patch.object(h.urllib.request, 'urlopen', side_effect=AssertionError('should be cached')):
                self.assertEqual(h.download_checked('https://example.invalid/file', path, digest), path)
            self.assertEqual(path.read_bytes(), b'verified')


class NotebookTests(unittest.TestCase):
    def test_notebook_has_synchronized_helpers_and_no_stale_outputs(self):
        notebook = json.loads((ROOT / 'YOLOv3_Object_Detection.py.ipynb').read_text())
        cells = notebook['cells']
        helper = [c for c in cells if c.get('metadata', {}).get('tags') == ['yolov3-helpers']]
        self.assertEqual(len(helper), 1)
        self.assertEqual(''.join(helper[0]['source']), (ROOT / 'yolov3_helpers.py').read_text())
        for cell in cells:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']), '<notebook cell>', 'exec')

    def test_custom_preflight_rejects_incompatible_checkpoint_metadata(self):
        notebook = json.loads((ROOT / 'YOLOv3_Object_Detection.py.ipynb').read_text())
        preflight = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).startswith('if ENABLE_CUSTOM:'))
        with tempfile.TemporaryDirectory(prefix='notebook-test-') as temp:
            root = Path(temp)
            for split, image in [('train', b'train'), ('valid', b'valid')]:
                folder = root / split; folder.mkdir()
                (folder / 'sample.jpg').write_bytes(image)
                (folder / 'sample.txt').write_text('0 0.5 0.5 0.2 0.2\n')
            darknet = root / 'darknet'; (darknet / 'cfg').mkdir(parents=True)
            (darknet / 'cfg/yolov3.cfg').write_text(fixture_cfg())
            backup = root / 'backup'; backup.mkdir()
            (backup / 'obj.names').write_text('different class\n')
            env = dict(vars(h), shutil=shutil, ENABLE_CUSTOM=True, CLASS_NAMES=['part'], TRAIN_DIR=root / 'train',
                       VALID_DIR=root / 'valid', WORK_DIR=root / 'work', BACKUP_DIR=backup, DARKNET=darknet)
            with self.assertRaisesRegex(RuntimeError, 'differs from this run'): exec(preflight, env)
            self.assertEqual((backup / 'obj.names').read_text(), 'different class\n')

if __name__ == '__main__':
    unittest.main()
