from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import os
from scripts.sound.synthesis.snapshot_copy import materialize_snapshot


class SnapshotCopyTests(unittest.TestCase):
    def test_hard_link_cannot_import_an_outside_file(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'models--Example--Model'/'snapshots'/('a'*40)
            source.mkdir(parents=True)
            (source.parent.parent/'blobs').mkdir()
            outside = root/'outside.bin'
            outside.write_bytes(b'private-sentinel')
            os.link(outside, source/'weights.bin')
            with self.assertRaises(ValueError):
                materialize_snapshot(source, root/'output')
            self.assertFalse((root/'output').exists())

    def test_standard_blob_link_is_copied_but_escape_is_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'models--Example--Model'/'snapshots'/('a'*40)
            source.mkdir(parents=True)
            blobs = source.parent.parent/'blobs'
            blobs.mkdir()
            (blobs/'weights').write_bytes(b'model')
            link = source/'weights.bin'
            try:
                link.symlink_to(blobs/'weights')
            except OSError:
                self.skipTest('Symlink creation is unavailable')
            materialize_snapshot(source, root/'good')
            self.assertFalse((root/'good'/'weights.bin').is_symlink())
            self.assertEqual((root/'good'/'weights.bin').read_bytes(), b'model')
            link.unlink()
            (root/'outside').write_bytes(b'private-sentinel')
            link.symlink_to(root/'outside')
            with self.assertRaises(ValueError):
                materialize_snapshot(source, root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_regular_snapshot_is_copied_and_existing_destination_is_preserved(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root/'models--Example--Model'
            source = repo/'snapshots'/('a'*40)
            source.mkdir(parents=True)
            (repo/'blobs').mkdir()
            (source/'config.json').write_text('{}')
            destination = root/'materialized'
            materialize_snapshot(source, destination)
            self.assertEqual((destination/'config.json').read_text(), '{}')
            with self.assertRaises(ValueError):
                materialize_snapshot(source, destination)
            self.assertEqual((destination/'config.json').read_text(), '{}')

    def test_private_entries_are_rejected_before_copy(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'models--Example--Model'/'snapshots'/('a'*40)
            source.mkdir(parents=True)
            (source.parent.parent/'blobs').mkdir()
            (source/'.env').write_text('private-sentinel')
            with self.assertRaises(ValueError) as failure:
                materialize_snapshot(source, root/'output')
            self.assertNotIn('private-sentinel', str(failure.exception))
            self.assertFalse((root/'output').exists())
