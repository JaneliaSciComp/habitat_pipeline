"""Tests for database/dataset_fingerprint.py.

The fingerprint is metadata-only by design (see the module docstring): hashing
gigabytes of Kilosort output over SMB to catch a case that `size` already
catches would cost minutes per run. These tests pin the properties that matter
for that choice — it changes when the data changes, it is stable when nothing
does, and it never quietly claims to have fingerprinted files it could not
read.
"""
import os
import time

import pytest

from database.dataset_fingerprint import (
    FINGERPRINT_METHOD,
    describe_inputs,
    fingerprint_inputs,
    fingerprint_session,
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'kilosort4'
    root.mkdir()
    (root / 'spike_times.npy').write_bytes(b'x' * 100)
    (root / 'spike_clusters.npy').write_bytes(b'y' * 200)
    events = tmp_path / 'events.csv'
    events.write_text('a,b\n1,2\n', encoding='utf-8')
    return root, events


class TestStability:
    def test_stable_across_calls(self, dataset):
        root, events = dataset
        first, method = fingerprint_inputs([root, events])
        second, _ = fingerprint_inputs([root, events])
        assert first == second
        assert method == FINGERPRINT_METHOD

    def test_input_order_does_not_matter(self, dataset):
        root, events = dataset
        assert fingerprint_inputs([root, events])[0] == \
               fingerprint_inputs([events, root])[0]

    def test_mount_point_does_not_matter(self, tmp_path):
        """Names are relative to the input root, so remounting is not a change."""
        digests = []
        for parent in ('mount_a', 'mount_b'):
            root = tmp_path / parent / 'kilosort4'
            root.mkdir(parents=True)
            (root / 'spike_times.npy').write_bytes(b'x' * 100)
            os.utime(root / 'spike_times.npy', (1_700_000_000, 1_700_000_000))
            digests.append(fingerprint_inputs([root])[0])
        assert digests[0] == digests[1]


class TestSensitivity:
    def test_changes_when_a_file_grows(self, dataset):
        root, events = dataset
        before, _ = fingerprint_inputs([root, events])
        (root / 'spike_times.npy').write_bytes(b'x' * 101)
        after, _ = fingerprint_inputs([root, events])
        assert before != after

    def test_changes_when_mtime_changes(self, dataset):
        """Catches a re-export that happens to produce the same byte count."""
        root, _ = dataset
        target = root / 'spike_times.npy'
        os.utime(target, (1_700_000_000, 1_700_000_000))
        before, _ = fingerprint_inputs([root])
        os.utime(target, (1_800_000_000, 1_800_000_000))
        after, _ = fingerprint_inputs([root])
        assert before != after

    def test_changes_when_a_file_is_added(self, dataset):
        root, events = dataset
        before, _ = fingerprint_inputs([root, events])
        (root / 'amplitudes.npy').write_bytes(b'z' * 10)
        assert fingerprint_inputs([root, events])[0] != before

    def test_different_datasets_differ(self, tmp_path):
        first = tmp_path / 'a'
        first.mkdir()
        (first / 'f.npy').write_bytes(b'a' * 10)
        second = tmp_path / 'b'
        second.mkdir()
        (second / 'f.npy').write_bytes(b'b' * 20)
        assert fingerprint_inputs([first])[0] != fingerprint_inputs([second])[0]


class TestHonestAboutWhatItCouldNotRead:
    def test_nothing_readable_returns_none_not_a_hash_of_emptiness(self, tmp_path):
        """A hash of the empty set would compare equal across unrelated datasets."""
        digest, method = fingerprint_inputs([tmp_path / 'does_not_exist'])
        assert digest is None
        assert 'no-readable-inputs' in method

    def test_a_missing_path_marks_the_method_partial(self, dataset, tmp_path):
        root, _ = dataset
        digest, method = fingerprint_inputs([root, tmp_path / 'gone.csv'])
        assert digest is not None
        assert method.startswith(FINGERPRINT_METHOD)
        assert 'partial:1' in method

    def test_a_complete_read_is_not_marked_partial(self, dataset):
        root, events = dataset
        _, method = fingerprint_inputs([root, events])
        assert 'partial' not in method

    def test_describe_reports_the_problems(self, dataset, tmp_path):
        root, _ = dataset
        entries, problems = describe_inputs([root, tmp_path / 'nope'])
        assert entries
        assert any('missing' in p for p in problems)

    def test_none_entries_are_skipped(self, dataset):
        root, events = dataset
        assert fingerprint_inputs([root, None, events])[0] == \
               fingerprint_inputs([root, events])[0]

    def test_method_string_is_versioned(self):
        """So a future upgrade to content hashing can't be compared against v1."""
        assert FINGERPRINT_METHOD.endswith('/v1')


class TestCost:
    def test_does_not_read_file_contents(self, tmp_path, monkeypatch):
        """The whole point: metadata only, never bytes."""
        root = tmp_path / 'ks'
        root.mkdir()
        big = root / 'spike_times.npy'
        big.write_bytes(b'x' * 1000)

        import pathlib
        original = pathlib.Path.read_bytes

        def _boom(self, *args, **kwargs):
            raise AssertionError(f'fingerprinting read the contents of {self}')

        monkeypatch.setattr(pathlib.Path, 'read_bytes', _boom)
        digest, _ = fingerprint_inputs([root])
        monkeypatch.setattr(pathlib.Path, 'read_bytes', original)
        assert digest is not None

    def test_file_count_is_bounded(self, tmp_path):
        root = tmp_path / 'many'
        root.mkdir()
        for i in range(50):
            (root / f'f{i:03d}.npy').write_bytes(b'x')
        entries, _ = describe_inputs([root], max_files=10)
        assert len(entries) == 10

    def test_ignores_irrelevant_suffixes(self, tmp_path):
        root = tmp_path / 'ks'
        root.mkdir()
        (root / 'real.npy').write_bytes(b'x')
        (root / 'noise.log').write_bytes(b'y')
        (root / 'stale.pyc').write_bytes(b'z')
        names = [name for name, _, _ in describe_inputs([root])[0]]
        assert names == ['real.npy']


class TestFingerprintSession:
    def test_accepts_explicit_paths(self, dataset):
        root, events = dataset
        digest, method = fingerprint_session(kilosort_path=root, event_paths=[events])
        assert digest
        assert method == FINGERPRINT_METHOD

    def test_a_raising_dsm_getter_does_not_break_logging(self, dataset):
        """An un-fingerprinted iteration is still worth recording."""
        root, _ = dataset

        class _Dsm:
            def get_kilosort_path(self):
                return root

            def get_tracking_files(self):
                raise FileNotFoundError('no tracking for this session')

            def get_behavioral_event_files(self):
                return None

        digest, _ = fingerprint_session(_Dsm())
        assert digest is not None

    def test_no_inputs_at_all_returns_none(self):
        assert fingerprint_session()[0] is None
