"""Tests for the lab notebook's retrofitted-column migration.

`Base.metadata.create_all` creates missing *tables* and never `ALTER`s an
existing one, so a `Column` added to `Iteration` after the database file
already exists lives in the ORM and not in the file. Every query touching it
then raises `OperationalError: no such column`.

`test_create_all_alone_does_not_add_columns` documents that behaviour
directly, so the reason `_ensure_added_columns` exists stays visible to
whoever reads this next and wonders why a plain `create_all` isn't enough.
"""
import sqlite3

import pytest
from sqlalchemy import create_engine, text

from database.database_core import Base
from database.lab_notebook import (
    _ADDED_COLUMNS,
    LabNotebook,
    _ensure_added_columns,
    normalize_animal_key,
    normalize_session_key,
)


def _columns(db_path, table):
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def _make_legacy_db(db_path):
    """Build a notebook, then drop the retrofitted columns to simulate the
    pre-migration file, keeping one row of real-looking data."""
    nb = LabNotebook(db_path)
    nb.log_iteration(
        'ephys.decode_location', {'null': 'shuffle', 'n_shuffles': 180},
        {'status': 'success', 'p_value': 0.0055}, animal_id='631', session_id='20251210',
    )
    nb.engine.dispose()

    # SQLite before 3.35 has no DROP COLUMN, so rebuild the table without the
    # retrofitted columns - the same shape the file had before they existed.
    conn = sqlite3.connect(str(db_path))
    try:
        retrofitted = {name for name, _ in _ADDED_COLUMNS['iterations']}
        cols = [r[1] for r in conn.execute("PRAGMA table_info(iterations)")]
        keep = [c for c in cols if c not in retrofitted]
        keep_csv = ', '.join(keep)
        conn.execute("CREATE TABLE iterations_old AS SELECT %s FROM iterations" % keep_csv)
        conn.execute("DROP TABLE iterations")
        conn.execute("ALTER TABLE iterations_old RENAME TO iterations")
        conn.commit()
    finally:
        conn.close()
    return keep


class TestEnsureAddedColumns:
    def test_migrates_a_legacy_database_preserving_rows(self, tmp_path):
        db = tmp_path / 'legacy.db'
        kept = _make_legacy_db(db)
        assert 'tier' not in _columns(db, 'iterations')

        nb = LabNotebook(db)
        assert 'iterations.tier' in nb.migrated_columns
        assert 'tier' in _columns(db, 'iterations')

        # The pre-existing row survived and its new columns read as unknown.
        rows = nb.iterations_for_session('20251210')
        assert len(rows) == 1
        assert rows[0].analysis_module == 'ephys.decode_location'
        assert rows[0].result_summary_dict()['p_value'] == 0.0055
        assert rows[0].tier is None
        assert rows[0].seed is None
        assert rows[0].dataset_fingerprint is None
        assert set(kept).issubset(_columns(db, 'iterations'))

    def test_is_idempotent(self, tmp_path):
        db = tmp_path / 'idem.db'
        _make_legacy_db(db)
        first = LabNotebook(db).migrated_columns
        second = LabNotebook(db).migrated_columns
        third = _ensure_added_columns(LabNotebook(db).engine)
        assert first, "first pass should have migrated something"
        assert second == []
        assert third == []

    def test_fresh_database_needs_no_migration(self, tmp_path):
        """create_all builds new tables complete, so this must be a no-op."""
        nb = LabNotebook(tmp_path / 'fresh.db')
        assert nb.migrated_columns == []
        for table, columns in _ADDED_COLUMNS.items():
            present = _columns(nb.db_path, table)
            for name, _ in columns:
                assert name in present, f"{table}.{name} missing from a fresh database"

    def test_create_all_alone_does_not_add_columns(self, tmp_path):
        """The documented reason this migration helper has to exist."""
        db = tmp_path / 'plain.db'
        _make_legacy_db(db)
        assert 'tier' not in _columns(db, 'iterations')

        engine = create_engine(f'sqlite:///{db}')
        Base.metadata.create_all(bind=engine)
        engine.dispose()

        assert 'tier' not in _columns(db, 'iterations'), (
            "create_all silently did not add the column - which is exactly why "
            "_ensure_added_columns exists"
        )

    def test_added_columns_are_all_nullable(self, tmp_path):
        """A NULL legacy row means 'not recorded', which is the honest value.

        A NOT NULL DEFAULT would retroactively assert something about the 12
        pre-existing iterations that nobody actually checked.
        """
        nb = LabNotebook(tmp_path / 'nullable.db')
        conn = sqlite3.connect(str(nb.db_path))
        try:
            for table, columns in _ADDED_COLUMNS.items():
                info = {r[1]: r for r in conn.execute(f"PRAGMA table_info({table})")}
                for name, _ in columns:
                    notnull, default = info[name][3], info[name][4]
                    assert notnull == 0, f"{table}.{name} is NOT NULL"
                    assert default is None, f"{table}.{name} has a default"
        finally:
            conn.close()

    def test_missing_table_is_skipped_not_an_error(self, tmp_path):
        """A database with none of these tables must not raise."""
        db = tmp_path / 'empty.db'
        engine = create_engine(f'sqlite:///{db}')
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE unrelated (id INTEGER)"))
        assert _ensure_added_columns(engine) == []


class TestNormalizeSessionKey:
    @pytest.mark.parametrize('session_id', [
        '20251210',
        'RatCity_20251210_1359_40Hz',
        'RatCity_20251210_1359_40Hz.rec',
        '20251210_094334',
        '/mnt/share/RatCity_20251210_1359_40Hz.rec/rat631',
    ])
    def test_every_form_of_the_same_session_agrees(self, session_id):
        """All of these name one recording in different parts of the codebase."""
        assert normalize_session_key(session_id) == '20251210'

    @pytest.mark.parametrize('bad', ['2025', '', 'rat631', 'no-date-here', None])
    def test_unresolvable_returns_none(self, bad):
        assert normalize_session_key(bad) is None

    def test_distinct_sessions_do_not_collide(self):
        assert normalize_session_key('20251210') != normalize_session_key('20251216')


class TestNormalizeAnimalKey:
    @pytest.mark.parametrize('animal_id', ['631', 'rat631', 631, 'Rat631'])
    def test_every_form_of_the_same_animal_agrees(self, animal_id):
        assert normalize_animal_key(animal_id) == '631'

    def test_none_is_none(self):
        assert normalize_animal_key(None) is None

    def test_distinct_animals_do_not_collide(self):
        assert normalize_animal_key('rat631') != normalize_animal_key('rat613')
