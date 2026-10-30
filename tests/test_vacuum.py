import pathlib

import pytest

from sqlite_utils import Database
from sqlite_utils.db import TransactionError


def _create_source(db):
    db["dogs"].insert_all(
        [
            {"id": 1, "name": "Cleo", "good": 1},
            {"id": 2, "name": "Pancakes", "good": 0},
        ],
        pk="id",
    )
    db["dogs"].create_index(["name"])
    db.create_view("good_dogs", "select * from dogs where good = 1")
    db.execute("""
        CREATE TRIGGER dogs_count
        AFTER INSERT ON dogs
        BEGIN
            UPDATE dogs SET good = good WHERE id = NEW.id;
        END;
        """)


def test_vacuum_into_file(tmpdir):
    source = Database(str(tmpdir / "source.db"))
    _create_source(source)
    destination = tmpdir / "backup.db"

    assert source.vacuum(into=destination) is None
    assert destination.exists()

    copy = Database(str(destination))
    assert copy.schema == source.schema
    assert list(copy["dogs"].rows) == list(source["dogs"].rows)

    # The source is left unchanged and remains usable
    source["dogs"].insert({"id": 3, "name": "Suna", "good": 1})
    assert source["dogs"].count == 3
    assert copy["dogs"].count == 2


def test_vacuum_into_memory(tmpdir):
    source = Database(memory=True)
    _create_source(source)
    destination = str(tmpdir / "backup.db")

    source.vacuum(into=destination)
    copy = Database(str(destination))
    assert copy.schema == source.schema
    assert list(copy["dogs"].rows) == list(source["dogs"].rows)


def test_vacuum_into_path_with_spaces_and_quotes(tmpdir):
    source = Database(memory=True)
    _create_source(source)
    destination = tmpdir / "it's a backup.db"

    source.vacuum(into=destination)
    assert destination.exists()
    assert Database(str(destination))["dogs"].count == 2


def test_vacuum_into_existing_file(tmpdir):
    source = Database(memory=True)
    _create_source(source)
    destination = tmpdir / "backup.db"
    destination.write_binary(b"do not touch")

    with pytest.raises(FileExistsError):
        source.vacuum(into=pathlib.Path(destination))

    assert destination.read_binary() == b"do not touch"


def test_vacuum_into_existing_empty_file(tmpdir):
    source = Database(memory=True)
    _create_source(source)
    destination = tmpdir / "backup.db"
    destination.write_binary(b"")

    with pytest.raises(FileExistsError):
        source.vacuum(into=destination)

    assert destination.read_binary() == b""


def test_vacuum_into_inside_transaction_raises(tmpdir):
    source = Database(str(tmpdir / "source.db"))
    _create_source(source)
    destination = tmpdir / "backup.db"
    source.begin()
    source["dogs"].insert({"id": 3, "name": "Suna", "good": 1})

    with pytest.raises(TransactionError):
        source.vacuum(into=destination)

    assert not destination.exists()
    assert source.conn.in_transaction
    source.rollback()
    assert source["dogs"].count == 2


def test_vacuum_into_inside_atomic_raises(tmpdir):
    source = Database(memory=True)
    _create_source(source)
    destination = tmpdir / "backup.db"

    with pytest.raises(TransactionError):
        with source.atomic():
            source["dogs"].insert({"id": 3, "name": "Suna", "good": 1})
            source.vacuum(into=destination)

    assert not destination.exists()
    assert source["dogs"].count == 2


def test_vacuum_into_is_traced(tmpdir):
    source = Database(memory=True)
    _create_source(source)
    destination = tmpdir / "backup.db"
    collected = []

    with source.tracer(lambda sql, params: collected.append((sql, params))):
        source.vacuum(into=destination)

    assert any("VACUUM INTO" in sql for sql, _ in collected)
