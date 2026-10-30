from sqlite_utils import Database
from sqlite_utils.db import TransactionError
import pathlib
import pytest


def _populate(db):
    db["dogs"].insert_all(
        [{"id": 1, "name": "Cleo", "age": 5}, {"id": 2, "name": "Pancakes", "age": 4}],
        pk="id",
    )
    db["dogs"].create_index(["name"])
    db.create_view("dog_names", "select name from dogs")
    db.executescript(
        "create trigger dogs_trigger after insert on dogs begin "
        "update dogs set age = 0 where id = new.id; end"
    )


def _schema_and_rows(db):
    return (
        set(db.execute("select type, name, sql from sqlite_master").fetchall()),
        db.execute("select * from dogs order by id").fetchall(),
    )


@pytest.mark.parametrize("memory", (True, False))
@pytest.mark.parametrize("use_pathlib", (True, False))
def test_vacuum_into(tmpdir, memory, use_pathlib):
    if memory:
        db = Database(memory=True)
    else:
        db = Database(str(tmpdir / "source.db"))
    _populate(db)
    expected = _schema_and_rows(db)
    into = tmpdir / "it's a backup.db"
    into = pathlib.Path(str(into)) if use_pathlib else str(into)
    assert db.vacuum(into=into) is None
    copy = Database(str(into))
    assert _schema_and_rows(copy) == expected
    # Source is unchanged and still usable
    assert _schema_and_rows(db) == expected
    db["dogs"].insert({"id": 3, "name": "Nibbles"})
    assert db["dogs"].count == 3
    assert copy["dogs"].count == 2


def test_vacuum_into_refuses_existing_file(tmpdir):
    db = Database(memory=True)
    db["dogs"].insert({"name": "Cleo"})
    into = tmpdir / "existing.db"
    into.write_binary(b"")
    with pytest.raises(FileExistsError) as ex:
        db.vacuum(into=str(into))
    assert "already exists" in str(ex.value)
    assert into.read_binary() == b""
    # Non-empty files are rejected too
    into.write_binary(b"not a database")
    with pytest.raises(FileExistsError):
        db.vacuum(into=str(into))
    assert into.read_binary() == b"not a database"


def test_vacuum_into_in_transaction_raises(tmpdir):
    db = Database(memory=True)
    db["dogs"].insert({"id": 1, "name": "Cleo"}, pk="id")
    into = tmpdir / "backup.db"
    db.begin()
    db["dogs"].insert({"id": 2, "name": "Pancakes"})
    with pytest.raises(TransactionError):
        db.vacuum(into=str(into))
    assert not into.exists()
    # Transaction is still open with its uncommitted changes intact
    assert db.conn.in_transaction
    assert db["dogs"].count == 2
    db.rollback()
    assert db["dogs"].count == 1


def test_vacuum_into_in_atomic_raises(tmpdir):
    db = Database(memory=True)
    db["dogs"].insert({"id": 1, "name": "Cleo"}, pk="id")
    into = tmpdir / "backup.db"
    with db.atomic():
        db["dogs"].insert({"id": 2, "name": "Pancakes"})
        with pytest.raises(TransactionError):
            db.vacuum(into=str(into))
        assert db.conn.in_transaction
    # atomic() still committed its own changes normally
    assert not into.exists()
    assert db["dogs"].count == 2


def test_vacuum_into_is_traced(tmpdir):
    collected = []
    db = Database(memory=True)
    db["dogs"].insert({"name": "Cleo"})
    into = str(tmpdir / "backup.db")
    with db.tracer(lambda sql, params: collected.append((sql, params))):
        db.vacuum(into=into)
    assert collected == [("VACUUM INTO ?", [into])]
