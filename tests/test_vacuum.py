from sqlite_utils import Database
from sqlite_utils.db import TransactionError
import pathlib
import pytest


def _populate(db):
    db["dogs"].insert_all(
        [{"id": 1, "name": "Cleo"}, {"id": 2, "name": "Pancakes"}], pk="id"
    )
    db["dogs"].create_index(["name"])
    db.create_view("dog_names", "select name from dogs")
    db.execute("create trigger dogs_ai after insert on dogs begin select 1; end")


def _schema_and_rows(db):
    return db.schema, list(db["dogs"].rows)


@pytest.mark.parametrize("memory", (False, True))
@pytest.mark.parametrize("as_path", (False, True))
def test_vacuum_into(tmpdir, memory, as_path):
    db = Database(memory=True) if memory else Database(str(tmpdir / "source.db"))
    _populate(db)
    before = _schema_and_rows(db)
    into = pathlib.Path(tmpdir) / "it's a backup.db"
    assert db.vacuum(into=into if as_path else str(into)) is None
    assert into.exists()
    copy = Database(into)
    assert _schema_and_rows(copy) == before
    assert copy["dogs"].indexes[0].columns == ["name"]
    assert copy.view_names() == ["dog_names"]
    assert [t.name for t in copy.triggers] == ["dogs_ai"]
    # Source is unchanged and still usable
    assert _schema_and_rows(db) == before
    db["dogs"].insert({"id": 3, "name": "Rex"})
    assert db["dogs"].count == 3
    assert copy["dogs"].count == 2


@pytest.mark.parametrize("contents", (b"", b"existing"))
def test_vacuum_into_existing_file(tmpdir, contents):
    db = Database(memory=True)
    _populate(db)
    into = pathlib.Path(tmpdir) / "backup.db"
    into.write_bytes(contents)
    traced = []
    with db.tracer(lambda sql, params: traced.append(sql)):
        with pytest.raises(FileExistsError, match="already exists"):
            db.vacuum(into=into)
    assert traced == []
    assert into.read_bytes() == contents


@pytest.mark.parametrize("use_atomic", (False, True))
def test_vacuum_into_in_transaction(tmpdir, use_atomic):
    db = Database(str(tmpdir / "source.db"))
    _populate(db)
    into = pathlib.Path(tmpdir) / "backup.db"

    def attempt():
        db["dogs"].insert({"id": 3, "name": "Rex"})
        with pytest.raises(TransactionError):
            db.vacuum(into=into)
        assert db.conn.in_transaction
        assert db["dogs"].count == 3

    if use_atomic:
        with pytest.raises(ZeroDivisionError):
            with db.atomic():
                attempt()
                1 / 0
    else:
        db.begin()
        attempt()
        db.rollback()
    assert not into.exists()
    # Uncommitted insert was neither committed nor rolled back by vacuum()
    assert db["dogs"].count == 2


def test_vacuum_into_tracer(tmpdir):
    traced = []
    db = Database(memory=True, tracer=lambda sql, params: traced.append((sql, params)))
    _populate(db)
    traced.clear()
    into = str(tmpdir / "backup.db")
    db.vacuum(into=into)
    assert traced == [("VACUUM INTO ?;", [into])]


def test_vacuum_without_into(tmpdir):
    traced = []
    db = Database(str(tmpdir / "source.db"))
    _populate(db)
    with db.tracer(lambda sql, params: traced.append(sql)):
        db.vacuum()
    assert traced == ["VACUUM;"]
