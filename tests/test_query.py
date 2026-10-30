import pytest
import types

from sqlite_utils import Database
from sqlite_utils.utils import sqlite3


def test_query(fresh_db):
    fresh_db["dogs"].insert_all([{"name": "Cleo"}, {"name": "Pancakes"}])
    results = fresh_db.query("select * from dogs order by name desc")
    assert isinstance(results, types.GeneratorType)
    assert list(results) == [{"name": "Pancakes"}, {"name": "Cleo"}]


requires_returning = pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)


@pytest.fixture
def file_db(tmp_path):
    path = str(tmp_path / "data.db")
    db = Database(path)
    db["dogs"].insert({"name": "Cleo"})
    return db, path


def other_connection_names(path):
    conn = sqlite3.connect(path)
    try:
        return [row[0] for row in conn.execute("select name from dogs order by rowid")]
    finally:
        conn.close()


def test_query_executes_eagerly(fresh_db):
    # The SQL runs when query() is called, not when the result is iterated,
    # so errors are raised at the call site
    with pytest.raises(sqlite3.OperationalError):
        fresh_db.query("select * from missing_table")
    assert not fresh_db.conn.in_transaction


def test_query_select_is_lazy(fresh_db):
    fresh_db["dogs"].insert_all([{"name": "Cleo"}, {"name": "Pancakes"}])
    results = fresh_db.query("select name from dogs order by rowid")
    assert next(results) == {"name": "Cleo"}
    assert not fresh_db.conn.in_transaction
    assert list(results) == [{"name": "Pancakes"}]


def test_query_rejects_statements_that_return_no_rows(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    with pytest.raises(ValueError) as ex:
        fresh_db.query("update dogs set name = 'Cleopaws'")
    assert "execute()" in str(ex.value)


def test_query_rejected_update_has_no_effect(file_db):
    db, path = file_db
    with pytest.raises(ValueError, match=r"execute\(\)"):
        db.query("update dogs set name = 'Cleopaws'")
    assert not db.conn.in_transaction
    assert list(db["dogs"].rows) == [{"name": "Cleo"}]
    assert other_connection_names(path) == ["Cleo"]


def test_query_rejected_create_table_has_no_effect(fresh_db):
    with pytest.raises(ValueError, match=r"execute\(\)"):
        fresh_db.query("create table t (id integer primary key)")
    assert not fresh_db.conn.in_transaction
    assert fresh_db.table_names() == []


def test_query_rejected_pragma_has_no_effect(fresh_db):
    with pytest.raises(ValueError, match=r"execute\(\)"):
        fresh_db.query("pragma user_version = 5")
    assert not fresh_db.conn.in_transaction
    assert fresh_db.execute("pragma user_version").fetchone()[0] == 0


@pytest.mark.parametrize(
    "sql",
    [
        "begin",
        "BEGIN IMMEDIATE",
        "commit",
        "end",
        "rollback",
        "savepoint foo",
        "release foo",
        "vacuum",
        "attach database ':memory:' as foo",
        "detach database foo",
        "  -- comment\n/* another */ begin",
    ],
)
def test_query_rejects_transaction_control_and_similar(fresh_db, sql):
    with pytest.raises(ValueError, match=r"execute\(\)"):
        fresh_db.query(sql)
    assert not fresh_db.conn.in_transaction
    assert [row[1] for row in fresh_db.execute("pragma database_list")] == ["main"]


def test_query_rejected_statement_inside_transaction(file_db):
    db, path = file_db
    db.begin()
    db.execute("insert into dogs (name) values ('Pancakes')")
    with pytest.raises(ValueError):
        db.query("delete from dogs")
    with pytest.raises(ValueError):
        db.query("commit")
    with pytest.raises(sqlite3.OperationalError):
        db.query("select * from missing_table")
    # Only the rejected statements were undone - the transaction is still open
    assert db.conn.in_transaction
    assert other_connection_names(path) == ["Cleo"]
    db.commit()
    assert other_connection_names(path) == ["Cleo", "Pancakes"]


def test_query_pragma_returning_rows(file_db):
    db, _ = file_db
    assert list(db.query("pragma journal_mode = wal")) == [{"journal_mode": "wal"}]
    assert db.journal_mode == "wal"
    assert [row["name"] for row in db.query("pragma table_info(dogs)")] == ["name"]
    assert list(db.query("pragma user_version")) == [{"user_version": 0}]
    assert not db.conn.in_transaction


@requires_returning
def test_query_insert_returning(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    rows = list(
        fresh_db.query("insert into dogs (name) values ('Pancakes') returning name")
    )
    assert rows == [{"name": "Pancakes"}]
    assert fresh_db["dogs"].count == 2


@requires_returning
def test_query_insert_returning_committed_without_iterating(file_db):
    db, path = file_db
    db.query("insert into dogs (name) values ('Pancakes') returning name")
    assert not db.conn.in_transaction
    assert other_connection_names(path) == ["Cleo", "Pancakes"]


@requires_returning
def test_query_insert_returning_committed_when_partially_iterated(file_db):
    db, path = file_db
    results = db.query("insert into dogs (name) values ('A'), ('B') returning name")
    assert next(results) == {"name": "A"}
    assert not db.conn.in_transaction
    assert other_connection_names(path) == ["Cleo", "A", "B"]
    assert list(results) == [{"name": "B"}]


@requires_returning
@pytest.mark.parametrize(
    "sql,expected",
    [
        ("update dogs set name = 'Cleopaws' returning name", ["Cleopaws"]),
        ("delete from dogs returning name", []),
    ],
)
def test_query_update_delete_returning_committed(file_db, sql, expected):
    db, path = file_db
    db.query(sql)
    assert not db.conn.in_transaction
    assert other_connection_names(path) == expected


@requires_returning
def test_query_returning_joins_open_transaction(file_db):
    db, path = file_db
    db.begin()
    rows = list(db.query("insert into dogs (name) values ('Pancakes') returning name"))
    assert rows == [{"name": "Pancakes"}]
    assert db.conn.in_transaction
    assert other_connection_names(path) == ["Cleo"]
    db.rollback()
    assert [row["name"] for row in db["dogs"].rows] == ["Cleo"]


@requires_returning
def test_query_returning_inside_atomic(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    with pytest.raises(ZeroDivisionError):
        with fresh_db.atomic():
            fresh_db.query("delete from dogs returning name")
            1 / 0
    assert [row["name"] for row in fresh_db["dogs"].rows] == ["Cleo"]


def test_execute_returning_dicts(fresh_db):
    # Like db.query() but returns a list, included for backwards compatibility
    # see https://github.com/simonw/sqlite-utils/issues/290
    fresh_db["test"].insert({"id": 1, "bar": 2}, pk="id")
    assert fresh_db.execute_returning_dicts("select * from test") == [
        {"id": 1, "bar": 2}
    ]
