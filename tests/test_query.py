import pytest
import types

from sqlite_utils import Database
from sqlite_utils.utils import sqlite3


def test_query(fresh_db):
    fresh_db["dogs"].insert_all([{"name": "Cleo"}, {"name": "Pancakes"}])
    results = fresh_db.query("select * from dogs order by name desc")
    assert isinstance(results, types.GeneratorType)
    assert list(results) == [{"name": "Pancakes"}, {"name": "Cleo"}]


def test_query_executes_eagerly(fresh_db):
    # The SQL runs when query() is called, not when the result is iterated,
    # so errors are raised at the call site
    with pytest.raises(sqlite3.OperationalError):
        fresh_db.query("select * from missing_table")
    # A failed query does not leave an implicit transaction open
    assert not fresh_db.conn.in_transaction


def test_query_rejects_statements_that_return_no_rows(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    with pytest.raises(ValueError) as ex:
        fresh_db.query("update dogs set name = 'Cleopaws'")
    assert "execute()" in str(ex.value)
    # The rejected statement had no effect and left no transaction open
    assert list(fresh_db["dogs"].rows) == [{"name": "Cleo"}]
    assert not fresh_db.conn.in_transaction


def test_query_rejects_create_table_without_creating_it(fresh_db):
    with pytest.raises(ValueError):
        fresh_db.query("create table t (id integer primary key)")
    assert "t" not in fresh_db.table_names()
    assert not fresh_db.conn.in_transaction


@pytest.mark.parametrize(
    "sql",
    (
        "begin",
        "commit",
        "end",
        "rollback",
        "savepoint sp",
        "release sp",
        "vacuum",
        "attach database 'attached.db' as attached",
        "detach database attached",
    ),
)
def test_query_rejects_transaction_control_and_unsupported_statements(fresh_db, sql):
    with pytest.raises(ValueError) as ex:
        fresh_db.query(sql)
    assert "execute()" in str(ex.value)
    assert not fresh_db.conn.in_transaction


def test_query_rejects_statement_with_leading_comment(fresh_db):
    with pytest.raises(ValueError):
        fresh_db.query("/* comment */ begin")
    assert not fresh_db.conn.in_transaction


def test_query_rejected_statement_inside_transaction_keeps_transaction(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    fresh_db.begin()
    fresh_db.execute("insert into dogs (name) values ('Marnie')")
    with pytest.raises(ValueError):
        fresh_db.query("update dogs set name = 'Cleopaws'")
    # Only the rejected statement was undone - the transaction stays open
    assert fresh_db.conn.in_transaction
    fresh_db.commit()
    assert [r["name"] for r in fresh_db["dogs"].rows] == ["Cleo", "Marnie"]


def test_query_row_returning_pragma(tmpdir):
    path = str(tmpdir / "test.db")
    db = Database(path)
    assert list(db.query("pragma journal_mode = wal")) == [{"journal_mode": "wal"}]
    assert not db.conn.in_transaction


def test_query_failing_write_leaves_no_transaction(fresh_db):
    fresh_db["dogs"].insert({"id": 1}, pk="id")
    with pytest.raises(sqlite3.IntegrityError):
        fresh_db.query("insert into dogs (id) values (1)")
    assert not fresh_db.conn.in_transaction
    assert list(fresh_db["dogs"].rows) == [{"id": 1}]


def test_query_rejects_pragma_that_returns_no_rows(fresh_db):
    with pytest.raises(ValueError) as ex:
        fresh_db.query("pragma user_version = 5")
    assert "execute()" in str(ex.value)
    assert list(fresh_db.query("pragma user_version")) == [{"user_version": 0}]
    assert not fresh_db.conn.in_transaction


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)
def test_query_insert_returning(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    rows = list(
        fresh_db.query("insert into dogs (name) values ('Pancakes') returning name")
    )
    assert rows == [{"name": "Pancakes"}]
    assert fresh_db["dogs"].count == 2
    assert not fresh_db.conn.in_transaction


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)
def test_query_insert_returning_committed_without_iteration(tmpdir):
    path = str(tmpdir / "test.db")
    db = Database(path)
    db["dogs"].insert({"name": "Cleo"})
    db.query("insert into dogs (name) values ('Pancakes') returning name")
    # Committed before query() returned, even though the results were ignored
    assert not db.conn.in_transaction
    other = sqlite3.connect(path)
    assert other.execute("select name from dogs order by name").fetchall() == [
        ("Cleo",),
        ("Pancakes",),
    ]
    other.close()


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)
def test_query_insert_returning_committed_with_partial_iteration(tmpdir):
    path = str(tmpdir / "test.db")
    db = Database(path)
    db["dogs"].insert({"name": "Cleo"})
    rows = db.query("insert into dogs (name) values ('A'), ('B') returning name")
    assert next(rows) == {"name": "A"}
    # Already committed, and the remaining rows are still available
    assert not db.conn.in_transaction
    other = sqlite3.connect(path)
    assert other.execute("select name from dogs order by name").fetchall() == [
        ("A",),
        ("B",),
        ("Cleo",),
    ]
    other.close()
    assert list(rows) == [{"name": "B"}]


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)
def test_query_update_returning_committed(tmpdir):
    path = str(tmpdir / "test.db")
    db = Database(path)
    db["dogs"].insert({"name": "Cleo"})
    assert list(db.query("update dogs set name = 'Cleopaws' returning name")) == [
        {"name": "Cleopaws"}
    ]
    assert not db.conn.in_transaction
    other = sqlite3.connect(path)
    assert other.execute("select name from dogs").fetchall() == [("Cleopaws",)]
    other.close()


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)
def test_query_returning_inside_transaction_joins_transaction(tmpdir):
    path = str(tmpdir / "test.db")
    db = Database(path)
    db["dogs"].insert({"name": "Cleo"})
    db.begin()
    rows = list(db.query("insert into dogs (name) values ('Pancakes') returning name"))
    assert rows == [{"name": "Pancakes"}]
    # The write belongs to the explicit transaction, so rollback undoes it
    assert db.conn.in_transaction
    db.rollback()
    assert [r["name"] for r in db["dogs"].rows] == ["Cleo"]


def test_execute_returning_dicts(fresh_db):
    # Like db.query() but returns a list, included for backwards compatibility
    # see https://github.com/simonw/sqlite-utils/issues/290
    fresh_db["test"].insert({"id": 1, "bar": 2}, pk="id")
    assert fresh_db.execute_returning_dicts("select * from test") == [
        {"id": 1, "bar": 2}
    ]
