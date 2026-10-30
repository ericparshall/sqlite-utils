import pytest
import types

from sqlite_utils import Database
from sqlite_utils.utils import sqlite3

requires_returning = pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)


def test_query(fresh_db):
    fresh_db["dogs"].insert_all([{"name": "Cleo"}, {"name": "Pancakes"}])
    results = fresh_db.query("select * from dogs order by name desc")
    assert isinstance(results, types.GeneratorType)
    assert list(results) == [{"name": "Pancakes"}, {"name": "Cleo"}]
    assert not fresh_db.conn.in_transaction


def test_query_executes_eagerly(fresh_db):
    # The SQL runs when query() is called, not when the result is iterated,
    # so errors are raised at the call site
    with pytest.raises(sqlite3.OperationalError):
        fresh_db.query("select * from missing_table")
    assert not fresh_db.conn.in_transaction


def test_query_rejects_statements_that_return_no_rows(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    with pytest.raises(ValueError) as ex:
        fresh_db.query("update dogs set name = 'Cleopaws'")
    assert "execute()" in str(ex.value)
    # The rejected statement was rolled back, and no transaction is left open
    assert list(fresh_db["dogs"].rows) == [{"name": "Cleo"}]
    assert not fresh_db.conn.in_transaction


def test_query_rejected_create_table_is_rolled_back(fresh_db):
    with pytest.raises(ValueError):
        fresh_db.query("create table t (id integer primary key)")
    assert fresh_db.table_names() == []
    assert not fresh_db.conn.in_transaction


def test_query_rejected_statement_inside_transaction_only_undoes_itself(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    fresh_db.begin()
    fresh_db["dogs"].insert({"name": "Pancakes"})
    with pytest.raises(ValueError):
        fresh_db.query("delete from dogs")
    # The transaction is still open and the earlier write survives
    assert fresh_db.conn.in_transaction
    fresh_db.commit()
    assert not fresh_db.conn.in_transaction
    assert list(fresh_db["dogs"].rows) == [{"name": "Cleo"}, {"name": "Pancakes"}]


@pytest.mark.parametrize(
    "sql",
    (
        "begin",
        "  BEGIN TRANSACTION",
        "commit",
        "end",
        "rollback",
        "savepoint sp",
        "release sp",
        "vacuum",
        "attach database ':memory:' as other",
        "detach database other",
    ),
)
def test_query_rejects_transaction_control_vacuum_attach_detach(fresh_db, sql):
    with pytest.raises(ValueError) as ex:
        fresh_db.query(sql)
    assert "execute()" in str(ex.value)
    assert not fresh_db.conn.in_transaction


def test_query_pragma_returning_rows(tmpdir):
    db = Database(str(tmpdir / "test.db"))
    # Changing to WAL mode is not allowed inside a transaction, so this
    # checks that PRAGMA statements are not wrapped in one
    assert list(db.query("pragma journal_mode = wal")) == [{"journal_mode": "wal"}]
    assert not db.conn.in_transaction
    db.close()


def test_query_pragma_returning_no_rows(fresh_db):
    with pytest.raises(ValueError) as ex:
        fresh_db.query("pragma user_version = 5")
    assert "execute()" in str(ex.value)
    assert not fresh_db.conn.in_transaction


@requires_returning
def test_query_insert_returning(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    rows = list(
        fresh_db.query("insert into dogs (name) values ('Pancakes') returning name")
    )
    assert rows == [{"name": "Pancakes"}]
    assert fresh_db["dogs"].count == 2
    assert not fresh_db.conn.in_transaction


@requires_returning
def test_query_update_and_delete_returning(fresh_db):
    fresh_db["dogs"].insert_all([{"id": 1, "name": "Cleo"}, {"id": 2}], pk="id")
    assert list(
        fresh_db.query("update dogs set name = 'Pancakes' where id = 2 returning *")
    ) == [{"id": 2, "name": "Pancakes"}]
    assert list(fresh_db.query("delete from dogs where id = 1 returning id")) == [
        {"id": 1}
    ]
    assert list(fresh_db["dogs"].rows) == [{"id": 2, "name": "Pancakes"}]
    assert not fresh_db.conn.in_transaction


@requires_returning
@pytest.mark.parametrize("iterate", ("none", "partial", "full"))
def test_query_returning_commits_before_iteration(tmpdir, iterate):
    path = str(tmpdir / "test.db")
    db = Database(path)
    db["dogs"].insert({"name": "Cleo"})
    rows = db.query("insert into dogs (name) values ('A'), ('B') returning name")
    if iterate == "partial":
        assert next(rows) == {"name": "A"}
    elif iterate == "full":
        assert list(rows) == [{"name": "A"}, {"name": "B"}]
    # The write is committed whether or not the rows were consumed
    assert not db.conn.in_transaction
    other = sqlite3.connect(path)
    assert other.execute("select count(*) from dogs").fetchone()[0] == 3
    other.close()
    db.close()


@requires_returning
def test_query_returning_inside_transaction_is_not_committed(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    fresh_db.begin()
    rows = fresh_db.query("insert into dogs (name) values ('Pancakes') returning name")
    assert fresh_db.conn.in_transaction
    fresh_db.rollback()
    # The rows are still available, but the write was undone
    assert list(rows) == [{"name": "Pancakes"}]
    assert list(fresh_db["dogs"].rows) == [{"name": "Cleo"}]


@requires_returning
def test_query_returning_inside_transaction_can_be_committed(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    fresh_db.begin()
    fresh_db.query("insert into dogs (name) values ('Pancakes') returning name")
    fresh_db.commit()
    assert list(fresh_db["dogs"].rows) == [{"name": "Cleo"}, {"name": "Pancakes"}]


def test_execute_returning_dicts(fresh_db):
    # Like db.query() but returns a list, included for backwards compatibility
    # see https://github.com/simonw/sqlite-utils/issues/290
    fresh_db["test"].insert({"id": 1, "bar": 2}, pk="id")
    assert fresh_db.execute_returning_dicts("select * from test") == [
        {"id": 1, "bar": 2}
    ]
