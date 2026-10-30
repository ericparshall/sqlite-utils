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
    assert not fresh_db.conn.in_transaction


def test_query_rejects_statements_that_return_no_rows(fresh_db):
    fresh_db["dogs"].insert({"name": "Cleo"})
    with pytest.raises(ValueError) as ex:
        fresh_db.query("update dogs set name = 'Cleopaws'")
    assert "execute()" in str(ex.value)
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
        "attach database 'other.db' as other",
        "detach database other",
    ),
)
def test_query_rejects_transaction_and_attachment_statements(fresh_db, sql):
    with pytest.raises(ValueError) as ex:
        fresh_db.query(sql)
    assert "execute()" in str(ex.value)
    assert not fresh_db.conn.in_transaction


def test_query_rejected_statement_inside_transaction_only_undoes_that_statement(
    fresh_db,
):
    fresh_db["dogs"].insert({"name": "Cleo"})
    fresh_db.begin()
    fresh_db["dogs"].insert({"name": "Pancakes"})
    with pytest.raises(ValueError):
        fresh_db.query("update dogs set name = 'Cleopaws'")
    assert fresh_db.conn.in_transaction
    fresh_db.commit()
    assert sorted(row["name"] for row in fresh_db["dogs"].rows) == [
        "Cleo",
        "Pancakes",
    ]


def test_query_pragma_that_returns_rows(tmpdir):
    db = Database(str(tmpdir / "test.db"))
    assert list(db.query("pragma journal_mode = wal")) == [{"journal_mode": "wal"}]
    assert not db.conn.in_transaction


def test_query_pragma_that_returns_no_rows_is_rejected(tmpdir):
    db = Database(str(tmpdir / "test.db"))
    assert list(db.query("pragma user_version")) == [{"user_version": 0}]
    with pytest.raises(ValueError):
        db.query("pragma user_version = 5")
    assert list(db.query("pragma user_version")) == [{"user_version": 0}]
    assert not db.conn.in_transaction


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)
@pytest.mark.parametrize("consume", ("none", "one", "all"))
def test_query_insert_returning_is_committed(tmpdir, consume):
    path = str(tmpdir / "test.db")
    db = Database(path)
    db["dogs"].insert({"name": "Cleo"})
    results = db.query(
        "insert into dogs (name) values ('Pancakes'), ('Marnie') returning name"
    )
    if consume == "one":
        assert next(results) == {"name": "Pancakes"}
    elif consume == "all":
        assert list(results) == [{"name": "Pancakes"}, {"name": "Marnie"}]
    assert not db.conn.in_transaction
    other = sqlite3.connect(path)
    assert other.execute("select name from dogs order by name").fetchall() == [
        ("Cleo",),
        ("Marnie",),
        ("Pancakes",),
    ]
    other.close()


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)
def test_query_update_and_delete_returning_are_committed(tmpdir):
    path = str(tmpdir / "test.db")
    db = Database(path)
    db["dogs"].insert_all([{"name": "Cleo"}, {"name": "Pancakes"}])
    assert list(db.query("update dogs set name = upper(name) returning name")) == [
        {"name": "CLEO"},
        {"name": "PANCAKES"},
    ]
    assert not db.conn.in_transaction
    assert list(db.query("delete from dogs returning name")) == [
        {"name": "CLEO"},
        {"name": "PANCAKES"},
    ]
    assert not db.conn.in_transaction
    other = sqlite3.connect(path)
    assert other.execute("select count(*) from dogs").fetchone()[0] == 0
    other.close()


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0),
    reason="RETURNING requires SQLite 3.35.0 or higher",
)
def test_query_returning_joins_explicit_transaction(tmpdir):
    path = str(tmpdir / "test.db")
    db = Database(path)
    db["dogs"].insert({"name": "Cleo"})
    db.begin()
    assert list(
        db.query("insert into dogs (name) values ('Pancakes') returning name")
    ) == [{"name": "Pancakes"}]
    assert db.conn.in_transaction
    other = sqlite3.connect(path)
    assert other.execute("select count(*) from dogs").fetchone()[0] == 1
    other.close()
    db.rollback()
    assert list(db["dogs"].rows) == [{"name": "Cleo"}]


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


def test_execute_returning_dicts(fresh_db):
    # Like db.query() but returns a list, included for backwards compatibility
    # see https://github.com/simonw/sqlite-utils/issues/290
    fresh_db["test"].insert({"id": 1, "bar": 2}, pk="id")
    assert fresh_db.execute_returning_dicts("select * from test") == [
        {"id": 1, "bar": 2}
    ]
